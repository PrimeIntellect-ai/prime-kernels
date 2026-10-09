"""FP8 expert MLP: DeepGEMM's blockwise FP8 grouped GEMMs with fused quantization passes.

Numerics follow DeepSeek-V3: e4m3 values, fp32 scales, fp32 accumulation; activations and
gradients get one scale per 1 x 128 group along the GEMM's K dimension, weights one per
128 x 128 block. That holds for the forward, the data gradients and the weight gradients (whose
K dimension is the tokens, so their operands are quantized per 128 tokens of each channel).

DeepGEMM's grouped GEMMs pick each 128-row block's expert from the block's first row, so the
experts' rows must start at multiples of 128. Two layouts make that hold without copies:

- The *padded* layout: `TileTable` tile ``t`` (up to 128 rows of one expert) is padded rows
  ``[128 t, 128 t + 128)``. The quantization of ``x`` / ``dout`` writes into it, so the gate/up
  and dh GEMMs read it and write their outputs in it, which the SwiGLU passes read. The K-grouped
  weight-gradient operands use the same padded rows, transposed.
- The *token-major* layout with a head GEMM, for the down and dx GEMMs, whose outputs are what
  the caller sees: their operands are written at the tokens' own rows, so DeepGEMM writes
  ``out`` / ``dx`` in place. Only an expert's *head rows*, those before the first multiple of 128
  in its group, sit in a block that starts in the previous expert and come out wrong; the SwiGLU
  passes also copy them, at most 127 per expert, into one 128-row block per expert, a small
  second GEMM computes those, and they are scattered over the wrong ones.

Passes, forward: quantize ``x`` per row group and, for the weight gradient, per column group into
the K-grouped layout, from one read of ``x``; quantize the weights per block, also writing the
transposed copies the data gradients use; gate/up GEMM; clamped SwiGLU + quantization of ``h``;
down GEMM (+ head GEMM and scatter). Backward: quantize ``dout`` like ``x``; dh GEMM; SwiGLU
backward, writing FP8 ``dgate | dup`` per row group and per column group and FP8 ``h`` per column
group; dx GEMM (+ head GEMM and scatter); the two K-grouped weight-gradient GEMMs. The passes
over activation tiles are the Gluon kernels of `quant`.
"""

import torch
import triton
import triton.language as tl

from prime_kernels.moe_experts import quant
from prime_kernels.moe_experts.kernels import TileTable, zero_tail

GROUP = 128


def padded_rows(num_rows: int, num_experts: int) -> int:
    """Rows of the padded layout: every tile `TileTable` can produce, ``GROUP`` rows each."""
    return (triton.cdiv(num_rows, GROUP) + num_experts) * GROUP


def unsupported_shape_reason(hidden_size: int, intermediate_size: int) -> str | None:
    if hidden_size % GROUP or intermediate_size % GROUP:
        return f"FP8 needs hidden ({hidden_size}) and intermediate ({intermediate_size}) sizes that are multiples of {GROUP}"
    return None


@triton.jit
def _head_rows(expert_start, count):
    """How many of an expert's first rows lie before the first multiple of 128 in its group."""
    return tl.minimum((128 - expert_start % 128) % 128, count)


@triton.jit
def _layouts_kernel(
    padded_layout_ptr,
    token_layout_ptr,
    head_layout_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_row_end_ptr,
    expert_row_start_ptr,
    counts_ptr,
    num_tiles_ptr,
    max_tiles,
    num_experts,
):
    """DeepGEMM's per-row expert indices (-1: no expert) of the padded and the token-major layouts
    (from tile ``t``) and of the head operand (expert ``t``). Token-major rows after the last
    group must be filled with -1 beforehand. A tile's padding rows get its expert, so DeepGEMM
    computes them, from the zeros the quantization writes there: the padded GEMM outputs are then
    zeros in those rows rather than never written, and the SwiGLU backward needs no masks."""
    t = tl.program_id(0)
    rows = tl.arange(0, 128)
    if t < max_tiles:
        expert = tl.load(tile_expert_ptr + t)
        row_start = tl.load(tile_row_start_ptr + t)
        in_tile = t < tl.load(num_tiles_ptr)
        tl.store(padded_layout_ptr + t * 128 + rows, tl.where(in_tile, expert, -1))
        valid = in_tile & (rows < tl.load(tile_row_end_ptr + t) - row_start)
        tl.store(token_layout_ptr + row_start + rows, expert, mask=valid)
    if t < num_experts:
        head = _head_rows(tl.load(expert_row_start_ptr + t), tl.load(counts_ptr + t).to(tl.int32))
        tl.store(head_layout_ptr + t * 128 + rows, tl.where(rows < head, t, -1))


@triton.jit
def _scatter_head_kernel(head_out_ptr, out_ptr, expert_row_start_ptr, counts_ptr, N, BLOCK_N: tl.constexpr):
    """Copy expert ``e``'s head rows from the head GEMM's output over the token-major output."""
    e = tl.program_id(0)
    start = tl.load(expert_row_start_ptr + e)
    head = _head_rows(start, tl.load(counts_ptr + e).to(tl.int32))
    if head == 0:
        return
    rows = tl.arange(0, 128)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows < head)[:, None] & (cols < N)[None, :]
    v = tl.load(head_out_ptr + (e * 128 + rows).to(tl.int64)[:, None] * N + cols[None, :], mask=mask)
    tl.store(out_ptr + (start + rows).to(tl.int64)[:, None] * N + cols[None, :], v, mask=mask)


@triton.jit
def _transpose_fp8(q):
    # Through 16 bits (exact both ways): Triton 3.7 scrambles a quarter of the elements of 8-bit
    # transposes.
    return tl.trans(q.to(tl.float16)).to(tl.float8e4nv)


@triton.jit
def _quantize_weight_kernel(
    w_ptr,
    w2_ptr,
    q_ptr,
    sf_ptr,
    qt_ptr,
    sf_t_ptr,
    stride_e,
    stride_e2,
    R,
    C,
    SPLIT_BLOCKS,
    TRANSPOSED: tl.constexpr,
):
    """One 128 x 128 block of an ``[E, R, C]`` weight (row blocks from ``SPLIT_BLOCKS`` on read
    from ``w2``): FP8 values and block scale, and with ``TRANSPOSED`` the ``[E, C, R]`` copy."""
    e = tl.program_id(0).to(tl.int64)
    rb = tl.program_id(1)
    cb = tl.program_id(2)
    rows = tl.arange(0, 128)
    cols = cb * 128 + tl.arange(0, 128)
    if rb < SPLIT_BLOCKS:
        src = w_ptr + e * stride_e + (rb * 128 + rows)[:, None] * C + cols[None, :]
    else:
        src = w2_ptr + e * stride_e2 + ((rb - SPLIT_BLOCKS) * 128 + rows)[:, None] * C + cols[None, :]
    v = tl.load(src).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(v)), 1e-4) / 448.0
    q = (v * (1.0 / scale)).to(tl.float8e4nv)
    num_rb = R // 128
    num_cb = C // 128
    tl.store(q_ptr + e * R * C + (rb * 128 + rows)[:, None] * C + cols[None, :], q)
    tl.store(sf_ptr + (e * num_rb + rb) * num_cb + cb, scale)
    if TRANSPOSED:
        tl.store(qt_ptr + e * R * C + cols[:, None] * R + (rb * 128 + rows)[None, :], _transpose_fp8(q))
        tl.store(sf_t_ptr + (e * num_cb + cb) * num_rb + rb, scale)


@triton.jit
def _weight_grad_cast_kernel(d_ptr, a_ptr, b_ptr, SPLIT, REST, BLOCK: tl.constexpr):
    """bf16 of an fp32 weight gradient, ``SPLIT + REST`` elements per expert: the first ``SPLIT``
    into ``a``, the rest into ``b``."""
    e = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = SPLIT + REST
    mask = offsets < total
    v = tl.load(d_ptr + e * total + offsets, mask=mask).to(tl.bfloat16)
    tl.store(a_ptr + e * SPLIT + offsets, v, mask=offsets < SPLIT)
    tl.store(b_ptr + e * REST + offsets - SPLIT, v, mask=mask & (offsets >= SPLIT))


def quantize_weight(
    weight: torch.Tensor, weight2: torch.Tensor | None, transposed: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 ``[E, R, C]`` blocks of ``weight`` (stacked on top of ``weight2`` along R when given)
    and their scales, plus the ``[E, C, R]`` transposed copy when asked (empty otherwise)."""
    num_experts, rows, cols = weight.shape
    for w in (weight, weight2):
        if w is not None and (w.stride(2) != 1 or w.stride(1) != cols):
            raise ValueError(f"expert weights must have contiguous rows, got strides {w.stride()}")
    R = rows + (weight2.shape[1] if weight2 is not None else 0)
    q = weight.new_empty(num_experts, R, cols, dtype=torch.float8_e4m3fn)
    sf = weight.new_empty(num_experts, R // GROUP, cols // GROUP, dtype=torch.float32)
    if transposed:
        qt = weight.new_empty(num_experts, cols, R, dtype=torch.float8_e4m3fn)
        sf_t = weight.new_empty(num_experts, cols // GROUP, R // GROUP, dtype=torch.float32)
    else:
        qt = q.new_empty(0)
        sf_t = sf.new_empty(0)
    w2 = weight2 if weight2 is not None else weight
    _quantize_weight_kernel[(num_experts, R // GROUP, cols // GROUP)](
        weight,
        w2,
        q,
        sf,
        qt if transposed else q,
        sf_t if transposed else sf,
        weight.stride(0),
        w2.stride(0),
        R,
        cols,
        rows // GROUP if weight2 is not None else R // GROUP,
        TRANSPOSED=transposed,
        num_warps=8,
    )
    return q, sf, qt, sf_t


class Layout:
    """One call's groups: the `TileTable` of 128-row tiles and DeepGEMM's per-row expert indices
    of the padded layout, the token-major layout and the head operand (see the module doc)."""

    def __init__(self, num_tokens_per_expert: torch.Tensor, num_rows: int) -> None:
        device = num_tokens_per_expert.device
        num_experts = num_tokens_per_expert.numel()
        self.table = TileTable(num_tokens_per_expert, num_rows, GROUP)
        self.counts = num_tokens_per_expert
        self.num_rows = num_rows
        self.max_tiles = self.table.max_tiles
        self.Mp = self.max_tiles * GROUP
        self.M_head = num_experts * GROUP
        self.padded_layout = torch.empty(self.Mp, dtype=torch.int32, device=device)
        self.token_layout = torch.full((num_rows,), -1, dtype=torch.int32, device=device)
        self.head_layout = torch.empty(self.M_head, dtype=torch.int32, device=device)
        _layouts_kernel[(max(self.max_tiles, num_experts),)](
            self.padded_layout,
            self.token_layout,
            self.head_layout,
            *self.tile_args(),
            self.max_tiles,
            num_experts,
        )
        # K of each expert in the K-grouped GEMMs: its rows padded to whole groups.
        self.ks = ((num_tokens_per_expert + GROUP - 1) // GROUP * GROUP).to(torch.int32)

    def tile_args(self):
        t = self.table
        return t.expert, t.row_start, t.row_end, t.expert_row_start, self.counts, t.num_tiles

    def check_columns(self, cols: int) -> None:
        """The tile kernels address within an expert's K-grouped block with 32-bit offsets."""
        if self.Mp * cols >= 2**31:
            raise ValueError(f"too many rows ({self.num_rows}) for {cols} columns in the FP8 path")


def quantize_activation(src: torch.Tensor, layout: Layout, columns: bool):
    """``src`` [rows, C] per row group into the padded layout ([Mp, C] FP8, [C / 128, Mp] scales)
    and, with ``columns``, per column group into the K-grouped layout (flat FP8 and
    [Mp / 128, C] scales; empty without)."""
    C = src.shape[1]
    Mp = layout.Mp
    layout.check_columns(C)
    q = src.new_empty(Mp, C, dtype=torch.float8_e4m3fn)
    q_sf = src.new_empty(C // GROUP, Mp, dtype=torch.float32)
    if columns:
        qt = src.new_empty(Mp * C, dtype=torch.float8_e4m3fn)
        qt_sf = src.new_empty(Mp // GROUP, C, dtype=torch.float32)
    else:
        qt, qt_sf = q.new_empty(0), q_sf.new_empty(0, C)
    quant.quantize_kernel[(layout.max_tiles, C // GROUP)](
        quant.descriptor(src),
        q,
        q_sf,
        qt if columns else q,
        qt_sf if columns else q_sf,
        *layout.tile_args(),
        Mp,
        C,
        COLUMNS=columns,
        num_warps=quant.NUM_WARPS,
    )
    return q, q_sf, qt, qt_sf


def _head_operand(layout: Layout, cols: int, like: torch.Tensor):
    q = like.new_empty(layout.M_head, cols, dtype=torch.float8_e4m3fn)
    sf = like.new_empty(cols // GROUP, layout.M_head, dtype=torch.float32)
    return q, sf


def swiglu_quantize(gate_up: torch.Tensor, layout: Layout, limit: float):
    """FP8 h in the token-major layout and its head operand, each with MN-major scales."""
    inter = gate_up.shape[1] // 2
    M = layout.num_rows
    h = gate_up.new_empty(M, inter, dtype=torch.float8_e4m3fn)
    h_sf = gate_up.new_empty(inter // GROUP, M, dtype=torch.float32)
    head, head_sf = _head_operand(layout, inter, gate_up)
    quant.swiglu_kernel[(layout.max_tiles, inter // GROUP)](
        quant.descriptor(gate_up),
        h,
        h_sf,
        head,
        head_sf,
        *layout.tile_args(),
        M,
        layout.M_head,
        inter,
        limit,
        num_warps=quant.NUM_WARPS,
    )
    return h, h_sf, head, head_sf


def swiglu_backward_quantize(dh: torch.Tensor, gate_up: torch.Tensor, layout: Layout, limit: float):
    """FP8 dgate | dup in the token-major layout + head operand (for dx) and in the K-grouped
    layout, and FP8 h in the K-grouped layout."""
    Mp, inter = dh.shape
    M = layout.num_rows
    layout.check_columns(2 * inter)
    dgu = dh.new_empty(M, 2 * inter, dtype=torch.float8_e4m3fn)
    dgu_sf = dh.new_empty(2 * inter // GROUP, M, dtype=torch.float32)
    head, head_sf = _head_operand(layout, 2 * inter, dh)
    dgu_t = dh.new_empty(Mp * 2 * inter, dtype=torch.float8_e4m3fn)
    dgu_t_sf = dh.new_empty(Mp // GROUP, 2 * inter, dtype=torch.float32)
    h_t = dh.new_empty(Mp * inter, dtype=torch.float8_e4m3fn)
    h_t_sf = dh.new_empty(Mp // GROUP, inter, dtype=torch.float32)
    quant.swiglu_backward_kernel[(layout.max_tiles, inter // GROUP)](
        quant.descriptor(dh),
        quant.descriptor(gate_up),
        dgu,
        dgu_sf,
        head,
        head_sf,
        dgu_t,
        dgu_t_sf,
        h_t,
        h_t_sf,
        *layout.tile_args(),
        M,
        layout.M_head,
        inter,
        limit,
        num_warps=quant.NUM_WARPS,
    )
    return (dgu, dgu_sf, head, head_sf), (dgu_t, dgu_t_sf), (h_t, h_t_sf)


def _m_grouped(a, a_sf, b, b_sf, d, grouped_layout) -> None:
    import deep_gemm

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a, a_sf.T), (b, b_sf), d, grouped_layout)


def padded_gemm(a, a_sf, b, b_sf, layout: Layout, out_cols: int) -> torch.Tensor:
    """``[Mp, out_cols]`` bf16 of ``a @ b[e].T`` over the padded layout; ``a``'s scales MN-major."""
    d = torch.empty(layout.Mp, out_cols, device=a.device, dtype=torch.bfloat16)
    _m_grouped(a, a_sf, b, b_sf, d, layout.padded_layout)
    return d


def token_gemm(a, a_sf, head, head_sf, b, b_sf, layout: Layout, out_cols: int) -> torch.Tensor:
    """``[rows, out_cols]`` bf16 of ``a @ b[e].T`` over the token-major layout, the head rows
    from the head GEMM; rows after the last group are zeros."""
    out = torch.empty(layout.num_rows, out_cols, device=a.device, dtype=torch.bfloat16)
    _m_grouped(a, a_sf, b, b_sf, out, layout.token_layout)
    head_out = torch.empty(layout.M_head, out_cols, device=a.device, dtype=torch.bfloat16)
    _m_grouped(head, head_sf, b, b_sf, head_out, layout.head_layout)
    block_n = 256
    _scatter_head_kernel[(layout.counts.numel(), triton.cdiv(out_cols, block_n))](
        head_out, out, layout.table.expert_row_start, layout.counts, out_cols, BLOCK_N=block_n, num_warps=4
    )
    zero_tail(out, layout.counts, torch.cuda.get_device_properties(out.device).multi_processor_count)
    return out


def weight_grad_accumulate(a_t, a_t_sf, b_t, b_t_sf, layout: Layout, out: torch.Tensor) -> None:
    """``out[e] += A_e^T B_e`` in fp32 (``out`` is ``[E, rows, cols]``), straight from DeepGEMM's
    accumulator, so gradients accumulated over micro-batches need no cast or extra pass."""
    import deep_gemm

    if out.dtype != torch.float32 or not out.is_contiguous():
        raise ValueError(f"the accumulator must be contiguous fp32, got {out.dtype}")
    ks = [layout.Mp] + [0] * (layout.counts.numel() - 1)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous((a_t, a_t_sf.T), (b_t, b_t_sf.T), out, ks, layout.ks, out)


def weight_grad(a_t, a_t_sf, b_t, b_t_sf, layout: Layout, rows: int, cols: int, split: int | None = None):
    """bf16 ``A_e^T B_e`` (``[E, rows, cols]``) over each expert's tokens, from operands in the
    K-grouped layout; with ``split``, as its ``[E, split, cols]`` and ``[E, rows - split, cols]``
    halves (the second empty without)."""
    import deep_gemm

    num_experts = layout.counts.numel()
    # DeepGEMM accumulates into d (it requires that), and skips the experts without tokens.
    d = torch.zeros(num_experts, rows, cols, device=a_t.device, dtype=torch.float32)
    # The host K list only steers DeepGEMM's heuristics; the kernel walks the device ks. Passing
    # the padded total keeps the group sizes on the device (no synchronization).
    ks = [layout.Mp] + [0] * (num_experts - 1)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous((a_t, a_t_sf.T), (b_t, b_t_sf.T), d, ks, layout.ks, d)
    split = rows if split is None else split
    a = d.new_empty(num_experts, split, cols, dtype=torch.bfloat16)
    b = d.new_empty(num_experts, rows - split, cols, dtype=torch.bfloat16)
    block = 4096
    _weight_grad_cast_kernel[(num_experts, triton.cdiv(rows * cols, block))](
        d, a, b, split * cols, (rows - split) * cols, BLOCK=block, num_warps=8
    )
    return a, b
