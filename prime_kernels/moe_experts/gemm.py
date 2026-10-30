"""Token-major grouped GEMMs with fused SwiGLU epilogues, as Gluon kernels for Hopper.

Every kernel is persistent and warp specialized into two partitions:

- two consumer warpgroups (the default partition) run the K loop of one 128 x 256 output tile on
  the tensor cores, issuing the TMA loads of the K steps ``STAGES - 1`` ahead (across tile
  boundaries), and park the finished tile, as bf16, in a shared-memory epilogue buffer,
- an epilogue warpgroup applies the SwiGLU forward or backward to the parked tile and writes the
  results while the consumers already run the next tile's K loop.

The epilogue partition is what makes the fusion free: on Hopper the accumulator lives in the
consumers' registers, so without it the activation math and its extra global traffic would stall
the tensor cores. (A separate producer warp would need a fourth warpgroup, which caps every
thread at 128 registers, too few for a 64 x 256 wgmma accumulator.)

Row tiles come from a `TileTable`: each covers rows of one expert only, so a tile's last rows may
belong to the next expert. Operand rows past an expert's end are loaded anyway (they only feed
output rows that are never written), and weights are addressed as one 2D ``[experts * rows, cols]``
view, which is safe because a row past the end of one expert's weight either feeds an output
column that is never written or multiplies an activation column past ``K``, which TMA fills with
zeros.
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

BLOCK_M = 128

MODE_PLAIN = 0
MODE_FC1 = 1
MODE_DSWIGLU = 2


@gluon.jit
def _tile_coords(tile, num_m, num_n, GROUP_M: gl.constexpr):
    group_size = GROUP_M * num_n
    first_m = (tile // group_size) * GROUP_M
    group_m = gl.minimum(num_m - first_m, GROUP_M)
    m = first_m + (tile % group_size) % group_m
    n = (tile % group_size) // group_m
    return m, n


@gluon.jit
def _tile_operands(local, num_m, num_n, tile_expert_ptr, tile_row_start_ptr, W_ROWS, GROUP_M: gl.constexpr):
    """First weight row, first token row and column tile of this program's ``local``-th tile."""
    m, n = _tile_coords(gl.program_id(0) + local * gl.num_programs(0), num_m, num_n, GROUP_M)
    # Past the last tile (the loads only run ahead into it, nothing is issued) read a valid entry.
    m = gl.maximum(gl.minimum(m, num_m - 1), 0)
    return gl.load(tile_expert_ptr + m) * W_ROWS, gl.load(tile_row_start_ptr + m), n


@gluon.jit
def _issue_load(
    counter,
    step,
    n,
    w_row,
    row_start,
    k_tiles,
    a_desc,
    a2_desc,
    b_desc,
    b2_desc,
    a_smem,
    b_smem,
    ready,
    B2_ROW_OFFSET,
    MODE: gl.constexpr,
    B_N_MAJOR: gl.constexpr,
    DUAL_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    STAGES: gl.constexpr,
):
    """Load the operands of K step ``step`` of a tile into stage ``counter % STAGES``."""
    s = counter % STAGES
    bar = ready.index(s)
    mbarrier.expect(bar, a_desc.block_type.nbytes + BLOCK_N * BLOCK_K * 2)
    if DUAL_K:
        # dx = dgate @ gate_proj[e] + dup @ up_proj[e]: one K loop over both products.
        first = step < k_tiles
        second = step >= k_tiles
        k = (step % k_tiles) * BLOCK_K
        tma.async_copy_global_to_shared(a_desc, [row_start, k], bar, a_smem.index(s), pred=first)
        tma.async_copy_global_to_shared(a2_desc, [row_start, k], bar, a_smem.index(s), pred=second)
        tma.async_copy_global_to_shared(b_desc, [w_row + k, n * BLOCK_N], bar, b_smem.index(s), pred=first)
        tma.async_copy_global_to_shared(
            b2_desc, [w_row + B2_ROW_OFFSET + k, n * BLOCK_N], bar, b_smem.index(s), pred=second
        )
    else:
        k = step * BLOCK_K
        tma.async_copy_global_to_shared(a_desc, [row_start, k], bar, a_smem.index(s))
        if MODE == 1:
            # fc1: the gate rows and the up rows of the same intermediate columns side by side,
            # so a single 256-wide MMA computes both.
            HALF: gl.constexpr = BLOCK_N // 2
            stage = b_smem.index(s)
            tma.async_copy_global_to_shared(b_desc, [w_row + n * HALF, k], bar, stage.slice(0, HALF))
            tma.async_copy_global_to_shared(
                b2_desc, [w_row + B2_ROW_OFFSET + n * HALF, k], bar, stage.slice(HALF, HALF)
            )
        elif B_N_MAJOR:
            tma.async_copy_global_to_shared(b_desc, [w_row + n * BLOCK_N, k], bar, b_smem.index(s))
        else:
            tma.async_copy_global_to_shared(b_desc, [w_row + k, n * BLOCK_N], bar, b_smem.index(s))


@gluon.jit
def _consumer(
    a_desc,
    a2_desc,
    b_desc,
    b2_desc,
    a_smem,
    b_smem,
    epi_smem,
    ready,
    epi_full,
    epi_free,
    tile_expert_ptr,
    tile_row_start_ptr,
    num_tiles_ptr,
    N,
    K,
    W_ROWS,
    B2_ROW_OFFSET,
    MODE: gl.constexpr,
    B_N_MAJOR: gl.constexpr,
    DUAL_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    STAGES: gl.constexpr,
    GROUP_M: gl.constexpr,
):
    layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[gl.num_warps(), 1], instr_shape=[16, BLOCK_N, 16]
    )
    num_m = gl.load(num_tiles_ptr)
    num_n = gl.cdiv(N, BLOCK_N)
    k_tiles = gl.cdiv(K, BLOCK_K)
    steps = k_tiles * (2 if DUAL_K else 1)
    my_tiles = gl.cdiv(gl.maximum(num_m * num_n - gl.program_id(0), 0), gl.num_programs(0))
    total = my_tiles * steps

    # The loads run STAGES - 1 K steps ahead of the MMAs; (load_local, load_step) is the K step
    # the next load is for, and (w_row, row_start, n) the operands of its tile.
    load_step = 0
    load_local = 0
    w_row, row_start, n = _tile_operands(0, num_m, num_n, tile_expert_ptr, tile_row_start_ptr, W_ROWS, GROUP_M)
    for i in gl.static_range(STAGES - 1):
        if i < total:
            _issue_load(
                i,
                load_step,
                n,
                w_row,
                row_start,
                k_tiles,
                a_desc,
                a2_desc,
                b_desc,
                b2_desc,
                a_smem,
                b_smem,
                ready,
                B2_ROW_OFFSET,
                MODE,
                B_N_MAJOR,
                DUAL_K,
                BLOCK_N,
                BLOCK_K,
                STAGES,
            )
        load_step += 1
        if load_step == steps:
            load_step = 0
            load_local += 1
            w_row, row_start, n = _tile_operands(
                load_local, num_m, num_n, tile_expert_ptr, tile_row_start_ptr, W_ROWS, GROUP_M
            )

    counter = 0
    for local in range(my_tiles):
        acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout)
        for step in range(steps):
            s = counter % STAGES
            mbarrier.wait(ready.index(s), (counter // STAGES) & 1)
            if B_N_MAJOR:
                b = b_smem.index(s).permute((1, 0))
            else:
                b = b_smem.index(s)
            acc = warpgroup_mma(a_smem.index(s), b, acc, is_async=True)
            acc = warpgroup_mma_wait(num_outstanding=1, deps=(acc,))
            # The MMA of step counter - 1 has retired, so its stage takes step counter + STAGES - 1.
            if counter + STAGES - 1 < total:
                _issue_load(
                    counter + STAGES - 1,
                    load_step,
                    n,
                    w_row,
                    row_start,
                    k_tiles,
                    a_desc,
                    a2_desc,
                    b_desc,
                    b2_desc,
                    a_smem,
                    b_smem,
                    ready,
                    B2_ROW_OFFSET,
                    MODE,
                    B_N_MAJOR,
                    DUAL_K,
                    BLOCK_N,
                    BLOCK_K,
                    STAGES,
                )
            load_step += 1
            if load_step == steps:
                load_step = 0
                load_local += 1
                w_row, row_start, n = _tile_operands(
                    load_local, num_m, num_n, tile_expert_ptr, tile_row_start_ptr, W_ROWS, GROUP_M
                )
            counter += 1
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

        # Park the tile as bf16 (what the unfused GEMM would return) for the epilogue partition.
        mbarrier.wait(epi_free, (local & 1) ^ 1)
        epi_smem._reinterpret(gl.bfloat16, [BLOCK_M, BLOCK_N], epi_smem.layout).store(acc.to(gl.bfloat16))
        mbarrier.arrive(epi_full, count=1)


@gluon.jit
def _epilogue(
    epi_smem,
    epi_full,
    epi_free,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    in0_ptr,
    in1_ptr,
    tile_row_start_ptr,
    tile_row_end_ptr,
    num_tiles_ptr,
    N,
    OUT_STRIDE,
    limit,
    MODE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GROUP_M: gl.constexpr,
):
    """Finish each parked tile; only rows before the tile's ``row_end`` are written, the rest
    belongs to the next expert. Offsets inside a tile are 32-bit, only its base pointer is 64-bit.
    The row-chunk loops stay rolled: unrolled, this partition's code evicts the consumers' K loop
    from the instruction cache.
    """
    CHUNK: gl.constexpr = epi_smem.shape[1]
    HALF: gl.constexpr = BLOCK_N // 2
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [gl.num_warps(), 1], [1, 0])
    half_layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [gl.num_warps(), 1], [1, 0])
    num_m = gl.load(num_tiles_ptr)
    num_n = gl.cdiv(N, BLOCK_N)
    local = 0
    for tile in range(gl.program_id(0), num_m * num_n, gl.num_programs(0)):
        m, n = _tile_coords(tile, num_m, num_n, GROUP_M)
        row_start = gl.load(tile_row_start_ptr + m)
        tile_rows = gl.load(tile_row_end_ptr + m) - row_start
        mbarrier.wait(epi_full, local & 1)
        if MODE == 0:
            base = row_start.to(gl.int64) * N
            cols = n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout))
            for r in range(BLOCK_M // CHUNK):
                rows = r * CHUNK + gl.arange(0, CHUNK, layout=gl.SliceLayout(1, layout))
                mask = (rows < tile_rows)[:, None] & (cols < N)[None, :]
                value = epi_smem.index(r).load(layout)
                gl.store(out0_ptr + base + rows[:, None] * N + cols[None, :], value, mask=mask)
        elif MODE == 1:
            # fc1: the tile holds gate (first half) and up (second half) of the intermediate
            # columns n * HALF + [0, HALF).
            intermediate = N // 2
            base = row_start.to(gl.int64) * intermediate
            cols = n * HALF + gl.arange(0, HALF, layout=gl.SliceLayout(0, half_layout))
            for r in range(BLOCK_M // CHUNK):
                rows = r * CHUNK + gl.arange(0, CHUNK, layout=gl.SliceLayout(1, half_layout))
                mask = (rows < tile_rows)[:, None] & (cols < intermediate)[None, :]
                offsets = base + rows[:, None] * intermediate + cols[None, :]
                chunk = epi_smem.index(r)
                gate = chunk.slice(0, HALF, dim=1).load(half_layout)
                up = chunk.slice(HALF, HALF, dim=1).load(half_layout)
                gl.store(out0_ptr + offsets, gate, mask=mask)
                gl.store(out1_ptr + offsets, up, mask=mask)
                gate_c = gl.minimum(gate.to(gl.float32), limit)
                up_c = gl.minimum(gl.maximum(up.to(gl.float32), -limit), limit)
                h = gate_c / (1.0 + gl.exp(-gate_c)) * up_c
                gl.store(out2_ptr + offsets, h.to(out2_ptr.dtype.element_ty), mask=mask)
        else:
            # dh -> dgate (out0), dup (out1) through the saved gate (in0) / up (in1); h (out2) is
            # recomputed for the down projection's weight gradient. dgate and dup rows are
            # OUT_STRIDE apart, so they can be the two halves of one [rows, 2 * N] tensor.
            base = row_start.to(gl.int64) * N
            out_base = row_start.to(gl.int64) * OUT_STRIDE
            cols = n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout))
            for r in range(BLOCK_M // CHUNK):
                rows = r * CHUNK + gl.arange(0, CHUNK, layout=gl.SliceLayout(1, layout))
                mask = (rows < tile_rows)[:, None] & (cols < N)[None, :]
                offsets = base + rows[:, None] * N + cols[None, :]
                out_offsets = out_base + rows[:, None] * OUT_STRIDE + cols[None, :]
                dh = epi_smem.index(r).load(layout).to(gl.float32)
                gate = gl.load(in0_ptr + offsets, mask=mask, other=0.0).to(gl.float32)
                up = gl.load(in1_ptr + offsets, mask=mask, other=0.0).to(gl.float32)
                gate_c = gl.minimum(gate, limit)
                up_c = gl.minimum(gl.maximum(up, -limit), limit)
                sig = 1.0 / (1.0 + gl.exp(-gate_c))
                silu = gate_c * sig
                dgate = gl.where(gate <= limit, dh * up_c * (sig * (1.0 + gate_c * (1.0 - sig))), 0.0)
                dup = gl.where((up >= -limit) & (up <= limit), dh * silu, 0.0)
                gl.store(out0_ptr + out_offsets, dgate.to(out0_ptr.dtype.element_ty), mask=mask)
                gl.store(out1_ptr + out_offsets, dup.to(out1_ptr.dtype.element_ty), mask=mask)
                gl.store(out2_ptr + offsets, (silu * up_c).to(out2_ptr.dtype.element_ty), mask=mask)
        mbarrier.arrive(epi_free, count=1)
        local += 1


@gluon.jit
def _grouped_gemm_kernel(
    a_desc,
    a2_desc,
    b_desc,
    b2_desc,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    in0_ptr,
    in1_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_row_end_ptr,
    num_tiles_ptr,
    N,
    K,
    W_ROWS,
    B2_ROW_OFFSET,
    OUT_STRIDE,
    limit,
    MODE: gl.constexpr,
    B_N_MAJOR: gl.constexpr,
    DUAL_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    STAGES: gl.constexpr,
    GROUP_M: gl.constexpr,
    EPILOGUE_CHUNK: gl.constexpr,
    EPILOGUE_REGS: gl.constexpr,
):
    a_smem = gl.allocate_shared_memory(a_desc.dtype, [STAGES] + a_desc.block_type.shape, a_desc.layout)
    if B_N_MAJOR:
        b_block: gl.constexpr = [BLOCK_N, BLOCK_K]
    else:
        b_block: gl.constexpr = [BLOCK_K, BLOCK_N]
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype, [STAGES] + b_block, gl.NVMMASharedLayout.get_default_for(b_block, b_desc.dtype)
    )
    # The parked tile is [row chunk, row, column] so the epilogue can walk row chunks with a rolled
    # loop. The 2D swizzle repeats every 8 rows, so for chunks of a multiple of 8 rows this is the
    # same memory layout as the plain [rows, columns] tile the consumers write.
    gl.static_assert(EPILOGUE_CHUNK % 8 == 0)
    epi_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [BLOCK_M // EPILOGUE_CHUNK, EPILOGUE_CHUNK, BLOCK_N],
        gl.SwizzledSharedLayout(8, 1, 8, [1, 0]),
    )
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    epi_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
    mbarrier.init(epi_bars.index(0), count=1)
    mbarrier.init(epi_bars.index(1), count=1)
    fence_async_shared()
    epi_full = epi_bars.index(0)
    epi_free = epi_bars.index(1)
    # Constexpr arguments must be spelled out in each tuple: building tuples by concatenation
    # turns them into tensors.
    gl.warp_specialize(
        [
            (
                _consumer,
                (
                    a_desc,
                    a2_desc,
                    b_desc,
                    b2_desc,
                    a_smem,
                    b_smem,
                    epi_smem,
                    ready,
                    epi_full,
                    epi_free,
                    tile_expert_ptr,
                    tile_row_start_ptr,
                    num_tiles_ptr,
                    N,
                    K,
                    W_ROWS,
                    B2_ROW_OFFSET,
                    MODE,
                    B_N_MAJOR,
                    DUAL_K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    STAGES,
                    GROUP_M,
                ),
            ),
            (
                _epilogue,
                (
                    epi_smem,
                    epi_full,
                    epi_free,
                    out0_ptr,
                    out1_ptr,
                    out2_ptr,
                    in0_ptr,
                    in1_ptr,
                    tile_row_start_ptr,
                    tile_row_end_ptr,
                    num_tiles_ptr,
                    N,
                    OUT_STRIDE,
                    limit,
                    MODE,
                    BLOCK_M,
                    BLOCK_N,
                    GROUP_M,
                ),
            ),
        ],
        [4],
        [EPILOGUE_REGS],
    )


def _desc(tensor: torch.Tensor, block: list[int]) -> TensorDescriptor:
    return TensorDescriptor.from_tensor(tensor, block, gl.NVMMASharedLayout.get_default_for(block, gl.bfloat16))


def weight_view(weight: torch.Tensor) -> tuple[torch.Tensor, int]:
    """A 2D ``[rows, cols]`` view of an ``[E, rows_e, cols]`` weight whose experts are a fixed
    number of rows apart (contiguous, or one half of a packed gate/up weight), and that stride."""
    num_experts, rows, cols = weight.shape
    if weight.stride(2) != 1 or weight.stride(1) != cols or weight.stride(0) % cols:
        raise ValueError(f"expert weights must be row-major with whole rows between experts, got {weight.stride()}")
    expert_rows = weight.stride(0) // cols
    view = weight.as_strided(((num_experts - 1) * expert_rows + rows, cols), (cols, 1))
    return view, expert_rows


DEFAULT_CONFIG = dict(block_n=256, block_k=64, stages=3, group_m=8, epilogue_chunk=8, epilogue_regs=128)


def launch(
    mode: int,
    a: torch.Tensor,
    weight: torch.Tensor,
    table,
    *,
    n_major: bool,
    out_cols: int,
    outputs: tuple[torch.Tensor, ...],
    inputs: tuple[torch.Tensor, ...] = (),
    a2: torch.Tensor | None = None,
    weight2: torch.Tensor | None = None,
    limit: float = 0.0,
    num_sms: int,
    block_n: int,
    block_k: int,
    stages: int,
    group_m: int,
    epilogue_chunk: int,
    epilogue_regs: int,
) -> None:
    """Run one grouped GEMM over ``table``'s row tiles; ``out_cols`` is the N the tiles cover."""
    assert table.block_m == BLOCK_M
    K = a.shape[1]
    w, w_rows = weight_view(weight)
    if mode == MODE_FC1:
        b_block = [block_n // 2, block_k]
    else:
        b_block = [block_n, block_k] if n_major else [block_k, block_n]
    b_desc = _desc(w, b_block)
    b2_desc = b_desc
    b2_row_offset = 0
    if weight2 is not None:
        w2, w2_rows = weight_view(weight2)
        if w2_rows != w_rows:
            raise ValueError("gate_proj and up_proj must have the same expert stride")
        if w2.data_ptr() == w.data_ptr() + weight.shape[1] * w.stride(0) * w.element_size():
            # Packed gate/up halves: address up as rows of one view over the whole packed weight.
            b2_row_offset = weight.shape[1]
            b_desc = b2_desc = _desc(w.as_strided((weight.shape[0] * w_rows, w.shape[1]), w.stride()), b_block)
        else:
            b2_desc = _desc(w2, b_block)
    a_desc = _desc(a, [BLOCK_M, block_k])
    a2_desc = _desc(a2, [BLOCK_M, block_k]) if a2 is not None else a_desc
    outs = list(outputs) + [outputs[0]] * (3 - len(outputs))
    ins = list(inputs) + [outputs[0]] * (2 - len(inputs))
    grid = min(num_sms, table.max_tiles * -(-out_cols // block_n))
    _grouped_gemm_kernel[(grid,)](
        a_desc,
        a2_desc,
        b_desc,
        b2_desc,
        *outs,
        *ins,
        table.expert,
        table.row_start,
        table.row_end,
        table.num_tiles,
        out_cols,
        K,
        w_rows,
        b2_row_offset,
        outs[0].stride(0),
        limit,
        MODE=mode,
        B_N_MAJOR=n_major,
        DUAL_K=a2 is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        STAGES=stages,
        GROUP_M=group_m,
        EPILOGUE_CHUNK=epilogue_chunk,
        EPILOGUE_REGS=epilogue_regs,
        num_warps=8,
    )


def fc1(x, gate_proj, up_proj, table, limit, *, num_sms, config=None):
    """gate, up and h = clamped_swiglu(gate, up), each ``[rows, intermediate]``."""
    num_rows, intermediate = x.shape[0], gate_proj.shape[1]
    gate, up, h = (x.new_empty(num_rows, intermediate) for _ in range(3))
    launch(
        MODE_FC1,
        x,
        gate_proj,
        table,
        n_major=True,
        out_cols=2 * intermediate,
        outputs=(gate, up, h),
        weight2=up_proj,
        limit=limit,
        num_sms=num_sms,
        **(config or DEFAULT_CONFIG),
    )
    return gate, up, h


def down(h, down_proj, table, *, num_sms, config=None):
    """h @ down_proj[e].T per group, ``[rows, H]``."""
    out = h.new_empty(h.shape[0], down_proj.shape[1])
    launch(
        MODE_PLAIN,
        h,
        down_proj,
        table,
        n_major=True,
        out_cols=out.shape[1],
        outputs=(out,),
        num_sms=num_sms,
        **(config or DEFAULT_CONFIG),
    )
    return out


def dswiglu(dout, down_proj, gate, up, table, limit, *, num_sms, config=None):
    """dgate and dup from dh = dout @ down_proj[e], as the two halves of one ``[rows, 2 * I]``
    tensor, and h recomputed from gate / up."""
    num_rows, intermediate = gate.shape
    dgate_dup = gate.new_empty(num_rows, 2 * intermediate)
    dgate, dup = dgate_dup[:, :intermediate], dgate_dup[:, intermediate:]
    h = torch.empty_like(gate)
    launch(
        MODE_DSWIGLU,
        dout,
        down_proj,
        table,
        n_major=False,
        out_cols=intermediate,
        outputs=(dgate, dup, h),
        inputs=(gate, up),
        limit=limit,
        num_sms=num_sms,
        **(config or DEFAULT_CONFIG),
    )
    return dgate_dup, h


def dx(dgate, dup, gate_proj, up_proj, table, *, num_sms, config=None):
    """dgate @ gate_proj[e] + dup @ up_proj[e], ``[rows, H]``."""
    out = dgate.new_empty(dgate.shape[0], gate_proj.shape[2])
    launch(
        MODE_PLAIN,
        dgate,
        gate_proj,
        table,
        n_major=False,
        out_cols=out.shape[1],
        outputs=(out,),
        a2=dup,
        weight2=up_proj,
        num_sms=num_sms,
        **(config or DEFAULT_CONFIG),
    )
    return out
