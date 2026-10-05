"""Fused mHC projection: RMS over the flattened streams, projection to the gate logits, and the
stream collapse with the previous sublayer's `pre` gate, in one pass over the streams.

For streams `x` of shape `(T, hc, d)` (flattened to `K = hc * d` per token), weight `w` of shape
`(n, K)` and an optional `pre_mix` of shape `(T, hc)`:

    rstd      = rsqrt(mean(x ** 2) + eps)                  per token, fp32
    mixes     = (x @ w.T) * rstd                           (T, n), fp32
    collapsed = sum_i pre_mix[:, i] * x[:, i, :]           (T, d), x.dtype

The projection runs on bf16 operands with fp32 accumulation (the weight is rounded to bf16 as
prime-rl's eager path does); everything else is fp32. The backward reads the streams once more
and writes `dx = (dmixes * rstd) @ w + coef * x + pre_mix * dcollapsed`, the weight gradient as
fixed-order split-token partials (deterministic), and the `pre_mix` gradient. Its GEMM operand
`dmixes * rstd` is rounded to bf16, again as prime-rl's eager path does.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Gate logits are padded to this width for the tensor-core tiles.
BLOCK_N = 32


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_D": bd}, num_warps=nw, num_stages=ns)
        for bm in (32, 64)
        for bd in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 3, 4)
        if bm * bd <= 64 * 128
    ],
    key=["HC", "D", "HAS_PRE"],
)
@triton.jit
def _projection_fwd_kernel(
    x_ptr,
    w_ptr,
    pre_ptr,
    mixes_ptr,
    rstd_ptr,
    collapsed_ptr,
    T,
    eps,
    HC: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    HAS_PRE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    K: tl.constexpr = HC * D
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < T
    rows64 = rows.to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + offs_d
        d_mask = d < D
        tile_mask = row_mask[:, None] & d_mask[None, :]
        collapsed = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        # Tensor-core accumulation loses precision over long K chains, so each d-block's partial
        # product is promoted into `acc` with ordinary fp32 adds.
        part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in tl.static_range(HC):
            k = i * D + d
            x = tl.load(x_ptr + rows64[:, None] * K + k[None, :], mask=tile_mask, other=0.0)
            # The weight is padded to BLOCK_N rows by the caller.
            w = tl.load(w_ptr + offs_n[:, None] * K + k[None, :], mask=d_mask[None, :], other=0.0)
            part = tl.dot(x, tl.trans(w), part)
            xf = x.to(tl.float32)
            sumsq += tl.sum(xf * xf, axis=1)
            if HAS_PRE:
                pre = tl.load(pre_ptr + rows * HC + i, mask=row_mask, other=0.0)
                collapsed += pre[:, None] * xf
        acc += part
        if HAS_PRE:
            tl.store(
                collapsed_ptr + rows64[:, None] * D + d[None, :],
                collapsed.to(collapsed_ptr.dtype.element_ty),
                mask=tile_mask,
            )

    rstd = tl.rsqrt(sumsq / K + eps)
    tl.store(rstd_ptr + rows, rstd, mask=row_mask)
    tl.store(
        mixes_ptr + rows[:, None] * N + offs_n[None, :],
        acc * rstd[:, None],
        mask=row_mask[:, None] & (offs_n[None, :] < N),
    )


@triton.jit
def _projection_bwd_prep_kernel(
    grad_mixes_ptr,
    mixes_ptr,
    rstd_ptr,
    grad_proj_ptr,
    coef_ptr,
    T,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Per-token backward factors: `grad_proj = dmixes * rstd` (bf16, padded to BLOCK_N) and
    `coef = -rstd^2 * sum(dmixes * mixes) / K`, the factor rstd's dependence on x contributes."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < T
    offs_n = tl.arange(0, BLOCK_N)
    mask = row_mask[:, None] & (offs_n[None, :] < N)
    grad_mixes = tl.load(grad_mixes_ptr + rows[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
    mixes = tl.load(mixes_ptr + rows[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    tl.store(
        grad_proj_ptr + rows[:, None] * BLOCK_N + offs_n[None, :],
        (grad_mixes * rstd[:, None]).to(tl.bfloat16),
        mask=row_mask[:, None],
    )
    tl.store(coef_ptr + rows, -rstd * rstd * tl.sum(grad_mixes * mixes, axis=1) / K, mask=row_mask)


def _set_bwd_block_shapes(nargs):
    block_m, block_d, block_n = nargs["BLOCK_M"], nargs["BLOCK_D"], nargs["BLOCK_N"]
    nargs["x_desc"].block_shape = [block_m, 1, block_d]
    nargs["grad_x_desc"].block_shape = [block_m, 1, block_d]
    nargs["w_desc"].block_shape = [block_n, 1, block_d]
    nargs["grad_proj_desc"].block_shape = [block_m, block_n]
    nargs["grad_collapsed_desc"].block_shape = [block_m, block_d]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_D": bd, "SPLITS": s},
            num_warps=nw,
            num_stages=ns,
            pre_hook=_set_bwd_block_shapes,
        )
        for bd in (64, 128)
        for s in (8, 16)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["HC", "D", "HAS_PRE"],
)
@triton.jit
def _projection_bwd_kernel(
    x_desc,
    w_desc,
    pre_ptr,
    grad_proj_desc,
    coef_ptr,
    grad_collapsed_desc,
    grad_x_desc,
    grad_w_partial_ptr,
    grad_pre_partial_ptr,
    T,
    HC: tl.constexpr,
    D: tl.constexpr,
    HAS_PRE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Grid: (cdiv(D, BLOCK_D) * HC, SPLITS). One stream's d-block over a contiguous token range."""
    K: tl.constexpr = HC * D
    pid = tl.program_id(0)
    split = tl.program_id(1)
    # The stream varies fastest so the HC programs sharing a `grad_collapsed` tile run together.
    i = pid % HC
    d_block = pid // HC
    d0 = d_block * BLOCK_D

    w = w_desc.load([0, i, d0]).reshape(BLOCK_N, BLOCK_D)
    grad_w = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    diagonal = tl.arange(0, BLOCK_M)[:, None] == tl.arange(0, BLOCK_M)[None, :]

    m_tiles = tl.cdiv(T, BLOCK_M)
    per_split = tl.cdiv(m_tiles, SPLITS)
    begin = split * per_split
    end = tl.minimum(begin + per_split, m_tiles)
    for m_tile in range(begin, end):
        m0 = m_tile * BLOCK_M
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < T
        grad_proj = grad_proj_desc.load([m0, 0])
        coef = tl.load(coef_ptr + rows, mask=row_mask, other=0.0)
        x = x_desc.load([m0, i, d0]).reshape(BLOCK_M, BLOCK_D)
        if HAS_PRE:
            grad_collapsed = grad_collapsed_desc.load([m0, d0])
            pre = tl.load(pre_ptr + rows * HC + i, mask=row_mask, other=0.0)
        xf = x.to(tl.float32)
        grad_x = tl.dot(grad_proj, w) + coef[:, None] * xf
        if HAS_PRE:
            grad_x += pre[:, None] * grad_collapsed.to(tl.float32)
            # The row-wise dot products of grad_collapsed and x are the diagonal of
            # grad_collapsed @ x.T. The tensor cores compute it exactly (bf16 products, fp32
            # sums) and much faster than an elementwise product and cross-thread reduction.
            gram = tl.dot(grad_collapsed, tl.trans(x))
            grad_pre = tl.sum(tl.where(diagonal, gram, 0.0), axis=1)
            tl.store(grad_pre_partial_ptr + (d_block * HC + i) * T + rows, grad_pre, mask=row_mask)
        grad_x_desc.store([m0, i, d0], grad_x.to(grad_x_desc.dtype).reshape(BLOCK_M, 1, BLOCK_D))
        grad_w += tl.dot(tl.trans(x), grad_proj)

    offs_d = d0 + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    tl.store(
        grad_w_partial_ptr + (split * BLOCK_N + offs_n[None, :]) * K + i * D + offs_d[:, None],
        grad_w,
        mask=(offs_d < D)[:, None],
    )


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """Contiguous with a 16-byte aligned base, as the TMA descriptors require."""
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _padded_weight(weight: torch.Tensor) -> torch.Tensor:
    padded = weight.new_zeros(BLOCK_N, weight.shape[1], dtype=torch.bfloat16)
    padded[: weight.shape[0]] = weight
    return padded


def _check(x: torch.Tensor, weight: torch.Tensor, pre_mix: torch.Tensor | None) -> None:
    if x.dim() != 3:
        raise ValueError(f"expected streams of shape (tokens, hc, d), got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"expected bf16 streams, got {x.dtype}")
    tokens, hc, d = x.shape
    if weight.dim() != 2 or weight.shape[1] != hc * d:
        raise ValueError(f"expected a weight of shape (n, {hc * d}), got {tuple(weight.shape)}")
    if weight.shape[0] > BLOCK_N:
        raise ValueError(f"at most {BLOCK_N} gate logits are supported, got {weight.shape[0]}")
    if d % 8:
        raise ValueError(f"d must be a multiple of 8 (16-byte rows for TMA), got {d}")
    if pre_mix is not None and tuple(pre_mix.shape) != (tokens, hc):
        raise ValueError(f"expected pre_mix of shape ({tokens}, {hc}), got {tuple(pre_mix.shape)}")


@torch.library.custom_op("prime_kernels::mhc_projection_forward", mutates_args=())
def mhc_projection_forward(
    x: torch.Tensor, weight: torch.Tensor, pre_mix: torch.Tensor | None, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(mixes, rstd, collapsed)`; `collapsed` is empty when `pre_mix` is None."""
    _check(x, weight, pre_mix)
    tokens, hc, d = x.shape
    n = weight.shape[0]
    x = x.contiguous()
    w = _padded_weight(weight)
    mixes = x.new_empty(tokens, n, dtype=torch.float32)
    rstd = x.new_empty(tokens, dtype=torch.float32)
    has_pre = pre_mix is not None
    collapsed = x.new_empty(tokens, d) if has_pre else x.new_empty(0)
    pre = pre_mix.float().contiguous() if has_pre else mixes
    if tokens > 0:
        grid = lambda meta: (triton.cdiv(tokens, meta["BLOCK_M"]),)
        _projection_fwd_kernel[grid](
            x, w, pre, mixes, rstd, collapsed, tokens, eps, HC=hc, D=d, N=n, HAS_PRE=has_pre, BLOCK_N=BLOCK_N
        )
    return mixes, rstd, collapsed


@mhc_projection_forward.register_fake
def _mhc_projection_forward_fake(x, weight, pre_mix, eps):
    _check(x, weight, pre_mix)
    tokens, hc, d = x.shape
    mixes = x.new_empty(tokens, weight.shape[0], dtype=torch.float32)
    rstd = x.new_empty(tokens, dtype=torch.float32)
    collapsed = x.new_empty(tokens, d) if pre_mix is not None else x.new_empty(0)
    return mixes, rstd, collapsed


@torch.library.custom_op("prime_kernels::mhc_projection_backward", mutates_args=())
def mhc_projection_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    pre_mix: torch.Tensor | None,
    mixes: torch.Tensor,
    rstd: torch.Tensor,
    grad_mixes: torch.Tensor,
    grad_collapsed: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(grad_x, grad_weight, grad_pre_mix)`; `grad_pre_mix` is empty without `pre_mix`.

    `grad_weight` is fp32 regardless of the weight dtype.
    """
    tokens, hc, d = x.shape
    n = weight.shape[0]
    x = _aligned(x)
    has_pre = pre_mix is not None and grad_collapsed is not None
    grad_x = torch.empty_like(x)
    if tokens == 0:
        grad_weight = x.new_zeros(n, hc * d, dtype=torch.float32)
        grad_pre_mix = x.new_zeros(0 if pre_mix is None else (0, hc), dtype=torch.float32)
        return grad_x, grad_weight, grad_pre_mix

    grad_proj = x.new_empty(tokens, BLOCK_N, dtype=torch.bfloat16)
    coef = x.new_empty(tokens, dtype=torch.float32)
    _projection_bwd_prep_kernel[(triton.cdiv(tokens, 64),)](
        grad_mixes.float().contiguous(), mixes, rstd, grad_proj, coef, tokens, K=hc * d, N=n, BLOCK_N=BLOCK_N, BLOCK_M=64
    )
    pre = pre_mix.float().contiguous() if has_pre else coef
    grad_collapsed = _aligned(grad_collapsed) if has_pre else x.view(-1, d)
    # Sized for the smallest BLOCK_D and the largest SPLITS the autotuner may pick; only the
    # slots the chosen config writes are summed below.
    max_d_blocks, max_splits = triton.cdiv(d, 64), 16
    grad_w_partial = x.new_empty(max_splits, BLOCK_N, hc * d, dtype=torch.float32)
    grad_pre_partial = x.new_empty(max_d_blocks, hc, tokens, dtype=torch.float32) if has_pre else coef
    # Block shapes are placeholders: the autotuner's pre-hook sets them per config.
    _projection_bwd_kernel[lambda meta: (triton.cdiv(d, meta["BLOCK_D"]) * hc, meta["SPLITS"])](
        TensorDescriptor.from_tensor(x, [1, 1, 8]),
        TensorDescriptor.from_tensor(_padded_weight(weight).view(BLOCK_N, hc, d), [1, 1, 8]),
        pre,
        TensorDescriptor.from_tensor(grad_proj, [1, 8]),
        coef,
        TensorDescriptor.from_tensor(grad_collapsed, [1, 8]),
        TensorDescriptor.from_tensor(grad_x, [1, 1, 8]),
        grad_w_partial,
        grad_pre_partial,
        tokens,
        HC=hc,
        D=d,
        HAS_PRE=has_pre,
        BLOCK_N=BLOCK_N,
    )
    config = _projection_bwd_kernel.best_config.kwargs
    # Fixed-order sums over the partials: deterministic.
    grad_weight = grad_w_partial[: config["SPLITS"], :n].sum(0)
    if has_pre:
        grad_pre_mix = grad_pre_partial[: triton.cdiv(d, config["BLOCK_D"])].sum(0).t().contiguous()
    elif pre_mix is not None:
        grad_pre_mix = torch.zeros_like(pre_mix, dtype=torch.float32)
    else:
        grad_pre_mix = x.new_empty(0, dtype=torch.float32)
    return grad_x, grad_weight, grad_pre_mix


@mhc_projection_backward.register_fake
def _mhc_projection_backward_fake(x, weight, pre_mix, mixes, rstd, grad_mixes, grad_collapsed):
    grad_pre_mix = (
        x.new_empty(pre_mix.shape, dtype=torch.float32) if pre_mix is not None else x.new_empty(0, dtype=torch.float32)
    )
    return torch.empty_like(x), x.new_empty(weight.shape, dtype=torch.float32), grad_pre_mix


def _setup_context(ctx, inputs, output) -> None:
    x, weight, pre_mix, _ = inputs
    mixes, rstd, _ = output
    ctx.save_for_backward(x, weight, pre_mix, mixes, rstd)
    ctx.mark_non_differentiable(rstd)
    ctx.has_pre = pre_mix is not None


def _backward(ctx, grad_mixes, grad_rstd, grad_collapsed):
    x, weight, pre_mix, mixes, rstd = ctx.saved_tensors
    if grad_mixes is None:
        grad_mixes = torch.zeros_like(mixes)
    grad_x, grad_weight, grad_pre_mix = mhc_projection_backward(
        x, weight, pre_mix, mixes, rstd, grad_mixes, grad_collapsed if ctx.has_pre else None
    )
    grad_pre_mix = grad_pre_mix.to(pre_mix.dtype) if ctx.has_pre else None
    return grad_x, grad_weight.to(weight.dtype), grad_pre_mix, None


mhc_projection_forward.register_autograd(_backward, setup_context=_setup_context)


def mhc_projection(
    x: torch.Tensor, weight: torch.Tensor, pre_mix: torch.Tensor | None = None, eps: float = 1e-20
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """RMS-normalized projection of the flattened streams, plus the optional `pre_mix` collapse.

    `x`: `(..., hc, d)` bf16 streams. `weight`: `(n, hc * d)`, `n <= 32`. `pre_mix`: `(..., hc)`.
    Returns `mixes` `(..., n)` fp32 and `collapsed` `(..., d)` in `x.dtype` (None without `pre_mix`).
    """
    lead = x.shape[:-2]
    hc, d = x.shape[-2:]
    flat_pre = pre_mix.reshape(-1, hc) if pre_mix is not None else None
    mixes, _, collapsed = mhc_projection_forward(x.reshape(-1, hc, d), weight, flat_pre, eps)
    mixes = mixes.view(*lead, weight.shape[0])
    return mixes, collapsed.view(*lead, d) if pre_mix is not None else None
