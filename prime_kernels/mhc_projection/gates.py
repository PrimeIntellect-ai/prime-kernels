"""The mHC gate activations, forward and backward, in one Triton kernel each.

From `mixes` `(T, (2 + hc) * hc)` fp32 (the RMS-normalized projection), `scale` `(3,)` and `base`
`((2 + hc) * hc,)`:

    pre  = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * sigmoid(mixes[:, hc:2hc] * scale[1] + base[hc:2hc])
    comb = sinkhorn(mixes[:, 2hc:].view(hc, hc) * scale[2] + base[2hc:].view(hc, hc))

where `sinkhorn` is a row softmax plus `eps`, a column normalization, then `iters - 1` rounds of
row and column normalization, each dividing by `sum + eps`. The backward recomputes the Sinkhorn
iterates once into a scratch buffer and walks them in reverse.
"""

import torch
import triton
import triton.language as tl

BLOCK_T = 32


@triton.jit
def _comb_logits(mixes_ptr, scale_ptr, base_ptr, rows, row_mask, HC: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, HC)
    offs = 2 * HC + r[:, None] * HC + r[None, :]
    mask = row_mask[:, None, None]
    mixes = tl.load(mixes_ptr + rows[:, None, None] * N + offs[None, :, :], mask=mask, other=0.0)
    base = tl.load(base_ptr + offs)
    return mixes * tl.load(scale_ptr + 2) + base[None, :, :]


@triton.jit
def _softmax_rows(z):
    e = tl.exp(z - tl.max(z, axis=2)[:, :, None])
    return e / tl.sum(e, axis=2)[:, :, None]


@triton.jit
def _sinkhorn_prefix(z, num_ops, eps):
    """The iterate after the first `num_ops` (>= 1) of the Sinkhorn steps.

    Step 0 is the row softmax plus eps; odd steps normalize columns, even steps rows.
    """
    m = _softmax_rows(z) + eps
    for step in range(1, num_ops):
        if step % 2 == 1:
            m = m / (tl.sum(m, axis=1)[:, None, :] + eps)
        else:
            m = m / (tl.sum(m, axis=2)[:, :, None] + eps)
    return m


@triton.jit
def _gates_fwd_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    T,
    eps,
    HC: tl.constexpr,
    N: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < T
    c = tl.arange(0, HC)
    mask2 = row_mask[:, None]

    pre_z = tl.load(mixes_ptr + rows[:, None] * N + c[None, :], mask=mask2, other=0.0)
    pre_z = pre_z * tl.load(scale_ptr) + tl.load(base_ptr + c)[None, :]
    tl.store(pre_ptr + rows[:, None] * HC + c[None, :], tl.sigmoid(pre_z) + eps, mask=mask2)

    post_z = tl.load(mixes_ptr + rows[:, None] * N + HC + c[None, :], mask=mask2, other=0.0)
    post_z = post_z * tl.load(scale_ptr + 1) + tl.load(base_ptr + HC + c)[None, :]
    tl.store(post_ptr + rows[:, None] * HC + c[None, :], 2 * tl.sigmoid(post_z), mask=mask2)

    comb_z = _comb_logits(mixes_ptr, scale_ptr, base_ptr, rows, row_mask, HC, N)
    comb = _sinkhorn_prefix(comb_z, 2 * ITERS, eps)
    tl.store(
        comb_ptr + rows[:, None, None] * (HC * HC) + c[None, :, None] * HC + c[None, None, :],
        comb,
        mask=row_mask[:, None, None],
    )


@triton.jit
def _gates_bwd_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    grad_pre_ptr,
    grad_post_ptr,
    grad_comb_ptr,
    grad_mixes_ptr,
    grad_z_ptr,
    scratch_ptr,
    T,
    eps,
    HC: tl.constexpr,
    N: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Writes the gradients w.r.t. `mixes` and w.r.t. the pre-activation logits `z`.

    `scratch` holds the input of every Sinkhorn step after the first, `(2 * ITERS, T, HC * HC)`.
    """
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < T
    c = tl.arange(0, HC)
    mask2 = row_mask[:, None]
    offs2 = rows[:, None] * N + c[None, :]

    scale = tl.load(scale_ptr)
    pre_z = tl.load(mixes_ptr + offs2, mask=mask2, other=0.0) * scale + tl.load(base_ptr + c)[None, :]
    sig = tl.sigmoid(pre_z)
    grad = tl.load(grad_pre_ptr + rows[:, None] * HC + c[None, :], mask=mask2, other=0.0) * sig * (1 - sig)
    tl.store(grad_z_ptr + offs2, grad, mask=mask2)
    tl.store(grad_mixes_ptr + offs2, grad * scale, mask=mask2)

    scale = tl.load(scale_ptr + 1)
    post_z = tl.load(mixes_ptr + HC + offs2, mask=mask2, other=0.0) * scale + tl.load(base_ptr + HC + c)[None, :]
    sig = tl.sigmoid(post_z)
    grad = 2 * tl.load(grad_post_ptr + rows[:, None] * HC + c[None, :], mask=mask2, other=0.0) * sig * (1 - sig)
    tl.store(grad_z_ptr + HC + offs2, grad, mask=mask2)
    tl.store(grad_mixes_ptr + HC + offs2, grad * scale, mask=mask2)

    comb_z = _comb_logits(mixes_ptr, scale_ptr, base_ptr, rows, row_mask, HC, N)
    mask3 = row_mask[:, None, None]
    grad = tl.load(
        grad_comb_ptr + rows[:, None, None] * (HC * HC) + c[None, :, None] * HC + c[None, None, :],
        mask=mask3,
        other=0.0,
    )
    scratch = scratch_ptr + rows[:, None, None] * (HC * HC) + c[None, :, None] * HC + c[None, None, :]
    m = _softmax_rows(comb_z) + eps
    for step in range(1, 2 * ITERS):
        tl.store(scratch + step * T * HC * HC, m, mask=mask3)
        if step % 2 == 1:
            m = m / (tl.sum(m, axis=1)[:, None, :] + eps)
        else:
            m = m / (tl.sum(m, axis=2)[:, :, None] + eps)
    # Back through steps 2 * ITERS - 1 .. 1. For out = m / s with s = sum(m) + eps along an axis,
    # d m = (d out - sum(d out * out)) / s along that axis.
    for back in range(0, 2 * ITERS - 1):
        step = 2 * ITERS - 1 - back
        m = tl.load(scratch + step * T * HC * HC, mask=mask3, other=1.0)
        if step % 2 == 1:
            col_sum = tl.sum(m, axis=1)[:, None, :] + eps
            grad = (grad - tl.sum(grad * (m / col_sum), axis=1)[:, None, :]) / col_sum
        else:
            row_sum = tl.sum(m, axis=2)[:, :, None] + eps
            grad = (grad - tl.sum(grad * (m / row_sum), axis=2)[:, :, None]) / row_sum
    # Step 0: softmax (+ eps, which has no gradient).
    p = _softmax_rows(comb_z)
    grad = p * (grad - tl.sum(grad * p, axis=2)[:, :, None])
    offs3 = rows[:, None, None] * N + 2 * HC + c[None, :, None] * HC + c[None, None, :]
    tl.store(grad_z_ptr + offs3, grad, mask=mask3)
    tl.store(grad_mixes_ptr + offs3, grad * tl.load(scale_ptr + 2), mask=mask3)


def _check(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int) -> None:
    n = (2 + hc) * hc
    if mixes.dim() != 2 or mixes.shape[1] != n:
        raise ValueError(f"expected mixes of shape (tokens, {n}), got {tuple(mixes.shape)}")
    if hc & (hc - 1):
        raise ValueError(f"hc must be a power of two, got {hc}")
    if tuple(scale.shape) != (3,) or tuple(base.shape) != (n,):
        raise ValueError(f"expected scale (3,) and base ({n},), got {tuple(scale.shape)}, {tuple(base.shape)}")


@torch.library.custom_op("prime_kernels::mhc_gates_forward", mutates_args=())
def mhc_gates_forward(
    mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(mixes, scale, base, hc)
    tokens = mixes.shape[0]
    pre = mixes.new_empty(tokens, hc, dtype=torch.float32)
    post = mixes.new_empty(tokens, hc, dtype=torch.float32)
    comb = mixes.new_empty(tokens, hc, hc, dtype=torch.float32)
    if tokens > 0:
        _gates_fwd_kernel[(triton.cdiv(tokens, BLOCK_T),)](
            mixes.float().contiguous(),
            scale.float().contiguous(),
            base.float().contiguous(),
            pre,
            post,
            comb,
            tokens,
            eps,
            HC=hc,
            N=mixes.shape[1],
            ITERS=iters,
            BLOCK_T=BLOCK_T,
        )
    return pre, post, comb


@mhc_gates_forward.register_fake
def _mhc_gates_forward_fake(mixes, scale, base, hc, iters, eps):
    _check(mixes, scale, base, hc)
    tokens = mixes.shape[0]
    pre = mixes.new_empty(tokens, hc, dtype=torch.float32)
    return pre, torch.empty_like(pre), mixes.new_empty(tokens, hc, hc, dtype=torch.float32)


@torch.library.custom_op("prime_kernels::mhc_gates_backward", mutates_args=())
def mhc_gates_backward(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    grad_pre: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,
    hc: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(grad_mixes, grad_z)`, the latter w.r.t. `mixes * scale + base`."""
    tokens = mixes.shape[0]
    grad_mixes = torch.empty_like(mixes, dtype=torch.float32)
    grad_z = torch.empty_like(mixes, dtype=torch.float32)
    scratch = mixes.new_empty(2 * iters, tokens, hc * hc, dtype=torch.float32)
    if tokens > 0:
        _gates_bwd_kernel[(triton.cdiv(tokens, BLOCK_T),)](
            mixes.float().contiguous(),
            scale.float().contiguous(),
            base.float().contiguous(),
            grad_pre.float().contiguous(),
            grad_post.float().contiguous(),
            grad_comb.float().contiguous(),
            grad_mixes,
            grad_z,
            scratch,
            tokens,
            eps,
            HC=hc,
            N=mixes.shape[1],
            ITERS=iters,
            BLOCK_T=BLOCK_T,
        )
    return grad_mixes, grad_z


@mhc_gates_backward.register_fake
def _mhc_gates_backward_fake(mixes, scale, base, grad_pre, grad_post, grad_comb, hc, iters, eps):
    return torch.empty_like(mixes, dtype=torch.float32), torch.empty_like(mixes, dtype=torch.float32)


def _setup_context(ctx, inputs, output) -> None:
    mixes, scale, base, hc, iters, eps = inputs
    ctx.save_for_backward(mixes, scale, base)
    ctx.hc, ctx.iters, ctx.eps = hc, iters, eps


def _backward(ctx, grad_pre, grad_post, grad_comb):
    mixes, scale, base = ctx.saved_tensors
    hc = ctx.hc
    grad_mixes, grad_z = mhc_gates_backward(
        mixes, scale, base, grad_pre, grad_post, grad_comb, hc, ctx.iters, ctx.eps
    )
    grad_base = grad_z.sum(0)
    grad_scale = torch.stack(
        [part.sum() for part in (grad_z * mixes.float()).split([hc, hc, hc * hc], dim=1)]
    )
    return grad_mixes.to(mixes.dtype), grad_scale.to(scale.dtype), grad_base.to(base.dtype), None, None, None


mhc_gates_forward.register_autograd(_backward, setup_context=_setup_context)


def mhc_gates(
    mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(pre, post, comb)` gates from `(..., (2 + hc) * hc)` mixes, all fp32."""
    lead = mixes.shape[:-1]
    pre, post, comb = mhc_gates_forward(mixes.reshape(-1, mixes.shape[-1]), scale, base, hc, iters, eps)
    return pre.view(*lead, hc), post.view(*lead, hc), comb.view(*lead, hc, hc)
