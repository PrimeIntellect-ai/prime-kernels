"""Host side of the SM90 sparse attention backward: Triton pre/post-processing around the CuTe DSL
main kernel, exposed as a `torch.library` custom op with the contract of prime-rl's
`dsv41_sparse_attn_backward` (and cuDNN's `flash_attn_bwd_sm90`)."""

import torch
import triton
import triton.language as tl

HEADS = 64
DIM = 512
TILE_N = 64

_compiled: dict = {}


@triton.jit
def _preprocess_kernel(
    out_ptr,
    dout_ptr,
    lse_ptr,
    sink_ptr,
    idx_ptr,
    delta_ptr,
    l2_ptr,
    dsink_ptr,
    idx_out_ptr,
    count_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per query: delta = rowsum(dO * O), the sink-aware LSE in log2 units, each head's sink gradient
    -p_sink * delta, and the slot list with its non-empty slots moved to the front (then -1s) plus
    their count."""
    t = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, H)
    acc = tl.zeros([H], dtype=tl.float32)
    for d0 in tl.static_range(0, D, BLOCK_D):
        offs = t * H * D + heads[:, None] * D + d0 + tl.arange(0, BLOCK_D)[None, :]
        o = tl.load(out_ptr + offs).to(tl.float32)
        g = tl.load(dout_ptr + offs).to(tl.float32)
        acc += tl.sum(o * g, axis=1)
    lse = tl.load(lse_ptr + t * H + heads)
    sink = tl.load(sink_ptr + heads)
    hi = tl.maximum(lse, sink)
    lse_sink = hi + tl.log(tl.exp(lse - hi) + tl.exp(sink - hi))
    tl.store(delta_ptr + t * H + heads, acc)
    tl.store(l2_ptr + t * H + heads, lse_sink * 1.4426950408889634)
    tl.store(dsink_ptr + t * H + heads, -tl.exp(sink - lse_sink) * acc)

    slots = tl.arange(0, BLOCK_K)
    in_row = slots < K
    idx = tl.load(idx_ptr + t * K + slots, mask=in_row, other=-1)
    valid = (idx >= 0).to(tl.int32)
    empty = (in_row & (idx < 0)).to(tl.int32)
    count = tl.sum(valid, axis=0)
    pos = tl.where(valid == 1, tl.cumsum(valid, axis=0) - valid, count + tl.cumsum(empty, axis=0) - empty)
    tl.store(idx_out_ptr + t * K + pos, tl.where(valid == 1, idx, -1), mask=in_row)
    tl.store(count_ptr + t, count)


@triton.jit
def _postprocess_kernel(acc_ptr, dkv_ptr, n_rows, D: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    """Undo the scatter's column permutation (see `SparseAttnBwdSm90.scatter`) and cast to bf16."""
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, D)
    w = cols % 16
    fake = (cols - w) + 4 * ((w % 8) // 2) + (w % 2) + 2 * (w // 8)
    mask = rows[:, None] < n_rows
    rows64 = rows.to(tl.int64)[:, None]
    vals = tl.load(acc_ptr + rows64 * D + fake[None, :], mask=mask)
    tl.store(dkv_ptr + rows64 * D + cols[None, :], vals.to(tl.bfloat16), mask=mask)


def _main_kernel(num_slots: int, device: torch.device):
    key = (num_slots, device.index)
    if key not in _compiled:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        from prime_kernels.dsa_sparse_attn_bwd.kernel import SparseAttnBwdSm90

        def fake(shape, dtype):
            t = torch.empty(shape, dtype=dtype, device=device)
            return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=t.ndim - 1)

        bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32
        args = (
            fake((2, HEADS, DIM), bf16),  # q
            fake((3, DIM), bf16),  # kv
            fake((2, HEADS, DIM), bf16),  # grad_out
            fake((2, HEADS), f32),  # log2 LSE with sink
            fake((2, HEADS), f32),  # delta
            fake((2, num_slots), i32),  # compacted slots
            fake((2,), i32),  # slot counts
            fake((2, HEADS, DIM), bf16),  # dq
            fake((3, DIM), f32),  # dkv accumulator
        )
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        _compiled[key] = cute.compile(
            SparseAttnBwdSm90(num_slots),
            *args,
            cutlass.Float32(1.0),
            cutlass.Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
    return _compiled[key]


def unsupported_shape_reason(num_heads: int, head_dim: int) -> str | None:
    if num_heads != HEADS or head_dim != DIM:
        return f"dsa_sparse_attn_bwd supports {HEADS} heads of a {DIM}-wide latent, got {num_heads} x {head_dim}"
    return None


def sparse_attn_backward_flat(
    q: torch.Tensor,  # (t, h, d) bf16
    kv: torch.Tensor,  # (n, d) bf16, shared K = V latent rows
    out: torch.Tensor,  # (t, h, d) bf16, forward output (sink applied)
    grad_out: torch.Tensor,  # (t, h, d) bf16
    lse: torch.Tensor,  # (t, h) f32, sink-free natural-log LSE (as FlashMLA returns it)
    indices: torch.Tensor,  # (t, k) int32 rows of kv, -1 = empty slot
    sinks: torch.Tensor,  # (h,) f32
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dq (t, h, d) bf16, dkv (n, d) bf16, dsinks (h,) f32 of sink-aware sparse attention."""
    import cuda.bindings.driver as cuda

    t, h, d = q.shape
    n = kv.shape[0]
    reason = unsupported_shape_reason(h, d)
    if reason is not None:
        raise ValueError(reason)
    if indices.shape[1] % TILE_N:
        indices = torch.nn.functional.pad(indices, (0, TILE_N - indices.shape[1] % TILE_N), value=-1)
    q, kv, out, grad_out, indices = (x.contiguous() for x in (q, kv, out, grad_out, indices))
    lse = lse.float().contiguous()
    sinks = sinks.float().contiguous()
    k = indices.shape[1]

    delta = torch.empty(t, h, dtype=torch.float32, device=q.device)
    lse2 = torch.empty_like(delta)
    dsink_part = torch.empty_like(delta)
    slots = torch.empty_like(indices)
    count = torch.empty(t, dtype=torch.int32, device=q.device)
    dq = torch.empty_like(q)
    dkv_acc = torch.zeros(n, d, dtype=torch.float32, device=q.device)
    dkv = torch.empty(n, d, dtype=kv.dtype, device=q.device)
    if t == 0:
        return dq, dkv.zero_(), torch.zeros_like(sinks)

    _preprocess_kernel[(t,)](
        out,
        grad_out,
        lse,
        sinks,
        indices,
        delta,
        lse2,
        dsink_part,
        slots,
        count,
        H=h,
        D=d,
        BLOCK_D=128,
        K=k,
        BLOCK_K=triton.next_power_of_2(k),
    )
    num_ctas = min(t, torch.cuda.get_device_properties(q.device).multi_processor_count)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    _main_kernel(k, q.device)(q, kv, grad_out, lse2, delta, slots, count, dq, dkv_acc, sm_scale, num_ctas, stream)
    _postprocess_kernel[(triton.cdiv(n, 16),)](dkv_acc, dkv, n, D=d, BLOCK_ROWS=16)
    return dq, dkv, dsink_part.sum(0)


@torch.library.custom_op("prime_kernels::dsa_sparse_attn_bwd", mutates_args=())
def dsa_sparse_attn_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in for prime-rl's `dsv41_sparse_attn_backward`: `q`, `out`, `grad_out` (1, t, h, d), `kv`
    (1, n, 1, d), `lse` (1, t, h), `indices` (1, t, 1, k) int32 (-1 = empty), `sinks` (h,) ->
    dq like q, dkv like kv, dsinks like sinks."""
    _, t, h, d = q.shape
    dq, dkv, dsinks = sparse_attn_backward_flat(
        q.view(t, h, d),
        kv.view(-1, d),
        out.view(t, h, d),
        grad_out.contiguous().view(t, h, d),
        lse.view(t, h),
        indices.reshape(t, -1),
        sinks,
        sm_scale,
    )
    return dq.view_as(q), dkv.view_as(kv).to(kv.dtype), dsinks.to(sinks.dtype)


@dsa_sparse_attn_backward.register_fake
def _dsa_sparse_attn_backward_fake(grad_out, q, kv, out, lse, indices, sinks, sm_scale):
    return torch.empty_like(q), torch.empty_like(kv), torch.empty_like(sinks)
