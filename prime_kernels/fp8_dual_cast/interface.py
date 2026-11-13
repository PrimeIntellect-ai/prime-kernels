"""Per-token and transposed per-token e4m3 casts of a bf16 matrix in one pass (optionally after DeepSeek-V4's
inverse RoPE), with the outputs of prime-rl's `per_token_cast_to_fp8_triton` and
`per_token_cast_to_fp8_tp_triton` (on the rotated matrix) bit for bit."""

import torch

BLOCK = 128
_compiled: dict = {}


def _kernel(rope: bool, head_dim: int, device: torch.device):
    key = (rope, head_dim, device.index)
    if key not in _compiled:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        from prime_kernels.fp8_dual_cast.kernel import DualCastSm90

        def fake(shape, dtype):
            t = torch.empty(shape, dtype=dtype, device=device)
            return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=t.ndim - 1)

        args = (
            fake((BLOCK, 2 * BLOCK), torch.int16),
            fake((2, 64), torch.float32),
            fake((BLOCK,), torch.int32),
            fake((BLOCK, 2 * BLOCK), torch.uint8),
            fake((2, BLOCK), torch.float32),
            fake((2 * BLOCK, BLOCK), torch.uint8),
            fake((1, 2 * BLOCK), torch.float32),
        )
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        _compiled[key] = cute.compile(
            DualCastSm90(rope, head_dim), *args, cutlass.Int32(1), stream, options="--enable-tvm-ffi"
        )
    return _compiled[key]


def _casts(x, cos_sin, positions, head_dim):
    import cuda.bindings.driver as cuda

    rows, cols = x.shape
    prows = (rows + BLOCK - 1) // BLOCK * BLOCK
    out = torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty(cols // BLOCK, (rows + 3) // 4 * 4, dtype=torch.float32, device=x.device)
    out_t = torch.empty(cols, prows, dtype=torch.float8_e4m3fn, device=x.device)
    sf_t = torch.empty(prows // BLOCK, cols, dtype=torch.float32, device=x.device)
    rope = cos_sin is not None
    if rope:
        assert cos_sin.shape[-1] == 64 and head_dim % BLOCK == 0 and cols % head_dim == 0
        cos_sin = cos_sin.float().contiguous()
        positions = positions.reshape(rows).to(torch.int32)
    else:
        cos_sin = torch.zeros(1, 64, dtype=torch.float32, device=x.device)
        positions = torch.zeros(max(rows, 1), dtype=torch.int32, device=x.device)
        head_dim = BLOCK
    if rows:
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        _kernel(rope, head_dim, x.device)(
            x.view(torch.int16), cos_sin, positions, out.view(torch.uint8), sf, out_t.view(torch.uint8), sf_t, rows,
            stream,
        )
    return out, sf, out_t, sf_t


@torch.library.custom_op("prime_kernels::fp8_dual_cast", mutates_args=())
def fp8_dual_cast(
    x: torch.Tensor, cos_sin: torch.Tensor | None, positions: torch.Tensor | None, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`dual_cast_to_fp8` as an op, returning the scale tensors whole: (x_fp8, sf (cols / 128, rows rounded up
    to 4)), (x_t_fp8 (cols, padded rows), sf_t (padded rows / 128, cols)); the casts' scales are
    `sf[:, :rows].T` and `sf_t.T`."""
    assert x.dim() == 2 and x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape[1] % BLOCK == 0
    return _casts(x, cos_sin, positions, head_dim)


@fp8_dual_cast.register_fake
def _fp8_dual_cast_fake(x, cos_sin, positions, head_dim):
    rows, cols = x.shape
    prows = (rows + BLOCK - 1) // BLOCK * BLOCK
    return (
        x.new_empty(rows, cols, dtype=torch.float8_e4m3fn),
        x.new_empty(cols // BLOCK, (rows + 3) // 4 * 4, dtype=torch.float32),
        x.new_empty(cols, prows, dtype=torch.float8_e4m3fn),
        x.new_empty(prows // BLOCK, cols, dtype=torch.float32),
    )


def dual_cast_to_fp8(
    x: torch.Tensor,  # (rows, cols) bf16, contiguous
    rope: tuple[torch.Tensor, torch.Tensor, int] | None = None,  # ([cos | sin] f32 (positions, 64), positions, head_dim)
):
    """((x_fp8, scales (rows, cols / 128) MN-major), (x_t_fp8 (cols, padded rows), scales (cols, padded rows / 128)))
    of `x`, or of `dsv4_rope(x, inverse=True)` with `rope`."""
    assert x.dim() == 2 and x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape[1] % BLOCK == 0
    rows = x.shape[0]
    out, sf, out_t, sf_t = _casts(x, *(rope if rope is not None else (None, None, BLOCK)))
    return (out, sf[:, :rows].T), (out_t, sf_t.T)
