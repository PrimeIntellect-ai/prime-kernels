"""Per-token FP8 (e4m3) quantization with power-of-two (UE8M0) scales.

Vendored from vLLM (Apache 2.0) by way of prime-rl's `fp8_indexer`, so the indexer here scores
with exactly the quantized operands prime-rl's Triton indexer uses.
"""

import torch
import triton
import triton.language as tl

FP8_MAX = tl.constexpr(448.0)
FP8_EPS = tl.constexpr(1e-10)


@triton.jit
def _per_token_quant_fp8_kernel(x_ptr, x_q_ptr, x_s_ptr, row_stride, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), FP8_EPS)
    scale = tl.math.exp2(tl.ceil(tl.log2(absmax / FP8_MAX)))
    x_q = tl.clamp(x / scale, -FP8_MAX, FP8_MAX).to(x_q_ptr.dtype.element_ty)
    tl.store(x_q_ptr + row * N + cols, x_q, mask=mask)
    tl.store(x_s_ptr + row, scale)


def per_token_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(..., N)` -> contiguous `(..., N)` float8_e4m3fn and `(...)` fp32 scales, one per row."""
    assert x.stride(-1) == 1
    rows = x.reshape(-1, x.shape[-1])
    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    x_s = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    if rows.shape[0]:
        block = triton.next_power_of_2(x.shape[-1])
        _per_token_quant_fp8_kernel[(rows.shape[0],)](
            rows, x_q, x_s, rows.stride(0), N=x.shape[-1], BLOCK=block, num_warps=min(max(block // 256, 1), 8)
        )
    return x_q, x_s
