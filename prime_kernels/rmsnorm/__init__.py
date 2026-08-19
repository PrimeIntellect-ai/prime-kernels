from __future__ import annotations

import torch

from . import _C  # noqa: F401  # imported for its side effect: registers the prime_rmsnorm ops

__all__ = ["fused_rmsnorm_residual"]

# One 512 byte scale tile per 128 row x 128 column block.
_SCALE_TILE = 512


def _scale_numel(n: int, c: int) -> int:
    return (n // 128) * (c // 128) * _SCALE_TILE


@torch.library.register_fake("prime_rmsnorm::fused_rmsnorm_residual")
def _fused_rmsnorm_residual_fake(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c = branch.shape
    device = branch.device
    return (
        torch.empty((n, c), dtype=torch.float8_e4m3fn, device=device),
        torch.empty(_scale_numel(n, c), dtype=torch.float8_e8m0fnu, device=device),
        torch.empty(n, dtype=torch.float32, device=device),
    )


def fused_rmsnorm_residual(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm over `branch`, `residual` added after the norm, output quantized to MXFP8.

    `branch` and `residual` are bf16 `(N, C)`, `weight` is bf16 `(C,)`; N and C must both be
    multiples of 128. Returns `(vals, scales, rrms)`:

    - `vals`  — fp8 e4m3 `(N, C)`, the quantized `rmsnorm(branch) * weight + residual`
    - `scales` — flat e8m0, one exponent per 32 values, in the blocked layout a tensor core
      GEMM expects: a sequence of 512 byte tiles, each covering a 128 x 128 block. This is
      the same layout `flash_moe.pack_scales_blocked` produces, already packed.
    - `rrms`  — fp32 `(N,)`, the reciprocal RMS per row, kept for the backward pass.

    Which of the kernel variants runs is chosen from C; see RESULTS.md for the measurements
    behind the boundaries.
    """
    return torch.ops.prime_rmsnorm.fused_rmsnorm_residual(branch, residual, weight, epsilon)
