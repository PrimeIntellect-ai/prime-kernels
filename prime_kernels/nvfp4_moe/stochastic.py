"""MXFP8 quantization with stochastic rounding, for the gradient GEMMs of the NVFP4 expert backward.

Scales are torchao's RCEIL E8M0 scales, so the outputs feed the same grouped GEMMs and scale swizzles as its
round-to-nearest quantizers. Each value then rounds to one of its two neighbouring E4M3 values with probability in
inverse proportion to its distance from it, so the quantization is unbiased in expectation.

Deterministic: every call draws from its own generator, seeded from the seed set by ``set_seed`` (once per training
step) and the call's index since then. A backward runs its quantizations in the same order on every replay, so the
same step reproduces the same bits.
"""

from __future__ import annotations

import torch

BLOCK = 32
_E4M3_MAX = 448.0
_E4M3_MANTISSA_BITS = 3
_E4M3_MIN_NORMAL_EXPONENT = -6

_seed = 0
_calls = 0


def set_seed(seed: int) -> None:
    """Seed the stochastic rounding of the next backward(s); call once per training step with a step-dependent seed."""
    global _seed, _calls
    _seed, _calls = seed, 0


def _generator(device: torch.device) -> torch.Generator:
    global _calls
    generator = torch.Generator(device=device)
    generator.manual_seed(hash((_seed, _calls)) & (2**63 - 1))
    _calls += 1
    return generator


def _round_e4m3(scaled: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Stochastically round FP32 ``scaled`` (within the E4M3 range) to E4M3: floor to the value's E4M3 grid step after
    adding uniform noise of one step."""
    magnitude = scaled.abs()
    exponent = torch.floor(torch.log2(magnitude)).clamp(min=_E4M3_MIN_NORMAL_EXPONENT)
    step = torch.exp2(exponent - _E4M3_MANTISSA_BITS)
    noise = torch.rand(scaled.shape, generator=generator, device=scaled.device)
    rounded = (torch.floor(magnitude / step + noise) * step).clamp(max=_E4M3_MAX)
    return rounded.copysign(scaled).to(torch.float8_e4m3fn)


def quantize_rows_sr(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` quantized to MXFP8 along its last dim (32-element blocks), stochastically rounded: E4M3 values of ``x``'s
    shape and E8M0 scales ``[..., last // 32]``."""
    from torchao.prototype.mx_formats.mx_tensor import _to_mx_rceil

    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // BLOCK, BLOCK).float()
    exponent, scaled = _to_mx_rceil(blocks, blocks.abs().amax(-1, keepdim=True), _E4M3_MAX)
    data = _round_e4m3(scaled, _generator(x.device)).reshape(x.shape)
    return data, exponent.reshape(*x.shape[:-1], -1).view(torch.float8_e8m0fnu)


def quantize_columns_sr(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[M, N]`` ``x`` quantized along M (the token dim of a weight gradient), stochastically rounded, laid out as
    torchao's column quantizer lays it out: E4M3 values ``[N, M]`` and E8M0 scales ``[N, M // 32]``."""
    return quantize_rows_sr(x.t().contiguous())
