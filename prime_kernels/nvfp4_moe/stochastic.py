"""MXFP8 quantization with stochastic rounding, for the gradient GEMMs of the NVFP4 expert backward.

Scales are torchao's RCEIL E8M0 scales (``ceil(log2(amax / 448))``, exactly), so the outputs feed the same grouped
GEMMs and scale swizzles as its round-to-nearest quantizers. Each value then rounds to one of its two neighbouring
E4M3 values with probability in inverse proportion to its distance from it, so the quantization is unbiased in
expectation.

The CUDA kernels (``mxfp8_stochastic_*``) quantize in one pass: a gradient's row-wise and column-wise forms from one
read, and the forward's NVFP4 operands straight from their packed form. The ``reference_*`` functions here are the
same algorithm in PyTorch, bit for bit, for tests.

Deterministic: the random bits of output ``i`` of a quantization come from Philox4x32-10 keyed by the seed set by
``set_seed`` (once per training step) and counted by ``(i // 8, call)``, where ``call`` numbers the quantizations
since then. A backward claims its calls in a fixed order, so the same step reproduces the same bits.
"""

from __future__ import annotations

import torch

BLOCK = 32
_E4M3_MAX = 448.0

_seed = 0
_calls = 0


def set_seed(seed: int) -> None:
    """Seed the stochastic rounding of the next backward(s); call once per training step with a step-dependent seed."""
    global _seed, _calls
    _seed, _calls = seed, 0


def seed() -> int:
    """The seed ``set_seed`` set for this step."""
    return _seed


def next_calls(count: int) -> int:
    """Claim ``count`` consecutive call numbers; returns the first."""
    global _calls
    first, _calls = _calls, _calls + count
    return first


def _kernel_seed() -> int:
    """The seed as the signed 64-bit integer the ops take."""
    seed = _seed & (2**64 - 1)
    return seed - 2**64 if seed >= 2**63 else seed


def _library():
    from . import _ops  # noqa: F401  registers the ops

    return torch.ops.prime_rl


def quantize_grad(
    grad: torch.Tensor, call_rows: int, call_cols: int, *, rows: bool = True, cols: bool = True
) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, tuple[torch.Tensor, torch.Tensor] | None]:
    """BF16 ``[M, N]`` quantized along N (values ``[M, N]``, scales ``[M, N // 32]``) and/or along M (torchao's
    column layout: values ``[N, M]``, scales ``[N, M // 32]``), in one pass."""
    row_data, row_scales, col_data, col_scales = _library().mxfp8_stochastic_bf16(
        grad.contiguous(), _kernel_seed(), call_rows, call_cols, rows, cols
    )
    return (
        (_e4m3(row_data), _e8m0(row_scales)) if rows else None,
        (_e4m3(col_data), _e8m0(col_scales)) if cols else None,
    )


def quantize_activation_columns(activations, call: int) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 activations ``[M, K]`` (a ``quantize._NVFP4Tensor`` with offsets) dequantized and quantized along M:
    values ``[K, M]``, scales ``[K, M // 32]``."""
    data, scales = _library().mxfp8_stochastic_nvfp4_activations(
        activations.data.view(torch.uint8),
        activations.block_scales,
        activations.global_scales,
        activations.offsets,
        _kernel_seed(),
        call,
    )
    return _e4m3(data), _e8m0(scales)


def quantize_weight_for_dgrad(weight, call: int) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 expert weights (logical ``[G, K, N]``) dequantized and quantized along N: values ``[G, K, N]``, scales
    ``[G, K, N // 32]``."""
    data, scales = _library().mxfp8_stochastic_nvfp4_weights(
        weight.data.view(torch.uint8), weight.block_scales, weight.global_scales, _kernel_seed(), call
    )
    return _e4m3(data), _e8m0(scales)


def quantize_rows_sr(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``[M, N]`` quantized along N with the next call number."""
    return quantize_grad(x, next_calls(1), 0, cols=False)[0]


def quantize_columns_sr(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``[M, N]`` quantized along M with the next call number: values ``[N, M]``, scales ``[N, M // 32]``."""
    return quantize_grad(x, 0, next_calls(1), rows=False)[1]


def _e4m3(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.float8_e4m3fn)


def _e8m0(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.float8_e8m0fnu)


# ── reference ──────────────────────────────────────────────────────────────────────────────────────────────────────

_MASK32 = 0xFFFFFFFF


def _mul_hi_lo(a: int, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """High and low 32 bits of ``a * b`` for a 32-bit constant ``a`` and int64 tensor ``b`` of 32-bit values, without
    overflowing int64: ``a`` split into 16-bit halves."""
    a_hi, a_lo = a >> 16, a & 0xFFFF
    low_product, high_product = a_lo * b, a_hi * b  # each < 2^48
    lo = (low_product + ((high_product & 0xFFFF) << 16)) & _MASK32
    hi = (high_product + (low_product >> 16)) >> 16
    return hi & _MASK32, lo


def _philox(counter: torch.Tensor, call: int, seed: int) -> list[torch.Tensor]:
    """Philox4x32-10 of the 64-bit counters ``counter`` (int64) with the third counter word ``call``."""
    c0, c1 = counter & _MASK32, (counter >> 32) & _MASK32
    c2, c3 = torch.full_like(counter, call & _MASK32), torch.zeros_like(counter)
    k0, k1 = seed & _MASK32, (seed >> 32) & _MASK32
    for _ in range(10):
        hi0, lo0 = _mul_hi_lo(0xD2511F53, c0)
        hi1, lo1 = _mul_hi_lo(0xCD9E8D57, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0, k1 = (k0 + 0x9E3779B9) & _MASK32, (k1 + 0xBB67AE85) & _MASK32
    return [c0, c1, c2, c3]


def reference_draws(index: torch.Tensor, call: int, seed: int | None = None) -> torch.Tensor:
    """The 16-bit random draws of outputs ``index`` (int64) of quantization ``call``."""
    words = torch.stack(_philox(index // 8, call, (_seed if seed is None else seed) & (2**64 - 1)), dim=-1)
    position = index % 8
    word = words.gather(-1, (position // 2).unsqueeze(-1)).squeeze(-1)
    return (word >> (16 * (position % 2))) & 0xFFFF


def _rceil(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torchao's RCEIL E8M0 exponent of ``amax / 448`` (ceil(log2), exactly) and its reciprocal scale."""
    descale = amax / _E4M3_MAX
    mantissa, exponent = torch.frexp(descale)
    ceil_log2 = torch.where(mantissa == 0.5, exponent - 1, exponent)
    biased = torch.where(descale == 0, 0, ceil_log2.clamp(-127, 127) + 127)
    biased = torch.where(torch.isinf(descale), 254, biased)
    biased = torch.where(torch.isnan(descale), 255, biased).to(torch.int32)
    reciprocal = torch.where(
        biased == 0, 1.0, torch.where(biased == 254, 2.0**-127, torch.exp2((127 - biased).float()))
    )
    return biased.to(torch.uint8), torch.where(biased == 255, descale, reciprocal)


def _round_e4m3(scaled: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
    magnitude = scaled.abs()
    _, exponent = torch.frexp(magnitude)
    binade = torch.where(magnitude == 0, -6, (exponent - 1).clamp(min=-6))
    step = torch.ldexp(torch.ones_like(magnitude), binade - 3)
    inverse = torch.ldexp(torch.ones_like(magnitude), 3 - binade)
    rounded = (torch.floor(magnitude * inverse + draws.float() * 2.0**-16) * step).clamp(max=_E4M3_MAX)
    return rounded.copysign(scaled).to(torch.float8_e4m3fn)


def reference_rows(x: torch.Tensor, call: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` quantized to MXFP8 along its last dim as the kernels do: E4M3 values of ``x``'s shape (output ``i`` is
    ``x``'s element ``i`` in row-major order) and E8M0 scales ``[..., last // 32]``."""
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // BLOCK, BLOCK).float()
    exponent, reciprocal = _rceil(blocks.abs().amax(-1, keepdim=True))
    scaled = (blocks * reciprocal).clamp(-_E4M3_MAX, _E4M3_MAX)
    index = torch.arange(x.numel(), device=x.device).reshape(scaled.shape)
    data = _round_e4m3(scaled, reference_draws(index, call, seed)).reshape(x.shape)
    return data, exponent.reshape(*x.shape[:-1], -1).view(torch.float8_e8m0fnu)


def reference_columns(x: torch.Tensor, call: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``[M, N]`` ``x`` quantized along M: values ``[N, M]``, scales ``[N, M // 32]``."""
    return reference_rows(x.t().contiguous(), call, seed)
