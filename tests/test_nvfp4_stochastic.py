import pytest
import torch

torchao = pytest.importorskip("torchao")

from prime_kernels.nvfp4_moe import stochastic  # noqa: E402


def dequantize(data, scales):
    exponent = scales.view(torch.uint8).float() - 127
    return data.float() * torch.exp2(exponent).repeat_interleave(stochastic.BLOCK, dim=-1)


def test_stochastic_rounding_is_unbiased_on_the_e4m3_grid_and_seeded():
    torch.manual_seed(0)
    x = torch.randn(4, 64) * torch.logspace(-8, 2, 64)  # normal and subnormal E4M3 values after scaling
    stochastic.set_seed(1)
    draws = [stochastic.quantize_rows_sr(x) for _ in range(2000)]
    values = torch.stack([dequantize(data, scales) for data, scales in draws])
    # Every draw is one of the two E4M3 neighbours of the round-to-nearest grid around x
    lower, upper = values.amin(0), values.amax(0)
    assert torch.all(lower <= x) and torch.all(x <= upper)
    # Unbiased: the mean over draws converges on x, well within the grid step
    step = (upper - lower).clamp(min=1e-30)
    assert ((values.mean(0) - x).abs() / step).max() < 0.1
    # Seeded: the same seed and call order give the same bits
    stochastic.set_seed(1)
    again = stochastic.quantize_rows_sr(x)
    assert torch.equal(again[0].view(torch.uint8), draws[0][0].view(torch.uint8))
    assert torch.equal(again[1].view(torch.uint8), draws[0][1].view(torch.uint8))


def test_column_quantization_uses_the_token_dim():
    x = torch.randn(64, 96)
    stochastic.set_seed(0)
    data, scales = stochastic.quantize_columns_sr(x)
    assert data.shape == (96, 64) and scales.shape == (96, 2)
    torch.testing.assert_close(dequantize(data, scales).T, x, rtol=0.13, atol=1e-6)
