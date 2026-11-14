import pytest
import torch

from prime_kernels.nvfp4_moe import stochastic


def dequantize(data, scales):
    exponent = scales.view(torch.uint8).float() - 127
    return data.float() * torch.exp2(exponent).repeat_interleave(stochastic.BLOCK, dim=-1)


def test_philox_matches_random123():
    words = stochastic._philox(torch.zeros(1, dtype=torch.int64), 0, 0)
    assert [hex(int(word)) for word in words] == ["0x6627e8d5", "0xe169c58d", "0xbc57ac4c", "0x9b00dbd8"]


def test_stochastic_rounding_is_unbiased_on_the_e4m3_grid_and_seeded():
    torch.manual_seed(0)
    x = torch.randn(4, 64) * torch.logspace(-8, 2, 64)  # normal and subnormal E4M3 values after scaling
    draws = [stochastic.reference_rows(x, call, seed=1) for call in range(2000)]
    values = torch.stack([dequantize(data, scales) for data, scales in draws])
    # Every draw is one of the two E4M3 neighbours of the round-to-nearest grid around x
    lower, upper = values.amin(0), values.amax(0)
    assert torch.all(lower <= x) and torch.all(x <= upper)
    # Unbiased: the mean over draws converges on x, well within the grid step
    step = (upper - lower).clamp(min=1e-30)
    assert ((values.mean(0) - x).abs() / step).max() < 0.1
    # Seeded: the same seed and call give the same bits, another seed other ones
    again = stochastic.reference_rows(x, 0, seed=1)
    assert torch.equal(again[0].view(torch.uint8), draws[0][0].view(torch.uint8))
    assert not torch.equal(stochastic.reference_rows(x, 0, seed=2)[0].view(torch.uint8), again[0].view(torch.uint8))


def test_scales_are_torchaos_rceil_scales():
    torchao = pytest.importorskip("torchao")
    from torchao.prototype.mx_formats.mx_tensor import _to_mx_rceil

    x = torch.randn(8, 256) * torch.logspace(-20, 20, 256)
    blocks = x.reshape(8, 8, 32)
    expected, _ = _to_mx_rceil(blocks, blocks.abs().amax(-1, keepdim=True), 448.0)
    _, scales = stochastic.reference_rows(x, 0, seed=0)
    assert torch.equal(scales.view(torch.uint8), expected.reshape(8, 8))
    del torchao


def test_column_quantization_uses_the_token_dim():
    x = torch.randn(64, 96)
    data, scales = stochastic.reference_columns(x, 0, seed=0)
    assert data.shape == (96, 64) and scales.shape == (96, 2)
    torch.testing.assert_close(dequantize(data, scales).T, x, rtol=0.13, atol=1e-6)


blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the stochastic MXFP8 kernels need SM100/SM103",
)


def same_bits(actual, expected):
    for a, e in zip(actual, expected):
        assert a.shape == e.shape and torch.equal(a.view(torch.uint8), e.view(torch.uint8))


@blackwell
def test_kernels_match_the_reference_bit_for_bit():
    import prime_kernels

    nvfp4 = prime_kernels.load("nvfp4_moe")
    torch.manual_seed(3)
    stochastic.set_seed(5)
    grad = (torch.randn(256, 384, device="cuda") * torch.logspace(-6, 3, 384, device="cuda")).bfloat16()
    rows, cols = stochastic.quantize_grad(grad, 3, 4)
    same_bits(rows, stochastic.reference_rows(grad, 3))
    same_bits(cols, stochastic.reference_columns(grad, 4))

    counts = [32, 0, 64, 160]
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    activations = nvfp4.quantize_activations(torch.randn(256, 2048, device="cuda").bfloat16(), offsets)
    same_bits(
        stochastic.quantize_activation_columns(activations, 7),
        stochastic.reference_columns(activations.dequantize(), 7),
    )
    weights = nvfp4.quantize_weights((torch.randn(4, 1536, 2048, device="cuda") * 0.2).bfloat16().transpose(-1, -2))
    same_bits(
        stochastic.quantize_weight_for_dgrad(weights, 9),
        stochastic.reference_rows(weights.dequantize().contiguous(), 9),
    )


@blackwell
def test_kernel_rounding_is_unbiased():
    torch.manual_seed(4)
    x = (torch.randn(32, 256, device="cuda") * torch.logspace(-4, 2, 256, device="cuda")).bfloat16()
    stochastic.set_seed(0)
    values = torch.stack([dequantize(*stochastic.quantize_grad(x, call, 0, cols=False)[0]) for call in range(1024)])
    step = (values.amax(0) - values.amin(0)).clamp(min=1e-30)
    assert ((values.mean(0) - x.float()).abs() / step).max() < 0.15
