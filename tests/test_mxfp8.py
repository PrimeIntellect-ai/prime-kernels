import prime_kernels
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="MXFP8 requires Blackwell",
)


@pytest.fixture
def kernel():
    return prime_kernels.load("mxfp8")


def reference(x):
    blocked = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = blocked.abs().amax(-1)
    _, binary_exponent = torch.frexp(amax.double())
    unbiased = binary_exponent - 9
    boundary = torch.ldexp(torch.full_like(amax, 448, dtype=torch.float64), unbiased)
    unbiased += (amax > boundary).to(torch.int32)
    exponent = torch.where(amax == 0, 0, unbiased + 127).clamp(0, 254)
    scale = exponent.to(torch.uint8)
    inv_scale = torch.ldexp(torch.ones_like(amax, dtype=torch.float64), 127 - exponent).float()
    inv_scale = torch.where(scale == 0, 0, inv_scale)
    q = (blocked * inv_scale.unsqueeze(-1)).reshape_as(x).to(torch.float8_e4m3fn)
    decoded = q.float().reshape_as(blocked) * torch.ldexp(torch.ones_like(amax), exponent - 127).unsqueeze(-1)
    return q, scale, decoded.reshape_as(x).bfloat16()


@pytest.mark.parametrize("shape", [(17, 96), (512, 4096), (3, 129, 256)])
@pytest.mark.parametrize("magnitude", [1.0, 0.02, 1e-20])
def test_nearest_cast_and_dequantize(kernel, shape, magnitude):
    torch.manual_seed(37)
    x = torch.randn((*shape[:-2], shape[-1], shape[-2]), device="cuda", dtype=torch.bfloat16).transpose(-1, -2)
    x = x * magnitude
    x[..., 0, :] = 0
    x[..., 1, :] = -0.0
    q, s = kernel.quantize(x)
    rq, rs, decoded = reference(x)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(s, rs)
    assert torch.equal(kernel.dequantize(q, s), decoded)


def test_all_finite_bf16_magnitudes_match_vllm(kernel):
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    bits = torch.arange(0x7F80, device="cuda", dtype=torch.int32).to(torch.uint16)
    x = bits.view(torch.bfloat16).unsqueeze(-1).repeat(1, 32)
    x[:, 1::2] *= -1
    q, s = kernel.quantize(x)
    vq, vs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    assert torch.equal(s, vs)
    assert torch.equal(q.view(torch.uint8), vq.view(torch.uint8))


def test_empty_input(kernel):
    x = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16)
    q, s = kernel.quantize(x)
    assert q.shape == x.shape
    assert s.shape == (0, 2)
    assert kernel.dequantize(q, s).shape == x.shape


def philox_uniform(shape, seed, salt, offset):
    index = torch.arange(torch.Size(shape).numel(), device="cuda", dtype=torch.int64) + offset
    mask = (1 << 32) - 1
    c0, c1 = index & mask, (index >> 32) & mask
    c2, c3 = torch.zeros_like(c0), torch.zeros_like(c0)
    stream = seed ^ salt
    k0, k1 = stream & mask, stream >> 32
    for _ in range(10):
        a, b = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = ((b >> 32) & mask) ^ c1 ^ k0, b & mask, ((a >> 32) & mask) ^ c3 ^ k1, a & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    signed = c0.to(torch.int32).to(torch.int64)
    magnitude = torch.where(signed < 0, -signed - 1, signed)
    return (magnitude.float() * 4.6566127342e-10).reshape(shape)


@pytest.mark.parametrize("offset", [0, (1 << 35) + 443])
def test_sr_matches_independent_neighbors_and_philox(kernel, offset):
    torch.manual_seed(47)
    x = torch.randn((4, 256, 129), device="cuda", dtype=torch.bfloat16).transpose(1, 2) * 0.02
    x[:, 0] = -0.0
    seed, salt = (1 << 54) + 57, 19
    kwargs = dict(rounding_seed=torch.tensor(seed, device="cuda"), rounding_salt=torch.tensor(salt, device="cuda"))
    q, scales = kernel.quantize(x, rounding_offset=offset, **kwargs)
    nearest, expected_scales, _ = reference(x)
    inverse = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), 127 - scales.int())
    inverse = torch.where(scales == 0, 0, inverse).repeat_interleave(32, -1)
    scaled = x.float() * inverse
    bits = nearest.view(torch.uint8).to(torch.int16)
    magnitude = bits & 127
    farther = scaled.abs() > nearest.float().abs()
    neighbor_bits = (bits & 128) | torch.where(farther, magnitude + 1, magnitude - 1).clamp(0, 126)
    neighbor = neighbor_bits.to(torch.uint8).view(torch.float8_e4m3fn).float()
    lower = torch.minimum(nearest.float().abs(), neighbor.abs())
    upper = torch.maximum(nearest.float().abs(), neighbor.abs())
    probability = torch.where(upper != lower, (scaled.abs() - lower) / (upper - lower), 0)
    expected = torch.copysign(
        torch.where(philox_uniform(x.shape, seed, salt, offset) < probability, upper, lower), scaled
    ).to(torch.float8_e4m3fn)
    assert torch.equal(scales, expected_scales)
    assert torch.equal(q.view(torch.uint8), expected.view(torch.uint8))
    torch.rand(1357, device="cuda")
    for start in (0, 2):
        shard, shard_scales = kernel.quantize(
            x[start : start + 2], rounding_offset=offset + start * 129 * 256, **kwargs
        )
        assert torch.equal(shard.view(torch.uint8), q[start : start + 2].view(torch.uint8))
        assert torch.equal(shard_scales, scales[start : start + 2])
    changed, _ = kernel.quantize(x, rounding_offset=offset, **(kwargs | {"rounding_seed": kwargs["rounding_seed"] + 1}))
    assert not torch.equal(changed.view(torch.uint8), q.view(torch.uint8))


def test_sr_preserves_grid_and_is_unbiased(kernel):
    kwargs = dict(rounding_seed=torch.tensor(47, device="cuda"), rounding_salt=torch.tensor(19, device="cuda"))
    x = torch.full((8192, 32), 1.0625, device="cuda", dtype=torch.bfloat16)
    x[:, 0] = 448
    x[::2] *= -1
    q, _ = kernel.quantize(x, **kwargs)
    assert abs(q.float()[:, 1:].abs().mean().item() - 1.0625) < 1e-3
    grid = q.bfloat16()
    again, _ = kernel.quantize(grid, **kwargs)
    assert torch.equal(again.view(torch.uint8), q.view(torch.uint8))
