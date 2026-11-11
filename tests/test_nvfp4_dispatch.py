import pytest
import torch

from prime_kernels.nvfp4_moe.dispatch import group_tokens, quantize_tokens, swizzle_scales, unswizzle_scales


def reference_index(r, c, scale_cols):
    """The 128x4-tile layout the GEMM reads, as tests/test_nvfp4_moe.py decodes it."""
    return (r // 128) * (128 * scale_cols) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4


@pytest.mark.parametrize("counts,k", [([32, 0, 160], 64), ([300], 2048), ([0, 1, 129, 64], 2688)])
def test_scale_swizzle_matches_the_gemm_layout_and_round_trips(counts, k):
    torch.manual_seed(0)
    rows = sum(counts)
    offsets = torch.tensor(counts, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    scales = torch.randint(1, 255, (rows, k // 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
    swizzled = swizzle_scales(scales, offsets, k)
    scale_cols = ((k // 16 + 3) // 4) * 4
    assert swizzled.shape == ((rows + len(counts) * 127 + 127) // 128 * 128, scale_cols)
    flat = swizzled.flatten().view(torch.uint8)
    start = region = 0
    for count in counts:
        for r in range(count):
            for c in range(k // 16):
                assert flat[region + reference_index(r, c, scale_cols)] == scales[start + r, c].view(torch.uint8)
        start += count
        region += (count + 127) // 128 * 128 * scale_cols
    assert torch.equal(unswizzle_scales(swizzled, offsets, rows, k).view(torch.uint8), scales.view(torch.uint8))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="NVFP4 quantization needs SM100/SM103",
)
@pytest.mark.parametrize("four_over_six", [False, True])
def test_quantizing_before_dispatch_gives_the_same_bytes(four_over_six):
    import prime_kernels

    kernel = prime_kernels.load("nvfp4_moe")
    torch.manual_seed(1729)
    tokens, k, n, experts, top_k = 200, 2048, 1536, 8, 2
    x = (torch.randn(tokens, k, device="cuda") * 0.3).bfloat16()
    route = torch.randint(0, experts, (tokens * top_k,), device="cuda")
    order = torch.argsort(route, stable=True)
    rows = (torch.arange(tokens, device="cuda").repeat_interleave(top_k))[order]
    counts = torch.bincount(route, minlength=experts)
    offsets = counts.cumsum(0, dtype=torch.int32)

    after = kernel.quantize_activations(x[rows], offsets, four_over_six=four_over_six)
    before = group_tokens(quantize_tokens(x, four_over_six=four_over_six).index_select(rows), offsets)
    assert torch.equal(before.data.view(torch.uint8), after.data.view(torch.uint8))
    assert torch.equal(before.global_scales, after.global_scales)
    valid = unswizzle_scales(after.block_scales, offsets, rows.numel(), k)
    assert torch.equal(unswizzle_scales(before.block_scales, offsets, rows.numel(), k).view(torch.uint8), valid.view(torch.uint8))

    w = (torch.randn(experts, n, k, device="cuda") * 0.02).bfloat16().transpose(-1, -2)
    expected = kernel.grouped_gemm(x[rows], w, offs=offsets, four_over_six=four_over_six)
    actual = kernel.grouped_gemm(x[rows], w, offs=offsets, four_over_six=four_over_six, activations=before)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
