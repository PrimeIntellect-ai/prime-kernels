import prime_kernels
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def kernel():
    return prime_kernels.load("nvfp4_moe")


def decode_rows(data, scales, global_scales, rows, k):
    """Decode the documented E2M1 and 128-by-4 scale layout without CUDA helpers."""
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    packed = data.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    fp4 = values[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    scale_cols = ((k // 16 + 3) // 4) * 4
    r = torch.arange(rows, device=data.device)[:, None]
    c = torch.arange(k // 16, device=data.device)[None, :]
    index = (r // 128) * (128 * scale_cols) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    block_scales = scales.flatten().float()[index]
    multiplier = block_scales * global_scales
    return (fp4 * multiplier.repeat_interleave(16, dim=-1)).bfloat16()


@pytest.mark.parametrize("k,n", [(64, 96), (2048, 1536), (2688, 3712)])
@pytest.mark.parametrize("four_over_six", [False, True])
def test_nvfp4_forward_and_backward_operands(kernel, k, n, four_over_six):
    torch.manual_seed(1729)
    counts = [32, 0, 64]
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    x = (torch.randn(sum(counts), k, device="cuda") * 0.2).bfloat16().requires_grad_()
    w = (torch.randn(3, n, k, device="cuda") * 0.2).bfloat16().transpose(-1, -2).requires_grad_()
    qx = kernel.quantize_activations(x, offsets, four_over_six=four_over_six)
    qw = kernel.quantize_weights(w, four_over_six=four_over_six)
    dx, dw = qx.dequantize(), qw.dequantize()
    scale_cols = ((k // 16 + 3) // 4) * 4
    row_start = scale_start = 0
    for count in counts:
        if count:
            reference = decode_rows(
                qx.data[row_start : row_start + count],
                qx.block_scales.flatten()[scale_start:],
                qx.global_scales[row_start : row_start + count, None],
                count,
                k,
            )
            torch.testing.assert_close(dx[row_start : row_start + count], reference, rtol=0, atol=0)
        row_start += count
        scale_start += ((count + 127) // 128) * 128 * scale_cols
    for expert in range(3):
        reference = decode_rows(qw.data[expert], qw.block_scales[expert], qw.global_scales[expert], n, k)
        torch.testing.assert_close(dw[expert].T, reference, rtol=0, atol=0)
    reference = F.grouped_mm(dx, dw, offs=offsets, out_dtype=torch.bfloat16)
    grad = torch.randn_like(reference)
    for backward in ("dequant_bf16", "bf16"):
        x.grad = w.grad = None
        y = kernel.grouped_gemm(x, w, offs=offsets, backward=backward, four_over_six=four_over_six)
        relative_rms = ((y.float() - reference.float()).square().mean() / reference.float().square().mean()).sqrt()
        assert relative_rms < 0.01, relative_rms.item()
        y.backward(grad)
        operands = (dx, dw) if backward == "dequant_bf16" else (x, w)
        expected_dx = F.grouped_mm(grad, operands[1].transpose(-1, -2), offs=offsets, out_dtype=torch.bfloat16)
        expected_dw = F.grouped_mm(operands[0].T, grad, offs=offsets, out_dtype=torch.bfloat16)
        torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
        torch.testing.assert_close(w.grad, expected_dw, rtol=0, atol=0)

    # MXFP8 gradient GEMMs on the dequantized operands approximate the dequantized BF16 gradients
    x.grad = w.grad = None
    kernel.grouped_gemm(x, w, offs=offsets, backward="dequant_mxfp8", four_over_six=four_over_six).backward(grad)
    expected_dx = F.grouped_mm(grad, dw.transpose(-1, -2), offs=offsets, out_dtype=torch.bfloat16)
    expected_dw = F.grouped_mm(dx.T, grad, offs=offsets, out_dtype=torch.bfloat16)
    for actual, expected in ((x.grad, expected_dx), (w.grad, expected_dw)):
        error = ((actual.float() - expected.float()).square().mean() / expected.float().square().mean()).sqrt()
        assert error < 0.05, error.item()

    # MXFP8 gradient GEMMs approximate the BF16 gradients on the original operands
    x.grad = w.grad = None
    kernel.grouped_gemm(x, w, offs=offsets, backward="mxfp8", four_over_six=four_over_six).backward(grad)
    expected_dx = F.grouped_mm(grad, w.transpose(-1, -2), offs=offsets, out_dtype=torch.bfloat16)
    expected_dw = F.grouped_mm(x.T, grad, offs=offsets, out_dtype=torch.bfloat16)
    for actual, expected in ((x.grad, expected_dx), (w.grad, expected_dw)):
        error = ((actual.float() - expected.float()).square().mean() / expected.float().square().mean()).sqrt()
        assert error < 0.05, error.item()


@pytest.mark.parametrize("backward", ["mxfp8", "dequant_mxfp8", "dequant_bf16"])
def test_pre_quantized_shards_match_the_bf16_weight(kernel, backward):
    """FSDP's quantized all-gather: each rank quantizes its whole-expert shard, the pieces concatenate along the
    expert dimension, and the GEMM on them matches the GEMM on the BF16 weight bit for bit."""
    from prime_kernels.nvfp4_moe.grouped_gemm import QuantizedExpertWeight, quantize_expert_weight

    torch.manual_seed(7)
    counts, k, n = [64, 32, 0, 96], 2048, 1536
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    x = (torch.randn(sum(counts), k, device="cuda") * 0.2).bfloat16().requires_grad_()
    rows = (torch.randn(4, n, k, device="cuda") * 0.2).bfloat16()  # the parameter's [experts, out, in] layout
    w = rows.transpose(-1, -2).requires_grad_()
    grad = torch.randn(sum(counts), n, device="cuda", dtype=torch.bfloat16)

    y = kernel.grouped_gemm(x, w, offs=offsets, backward=backward, four_over_six=True)
    y.backward(grad)
    expected = (y.detach(), x.grad.clone(), w.grad.clone())

    shards = [quantize_expert_weight(rows[i : i + 2], four_over_six=True, backward=backward) for i in (0, 2)]
    gathered = QuantizedExpertWeight.from_tensors(
        tuple(torch.cat(parts) for parts in zip(*(shard.tensors() for shard in shards)))
    )
    x.grad = w.grad = None
    y = kernel.grouped_gemm(x, w, offs=offsets, backward=backward, four_over_six=True, weight_quantized=gathered)
    y.backward(grad)
    for actual, wanted in zip((y.detach(), x.grad, w.grad), expected):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("four_over_six", [False, True])
def test_nvfp4_token_locality_and_zero_rows(kernel, four_over_six):
    torch.manual_seed(41)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    x[0] = 0
    offsets = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
    first = kernel.quantize_activations(x, offsets, four_over_six=four_over_six)
    x[1] *= 1024
    second = kernel.quantize_activations(x, offsets, four_over_six=four_over_six)
    keep = torch.arange(64, device="cuda") != 1
    assert torch.equal(first.data.view(torch.uint8)[keep], second.data.view(torch.uint8)[keep])
    assert torch.equal(first.global_scales[keep], second.global_scales[keep])
    assert torch.isfinite(first.dequantize()).all()
    assert torch.count_nonzero(first.dequantize()[0]) == 0
    empty = kernel.grouped_gemm(
        x[:0],
        torch.zeros(2, 256, 256, device="cuda", dtype=torch.bfloat16),
        offs=torch.zeros(2, device="cuda", dtype=torch.int32),
        four_over_six=four_over_six,
    )
    assert empty.shape == (0, 256)


@pytest.mark.parametrize("grad_scale", [1.0, 1e-10, 1e-12])
def test_mxfp8_backward_keeps_rl_sized_gradients(kernel, grad_scale):
    """prime-rl's RL loss normalization puts gradients at 1e-9 to 1e-11. A quantizer scale floor (like the 2^-22 one
    that zeroed prime-mega-moe's MXFP8 expert gradients in INTELLECT-4 RL) would flush them: the MXFP8 gradients must
    stay as accurate, relative to their size, at every scale."""
    torch.manual_seed(3)
    counts, k, n = [64, 96], 2048, 1536
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    x = (torch.randn(sum(counts), k, device="cuda") * 0.2).bfloat16().requires_grad_()
    w = (torch.randn(2, n, k, device="cuda") * 0.02).bfloat16().transpose(-1, -2).requires_grad_()
    grad = (torch.randn(sum(counts), n, device="cuda") * grad_scale).bfloat16()
    kernel.grouped_gemm(x, w, offs=offsets, backward="mxfp8", four_over_six=True).backward(grad)
    expected_dx = F.grouped_mm(grad.float(), w.float().transpose(-1, -2), offs=offsets)
    expected_dw = F.grouped_mm(x.float().T, grad.float(), offs=offsets)
    for actual, expected in ((x.grad, expected_dx), (w.grad, expected_dw)):
        error = ((actual.float() - expected).square().mean() / expected.square().mean()).sqrt()
        assert error < 0.05, (grad_scale, error.item())
