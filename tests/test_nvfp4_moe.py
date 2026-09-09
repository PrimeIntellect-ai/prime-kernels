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
def test_nvfp4_forward_and_backward_operands(kernel, k, n):
    torch.manual_seed(1729)
    counts = [32, 0, 64]
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    x = (torch.randn(sum(counts), k, device="cuda") * 0.2).bfloat16().requires_grad_()
    w = (torch.randn(3, n, k, device="cuda") * 0.2).bfloat16().transpose(-1, -2).requires_grad_()
    qx, qw = kernel.quantize_activations(x, offsets), kernel.quantize_weights(w)
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
        y = kernel.grouped_gemm(x, w, offs=offsets, backward=backward)
        relative_rms = ((y.float() - reference.float()).square().mean() / reference.float().square().mean()).sqrt()
        assert relative_rms < 0.01, relative_rms.item()
        y.backward(grad)
        operands = (dx, dw) if backward == "dequant_bf16" else (x, w)
        expected_dx = F.grouped_mm(grad, operands[1].transpose(-1, -2), offs=offsets, out_dtype=torch.bfloat16)
        expected_dw = F.grouped_mm(operands[0].T, grad, offs=offsets, out_dtype=torch.bfloat16)
        torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
        torch.testing.assert_close(w.grad, expected_dw, rtol=0, atol=0)


def test_nvfp4_token_locality_and_zero_rows(kernel):
    torch.manual_seed(41)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    x[0] = 0
    offsets = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
    first = kernel.quantize_activations(x, offsets)
    x[1] *= 1024
    second = kernel.quantize_activations(x, offsets)
    keep = torch.arange(64, device="cuda") != 1
    assert torch.equal(first.data.view(torch.uint8)[keep], second.data.view(torch.uint8)[keep])
    assert torch.equal(first.global_scales[keep], second.global_scales[keep])
    assert torch.isfinite(first.dequantize()).all()
    assert torch.count_nonzero(first.dequantize()[0]) == 0
    empty = kernel.grouped_gemm(
        x[:0],
        torch.zeros(2, 256, 256, device="cuda", dtype=torch.bfloat16),
        offs=torch.zeros(2, device="cuda", dtype=torch.int32),
    )
    assert empty.shape == (0, 256)
