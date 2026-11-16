import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("MXFP8 grouped GEMMs need SM100/SM103", allow_module_level=True)

from prime_kernels.mxfp8_moe.grouped_gemm import _quantize_dim1, quantize_grad, quantize_rows


@pytest.mark.parametrize("rows,cols", [(256, 2048), (4096, 1536), (96, 4096)])
def test_one_pass_grad_quantization_matches_two_passes(rows, cols):
    grad = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.01
    (row_data, row_scales), (column_data, column_scales) = quantize_grad(grad)
    expected_rows, expected_columns = quantize_rows(grad), _quantize_dim1(grad)
    for actual, expected in ((row_data, expected_rows[0]), (row_scales, expected_rows[1])):
        assert actual.shape == expected.shape
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for actual, expected in ((column_data, expected_columns[0]), (column_scales, expected_columns[1])):
        assert actual.shape == expected.shape and actual.stride() == expected.stride()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
