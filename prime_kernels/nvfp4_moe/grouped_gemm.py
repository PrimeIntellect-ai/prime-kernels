from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from ._ops import _grouped_mm as _grouped_mm_kernel
from .quantize import (
    _NVFP4Tensor,
    quantize_activations,
    quantize_weights,
)

NVFP4Backward = Literal["mxfp8", "dequant_bf16", "bf16"]


class _GroupedNVFP4MM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        backward: NVFP4Backward,
        four_over_six: bool,
    ) -> torch.Tensor:
        activations_nvfp4 = quantize_activations(matrix, offsets, four_over_six=four_over_six)
        weight_nvfp4 = quantize_weights(weight, four_over_six=four_over_six)
        output = _grouped_mm_kernel(
            activations_nvfp4.data,
            weight_nvfp4.data.transpose(-2, -1),
            activations_nvfp4.block_scales,
            weight_nvfp4.block_scales,
            offsets,
            activations_nvfp4.global_scales,
            weight_nvfp4.global_scales,
        )
        ctx.backward_mode = backward
        if backward in ("bf16", "mxfp8"):
            ctx.save_for_backward(matrix, weight, offsets)
        else:
            ctx.save_for_backward(
                activations_nvfp4.data,
                activations_nvfp4.block_scales,
                activations_nvfp4.global_scales,
                weight_nvfp4.data,
                weight_nvfp4.block_scales,
                weight_nvfp4.global_scales,
                offsets,
            )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous().bfloat16()
        if ctx.backward_mode == "mxfp8":
            # MXFP8 gradient GEMMs on the original BF16 operands (E4M3 values, E8M0 scales per 32 elements): the
            # backward never meets the sampler, so it trades exactness for speed
            from prime_kernels.mxfp8_moe.grouped_gemm import _dgrad, _wgrad

            matrix, weight, offsets = ctx.saved_tensors
            grad_matrix = _dgrad(grad_output, weight, offsets) if ctx.needs_input_grad[0] else None
            grad_weight = (
                _wgrad(grad_output, matrix, offsets, high_precision=False) if ctx.needs_input_grad[1] else None
            )
            return grad_matrix, grad_weight, None, None, None
        if ctx.backward_mode == "bf16":
            matrix, weight, offsets = ctx.saved_tensors
        else:
            (
                matrix_data,
                matrix_block_scales,
                matrix_global_scales,
                weight_data,
                weight_block_scales,
                weight_global_scales,
                offsets,
            ) = ctx.saved_tensors

        grad_matrix = grad_weight = None
        if ctx.needs_input_grad[0]:
            if ctx.backward_mode == "dequant_bf16":
                weight = _NVFP4Tensor(
                    weight_data,
                    weight_block_scales,
                    weight_global_scales,
                    None,
                ).dequantize()
            grad_matrix = F.grouped_mm(
                grad_output,
                weight.transpose(-2, -1),
                offs=offsets,
                out_dtype=torch.bfloat16,
            )
        if ctx.needs_input_grad[1]:
            if ctx.backward_mode == "dequant_bf16":
                matrix = _NVFP4Tensor(
                    matrix_data,
                    matrix_block_scales,
                    matrix_global_scales,
                    offsets,
                ).dequantize()
            grad_weight = F.grouped_mm(
                matrix.transpose(0, 1),
                grad_output,
                offs=offsets,
                out_dtype=torch.bfloat16,
            )
        return grad_matrix, grad_weight, None, None, None


def grouped_gemm(
    matrix: torch.Tensor,
    weight: torch.Tensor,
    *,
    offs: torch.Tensor,
    backward: NVFP4Backward = "dequant_bf16",
    four_over_six: bool = False,
) -> torch.Tensor:
    """Grouped NVFP4 forward with the selected backward.

    ``matrix`` has shape ``[M, K]``, ``weight`` has shape ``[G, K, N]``, and
    ``offs`` contains the cumulative row count for each of the ``G`` groups.
    ``mxfp8`` runs both gradient GEMMs in MXFP8 on the original BF16 operands
    (needs ``mxfp8_moe``'s torchao kernels); ``dequant_bf16`` runs them in BF16
    on operands reconstructed from the packed forward tensors; ``bf16`` runs
    them in BF16 on the original operands.
    ``four_over_six`` enables FlashInfer's default 448-bound, MAE 4/6 recipe.
    """

    if backward not in ("mxfp8", "dequant_bf16", "bf16"):
        raise ValueError(f"unsupported NVFP4 backward mode: {backward}")
    if matrix.ndim != 2 or weight.ndim != 3:
        raise ValueError("matrix must be 2D and weight must be 3D")
    if weight.shape[0] != offs.numel():
        raise ValueError("weight and offsets must have the same number of groups")
    if matrix.shape[1] != weight.shape[1]:
        raise ValueError("matrix and weight contraction dimensions must match")
    if weight.device != matrix.device or weight.dtype != matrix.dtype:
        raise ValueError("weight must have the same CUDA device and dtype as matrix")
    return _GroupedNVFP4MM.apply(matrix, weight, offs, backward, four_over_six)
