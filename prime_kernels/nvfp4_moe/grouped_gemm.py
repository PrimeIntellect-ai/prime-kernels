from __future__ import annotations

from dataclasses import dataclass

import torch

from ._ops import _grouped_mm as _grouped_mm_kernel
from .quantize import (
    _NVFP4Tensor,
    quantize_activations,
    quantize_weights,
)


@dataclass(frozen=True)
class QuantizedExpertWeight:
    """An expert weight ``[G, K, N]`` (logical) already quantized to its NVFP4 forward operand. Every tensor leads with
    the expert dimension, so the quantized forms of whole-expert shards concatenate along it: quantize each FSDP shard
    once and all-gather these instead of BF16. The backward dequantizes it, so it needs nothing else."""

    forward: _NVFP4Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The tensors to all-gather, each ``[G, ...]``."""
        return (self.forward.data.view(torch.uint8), self.forward.block_scales, self.forward.global_scales)

    @staticmethod
    def from_tensors(tensors: tuple[torch.Tensor, ...]) -> "QuantizedExpertWeight":
        data, block_scales, global_scales = tensors
        return QuantizedExpertWeight(_NVFP4Tensor(data.view(torch.float4_e2m1fn_x2), block_scales, global_scales, None))


def quantize_expert_weight(weight_rows: torch.Tensor) -> QuantizedExpertWeight:
    """Quantize ``[G, N, K]`` expert weights (the ``[out, in]`` layout of a parameter, i.e. the transpose of the
    GEMM's logical ``[G, K, N]``) for ``grouped_gemm(weight_quantized=...)``: the same bytes ``grouped_gemm`` would
    make from the BF16 weight."""
    return QuantizedExpertWeight(quantize_weights(weight_rows.transpose(-2, -1)))


class _GroupedNVFP4MM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        weight_quantized: QuantizedExpertWeight | None,
        activations: _NVFP4Tensor | None,
    ) -> torch.Tensor:
        activations_nvfp4 = activations or quantize_activations(matrix, offsets)
        weight_nvfp4 = weight_quantized.forward if weight_quantized is not None else quantize_weights(weight)
        output = _grouped_mm_kernel(
            activations_nvfp4.data,
            weight_nvfp4.data.transpose(-2, -1),
            activations_nvfp4.block_scales,
            weight_nvfp4.block_scales,
            offsets,
            activations_nvfp4.global_scales,
            weight_nvfp4.global_scales,
        )
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
        # Both gradient GEMMs in MXFP8 (E4M3 values, E8M0 scales per 32 elements) on the forward's dequantized NVFP4
        # operands, so the gradients see the forward's rounding and clipping; every operand, the output gradient
        # included, is quantized with stochastic rounding
        from prime_kernels.mxfp8_moe.grouped_gemm import (
            _dgrad,
            _wgrad_quantized,
            triton_mx_block_rearrange_per_group_3d,
        )

        from . import stochastic

        matrix_data, matrix_scales, matrix_global, weight_data, weight_scales, weight_global, offsets = ctx.saved_tensors
        # The operands' call numbers, claimed whichever gradients are needed so a step always draws the same bits
        call = stochastic.next_calls(4)
        need_matrix, need_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_rows, grad_cols = stochastic.quantize_grad(
            grad_output.contiguous().bfloat16(), call + 1, call + 2, rows=need_matrix, cols=need_weight
        )
        grad_matrix = grad_weight = None
        if need_matrix:
            weight = _NVFP4Tensor(weight_data, weight_scales, weight_global, None)
            data, scales = stochastic.quantize_weight_for_dgrad(weight, call)
            quantized_weight = (data.transpose(-2, -1), triton_mx_block_rearrange_per_group_3d(scales))
            grad_matrix = _dgrad(grad_output, None, offsets, quantized_weight, grad_rows)
        if need_weight:
            matrix = _NVFP4Tensor(matrix_data, matrix_scales, matrix_global, offsets)
            grad_weight = _wgrad_quantized(grad_cols, stochastic.quantize_activation_columns(matrix, call + 3), offsets)
        return grad_matrix, grad_weight, None, None, None


def grouped_gemm(
    matrix: torch.Tensor,
    weight: torch.Tensor,
    *,
    offs: torch.Tensor,
    weight_quantized: QuantizedExpertWeight | None = None,
    activations: _NVFP4Tensor | None = None,
) -> torch.Tensor:
    """Grouped NVFP4 forward, MXFP8 backward.

    ``matrix`` has shape ``[M, K]``, ``weight`` has shape ``[G, K, N]``, and ``offs`` contains the cumulative row
    count for each of the ``G`` groups. The forward quantizes ``matrix`` with per-token scales and 4/6, ``weight`` with
    per-expert scales. The backward runs both gradient GEMMs in MXFP8 on the dequantized forward operands, with
    stochastic rounding seeded by ``stochastic.set_seed`` (needs ``mxfp8_moe``'s torchao kernels).

    ``weight_quantized``, ``weight`` already quantized (``quantize_expert_weight``, e.g. per FSDP shard before the
    all-gather), skips the weight quantizer; ``weight`` is then never read (it may be a stand-in without data), and its
    gradient is still returned for it.

    ``activations``, ``matrix`` already quantized (``dispatch.group_tokens`` of tokens quantized before dispatch),
    skips the activation quantizer; ``matrix`` is then never read either, and its gradient is still returned for it.
    """

    if matrix.ndim != 2 or weight.ndim != 3:
        raise ValueError("matrix must be 2D and weight must be 3D")
    if weight.shape[0] != offs.numel():
        raise ValueError("weight and offsets must have the same number of groups")
    if matrix.shape[1] != weight.shape[1]:
        raise ValueError("matrix and weight contraction dimensions must match")
    if weight.device != matrix.device or weight.dtype != matrix.dtype:
        raise ValueError("weight must have the same CUDA device and dtype as matrix")
    return _GroupedNVFP4MM.apply(matrix, weight, offs, weight_quantized, activations)
