from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class QuantizedExpertWeight:
    """An expert weight ``[G, K, N]`` (logical) already quantized: its NVFP4 forward operand and, for the ``mxfp8``
    backward, its MXFP8 input-gradient operand. Every tensor leads with the expert dimension, so the quantized forms
    of whole-expert shards concatenate along it: quantize each FSDP shard once and all-gather these instead of BF16."""

    forward: _NVFP4Tensor
    dgrad: tuple[torch.Tensor, torch.Tensor] | None = None
    """MXFP8 values ``[G, K, N]`` (stored ``[G, K, N]`` contiguous, quantized along N) and swizzled scales ``[G, ...]``."""

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The tensors to all-gather, each ``[G, ...]``."""
        forward = (self.forward.data.view(torch.uint8), self.forward.block_scales, self.forward.global_scales)
        return forward + (self.dgrad if self.dgrad is not None else ())

    @staticmethod
    def from_tensors(tensors: tuple[torch.Tensor, ...]) -> "QuantizedExpertWeight":
        data, block_scales, global_scales, *dgrad = tensors
        forward = _NVFP4Tensor(data.view(torch.float4_e2m1fn_x2), block_scales, global_scales, None)
        return QuantizedExpertWeight(forward, tuple(dgrad) if dgrad else None)


def quantize_expert_weight(
    weight_rows: torch.Tensor, *, four_over_six: bool, backward: NVFP4Backward
) -> QuantizedExpertWeight:
    """Quantize ``[G, N, K]`` expert weights (the ``[out, in]`` layout of a parameter, i.e. the transpose of the
    GEMM's logical ``[G, K, N]``) for ``grouped_gemm(weight_quantized=...)``: the same bytes ``grouped_gemm`` would
    make from the BF16 weight."""
    from prime_kernels.mxfp8_moe.grouped_gemm import _quantize_rows

    weight = weight_rows.transpose(-2, -1)
    forward = quantize_weights(weight, four_over_six=four_over_six)
    if backward != "mxfp8":
        return QuantizedExpertWeight(forward)
    from torchao.prototype.moe_training.kernels.mxfp8 import triton_mx_block_rearrange_per_group_3d

    data, scales = _quantize_rows(weight.contiguous())
    return QuantizedExpertWeight(forward, (data, triton_mx_block_rearrange_per_group_3d(scales)))


class _GroupedNVFP4MM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        backward: NVFP4Backward,
        four_over_six: bool,
        weight_quantized: QuantizedExpertWeight | None,
        activations: _NVFP4Tensor | None,
    ) -> torch.Tensor:
        activations_nvfp4 = activations or quantize_activations(matrix, offsets, four_over_six=four_over_six)
        if weight_quantized is not None:
            if backward == "bf16" or (backward == "mxfp8" and weight_quantized.dgrad is None):
                raise ValueError(f"a pre-quantized weight cannot serve the {backward!r} backward without its dgrad form")
            weight_nvfp4 = weight_quantized.forward
            # The weight may be a stand-in without BF16 data (an all-gathered quantized weight): never read it
            weight = torch.empty(0, device=matrix.device, dtype=matrix.dtype)
        else:
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
        ctx.weight_dgrad = weight_quantized.dgrad if weight_quantized is not None else None
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
            from prime_kernels.mxfp8_moe.grouped_gemm import _dgrad, _dgrad_weight, _wgrad, quantize_grad

            matrix, weight, offsets = ctx.saved_tensors
            # Both gradient GEMMs need grad_output: quantize it along rows and columns in one read
            rows = columns = None
            if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
                rows, columns = quantize_grad(grad_output)
            grad_matrix = grad_weight = None
            if ctx.needs_input_grad[0]:
                if ctx.weight_dgrad is not None:
                    data, scales = ctx.weight_dgrad
                    quantized = (data.transpose(-2, -1), scales)
                else:
                    quantized = _dgrad_weight(weight)
                grad_matrix = _dgrad(grad_output, weight, offsets, quantized, quantized_grad=rows)
            if ctx.needs_input_grad[1]:
                grad_weight = _wgrad(grad_output, matrix, offsets, high_precision=False, quantized_grad=columns)
            return grad_matrix, grad_weight, None, None, None, None, None
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
        return grad_matrix, grad_weight, None, None, None, None, None


def grouped_gemm(
    matrix: torch.Tensor,
    weight: torch.Tensor,
    *,
    offs: torch.Tensor,
    backward: NVFP4Backward = "dequant_bf16",
    four_over_six: bool = False,
    weight_quantized: QuantizedExpertWeight | None = None,
    activations: _NVFP4Tensor | None = None,
) -> torch.Tensor:
    """Grouped NVFP4 forward with the selected backward.

    ``matrix`` has shape ``[M, K]``, ``weight`` has shape ``[G, K, N]``, and
    ``offs`` contains the cumulative row count for each of the ``G`` groups.
    ``mxfp8`` runs both gradient GEMMs in MXFP8 on the original BF16 operands
    (needs ``mxfp8_moe``'s torchao kernels); ``dequant_bf16`` runs them in BF16
    on operands reconstructed from the packed forward tensors; ``bf16`` runs
    them in BF16 on the original operands.
    ``four_over_six`` enables FlashInfer's default 448-bound, MAE 4/6 recipe.

    ``weight_quantized``, ``weight`` already quantized (``quantize_expert_weight``, e.g. per FSDP shard before the
    all-gather), skips the weight quantizers; ``weight`` is then only read by the ``bf16`` backward, and its gradient
    is still returned for it.

    ``activations``, ``matrix`` already quantized (``dispatch.group_tokens`` of tokens quantized before dispatch),
    skips the activation quantizer; ``matrix`` stays the BF16 operand of the backward.
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
    return _GroupedNVFP4MM.apply(matrix, weight, offs, backward, four_over_six, weight_quantized, activations)
