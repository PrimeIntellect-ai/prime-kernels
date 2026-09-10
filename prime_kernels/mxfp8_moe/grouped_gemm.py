# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license in
# LICENSE.torchao. Derived from torchao commit 02105d46c.

from __future__ import annotations

import torch
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mx_block_rearrange_2d_M_groups_cuda,
    mxfp8_quantize_cuda_3d,
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_2d_M_groups,
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.mx_formats.config import MXFP8Dim1CastKernelChoice, ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import triton_mxfp8_dequant_dim0, triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.quantization.quantize_.common import KernelPreference

TOKEN_GROUP_ALIGNMENT = 32
_CUDA_REARRANGE_MAX_GROUPS = 32
_QUANT_NUMEL_LIMIT = 1 << 31
_QUANT_CHUNK_NUMEL = 1 << 30
_SCALING_MODE = ScaleCalculationMode.RCEIL


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or x.numel() < _QUANT_NUMEL_LIMIT:
        return triton_to_mxfp8_dim0(
            x,
            inner_block_size=TOKEN_GROUP_ALIGNMENT,
            scaling_mode=_SCALING_MODE.value.lower(),
        )

    rows_per_chunk = max(1, _QUANT_CHUNK_NUMEL // x.shape[1])
    quantized = [
        triton_to_mxfp8_dim0(
            chunk,
            inner_block_size=TOKEN_GROUP_ALIGNMENT,
            scaling_mode=_SCALING_MODE.value.lower(),
        )
        for chunk in x.split(rows_per_chunk, dim=0)
    ]
    return torch.cat([data for data, _ in quantized]), torch.cat([scales for _, scales in quantized])


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"MXFP8 row quantization expects a 2D tensor, got shape {tuple(x.shape)}")
    if x.shape[1] % TOKEN_GROUP_ALIGNMENT:
        raise ValueError(
            f"MXFP8 row quantization requires columns divisible by {TOKEN_GROUP_ALIGNMENT}, got {x.shape[1]}"
        )
    if x.shape[0] == 0:
        data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scales = torch.empty(
            (0, x.shape[1] // TOKEN_GROUP_ALIGNMENT),
            dtype=torch.float8_e8m0fnu,
            device=x.device,
        )
        return data, scales
    return _quantize_rows(x)


def dequantize_rows(data: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if data.shape[0] == 0:
        return torch.empty(data.shape, dtype=dtype, device=data.device)
    return triton_mxfp8_dequant_dim0(
        data,
        scales.view(torch.uint8),
        out_dtype=dtype,
        scale_block_size=TOKEN_GROUP_ALIGNMENT,
    )


def _rearrange_token_scales(scales: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if offsets.numel() > _CUDA_REARRANGE_MAX_GROUPS:
        return triton_mx_block_rearrange_2d_M_groups(scales, offsets)
    return mx_block_rearrange_2d_M_groups_cuda(scales, offsets)


def _quantize_dim1(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mx = _to_mxfp8_dim1_kernel_wrapper(
        x,
        TOKEN_GROUP_ALIGNMENT,
        elem_dtype=torch.float8_e4m3fn,
        hp_dtype=x.dtype,
        kernel_preference=KernelPreference.AUTO,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=_SCALING_MODE,
    )
    return mx.qdata, mx.scale


def _forward(input_act: torch.Tensor, weight_t: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    input_data, input_scales = quantize_rows(input_act)
    weight_data, weight_scales = _quantize_rows(weight_t.transpose(-2, -1))
    return torch._scaled_grouped_mm(
        input_data,
        weight_data.transpose(-2, -1),
        _rearrange_token_scales(input_scales, offsets),
        triton_mx_block_rearrange_per_group_3d(weight_scales),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )


def _dgrad(grad_output: torch.Tensor, weight_t: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    grad_data, grad_scales = quantize_rows(grad_output)
    weight_data, weight_scales = mxfp8_quantize_cuda_3d(
        weight_t.transpose(-2, -1),
        TOKEN_GROUP_ALIGNMENT,
        scaling_mode=_SCALING_MODE.value.lower(),
    )
    return torch._scaled_grouped_mm(
        grad_data,
        weight_data,
        _rearrange_token_scales(grad_scales, offsets),
        weight_scales,
        offs=offsets,
        out_dtype=torch.bfloat16,
    )


def _wgrad(
    grad_output: torch.Tensor,
    input_act: torch.Tensor,
    offsets: torch.Tensor,
    *,
    high_precision: bool,
) -> torch.Tensor:
    if high_precision:
        grad_weight = torch._grouped_mm(
            grad_output.transpose(-2, -1),
            input_act,
            offs=offsets,
            out_dtype=torch.bfloat16,
        )
        return grad_weight.transpose(-2, -1)

    grad_data, grad_scales = _quantize_dim1(grad_output)
    input_data, input_scales = _quantize_dim1(input_act)
    scale_offsets = offsets // TOKEN_GROUP_ALIGNMENT
    grad_weight = torch._scaled_grouped_mm(
        grad_data,
        input_data.transpose(-2, -1),
        triton_mx_block_rearrange_2d_K_groups(grad_scales, scale_offsets),
        triton_mx_block_rearrange_2d_K_groups(input_scales, scale_offsets),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    return grad_weight.transpose(-2, -1)


class _GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        weight_t: torch.Tensor,
        offsets: torch.Tensor,
        high_precision_wgrad: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(input_act, weight_t, offsets)
        ctx.high_precision_wgrad = high_precision_wgrad
        return _forward(input_act, weight_t, offsets)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_act, weight_t, offsets = ctx.saved_tensors
        grad_input = _dgrad(grad_output, weight_t, offsets) if ctx.needs_input_grad[0] else None
        grad_weight = (
            _wgrad(
                grad_output,
                input_act,
                offsets,
                high_precision=ctx.high_precision_wgrad,
            )
            if ctx.needs_input_grad[1]
            else None
        )
        return grad_input, grad_weight, None, None


def grouped_gemm(
    input_act: torch.Tensor,
    weight_t: torch.Tensor,
    offsets: torch.Tensor,
    *,
    high_precision_wgrad: bool = False,
) -> torch.Tensor:
    """Run differentiable MXFP8 grouped GEMM on already aligned token groups."""
    if input_act.ndim != 2 or weight_t.ndim != 3:
        raise ValueError(
            f"MXFP8 grouped GEMM expects 2D activations and 3D weights, got {input_act.ndim}D and {weight_t.ndim}D"
        )
    if offsets.dtype != torch.int32:
        raise ValueError(f"MXFP8 grouped GEMM offsets must be int32, got {offsets.dtype}")
    return _GroupedGemm.apply(input_act, weight_t, offsets, high_precision_wgrad)
