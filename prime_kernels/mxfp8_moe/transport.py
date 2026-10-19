# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license in
# LICENSE.torchao. Derived from torchao commit 02105d46c.

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .grouped_gemm import dequantize_rows, quantize_rows


def _all_to_all(
    x: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: ProcessGroup,
) -> torch.Tensor:
    output = x.new_empty((sum(output_splits), *x.shape[1:]))
    dist.all_to_all_single(output, x.contiguous(), output_splits, input_splits, group=group)
    return output


def _quantized_all_to_all(
    x: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: ProcessGroup,
) -> torch.Tensor:
    data, scales = quantize_rows(x)
    output_data = _all_to_all(data, output_splits, input_splits, group)
    output_scales = _all_to_all(scales.view(torch.uint8), output_splits, input_splits, group)
    return dequantize_rows(output_data, output_scales.view(torch.float8_e8m0fnu), x.dtype)


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group: ProcessGroup,
    ) -> torch.Tensor:
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        return _quantized_all_to_all(x, output_splits, input_splits, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _all_to_all(grad_output, ctx.input_splits, ctx.output_splits, ctx.group)
        return grad_input, None, None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group: ProcessGroup,
    ) -> torch.Tensor:
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        return _all_to_all(x, output_splits, input_splits, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _quantized_all_to_all(grad_output, ctx.input_splits, ctx.output_splits, ctx.group)
        return grad_input, None, None, None


def all_to_all_dispatch(
    x: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: ProcessGroup,
) -> torch.Tensor:
    """Send expert inputs as MXFP8 and return explicit high-precision tensors."""
    return _Dispatch.apply(x, output_splits, input_splits, group)


def all_to_all_combine(
    x: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: ProcessGroup,
) -> torch.Tensor:
    """Combine expert outputs in high precision and send their gradients as MXFP8."""
    return _Combine.apply(x, output_splits, input_splits, group)
