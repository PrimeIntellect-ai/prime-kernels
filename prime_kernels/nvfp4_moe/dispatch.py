"""Quantize tokens once, before they are dispatched to their experts.

Activation scales are per token (one FP32 global scale per row, E4M3 scales per 16-element block of the row), so a
token's NVFP4 form does not depend on which expert it goes to: quantizing before the permutation (and the EP
all-to-all) gives the same bytes as quantizing each expert's rows after it, with ``top_k`` times less quantizer work
and ~3.5x fewer bytes moved. Only the block scales' layout is per group: the GEMM reads them swizzled in 128-row
tiles that start fresh at every group. Tokens therefore travel with row-major scales (``NVFP4Tokens``), and
``group_tokens`` swizzles them once the rows sit in expert order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .quantize import _NVFP4Tensor, quantize_activations

_BLOCK = 16
_TILE_ROWS = 128


def _scale_columns(contraction_size: int) -> int:
    return (contraction_size // _BLOCK + 3) // 4 * 4


def _swizzled_index(rows: torch.Tensor, columns: torch.Tensor, scale_columns: int) -> torch.Tensor:
    """Offset of (row within its group, scale column) inside the group's swizzled scale region: 128x4 tiles of
    32x16-byte atoms, the layout the NVFP4 GEMM reads."""
    return (
        (rows // _TILE_ROWS) * (_TILE_ROWS * scale_columns)
        + (columns // 4) * 512
        + (rows % 32) * 16
        + ((rows % _TILE_ROWS) // 32) * 4
        + columns % 4
    )


def _group_layout(offsets: torch.Tensor, rows: int, scale_columns: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's position within its group, and the offset of its group's scale region."""
    offsets = offsets.long()
    row = torch.arange(rows, device=offsets.device)
    group = torch.searchsorted(offsets, row, right=True)
    starts = torch.cat([offsets.new_zeros(1), offsets[:-1]])
    counts = offsets - starts
    region = (counts + _TILE_ROWS - 1) // _TILE_ROWS * _TILE_ROWS * scale_columns
    region_start = torch.cumsum(region, 0) - region
    return row - starts[group], region_start[group]


@dataclass(frozen=True)
class NVFP4Tokens:
    """Tokens quantized to NVFP4 with row-major scales, so any row permutation or all-to-all can move them."""

    data: torch.Tensor
    """``[M, K // 2]`` uint8, two E2M1 values per byte."""
    block_scales: torch.Tensor
    """``[M, K // 16]`` E4M3, one per 16-element block of the row."""
    global_scales: torch.Tensor
    """``[M]`` FP32 decode scale per token."""

    def index_select(self, index: torch.Tensor) -> "NVFP4Tokens":
        return NVFP4Tokens(self.data[index], self.block_scales[index], self.global_scales[index])

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data, self.block_scales, self.global_scales


def unswizzle_scales(swizzled: torch.Tensor, offsets: torch.Tensor, rows: int, contraction_size: int) -> torch.Tensor:
    """Row-major ``[rows, K // 16]`` block scales from the grouped swizzled layout ``quantize_activations`` writes."""
    scale_columns = _scale_columns(contraction_size)
    within, region_start = _group_layout(offsets, rows, scale_columns)
    columns = torch.arange(contraction_size // _BLOCK, device=swizzled.device)
    index = region_start[:, None] + _swizzled_index(within[:, None], columns[None, :], scale_columns)
    return swizzled.flatten()[index]


def swizzle_scales(block_scales: torch.Tensor, offsets: torch.Tensor, contraction_size: int) -> torch.Tensor:
    """The grouped swizzled layout the NVFP4 GEMM reads, from row-major ``[rows, K // 16]`` block scales whose rows
    are grouped by ``offsets``. Padding rows of each group's last tile stay zero."""
    rows = block_scales.shape[0]
    scale_columns = _scale_columns(contraction_size)
    groups = offsets.numel()
    padded_rows = (rows + groups * (_TILE_ROWS - 1) + _TILE_ROWS - 1) // _TILE_ROWS * _TILE_ROWS
    out = torch.zeros(padded_rows * scale_columns, dtype=torch.uint8, device=block_scales.device)
    within, region_start = _group_layout(offsets, rows, scale_columns)
    columns = torch.arange(contraction_size // _BLOCK, device=block_scales.device)
    index = region_start[:, None] + _swizzled_index(within[:, None], columns[None, :], scale_columns)
    out[index.flatten()] = block_scales.reshape(-1).view(torch.uint8)
    return out.view(block_scales.dtype).view(padded_rows, scale_columns)


def quantize_tokens(x: torch.Tensor, *, four_over_six: bool = False) -> NVFP4Tokens:
    """Quantize ``[M, K]`` tokens before dispatch: the same bytes ``quantize_activations`` gives each row after it."""
    rows, contraction_size = x.shape
    one_group = torch.tensor([rows], device=x.device, dtype=torch.int32)
    quantized = quantize_activations(x, one_group, four_over_six=four_over_six)
    scales = unswizzle_scales(quantized.block_scales, one_group, rows, contraction_size)
    return NVFP4Tokens(quantized.data.view(torch.uint8), scales, quantized.global_scales)


def group_tokens(tokens: NVFP4Tokens, offsets: torch.Tensor) -> _NVFP4Tensor:
    """Tokens already in expert order (rows grouped by ``offsets``) as the GEMM's activation operand."""
    contraction_size = tokens.data.shape[1] * 2
    return _NVFP4Tensor(
        data=tokens.data.view(torch.float4_e2m1fn_x2),
        block_scales=swizzle_scales(tokens.block_scales, offsets, contraction_size),
        global_scales=tokens.global_scales,
        offsets=offsets,
    )
