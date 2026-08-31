# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# A 32K-token packed batch produces at most a 1 GiB score workspace.
SCORE_WORKSPACE_BYTES = 1024**3
TOPK_WORKSPACE_BYTES = 1024**2


@triton.jit
def _selection_scores_kernel(
    query_ptr,
    key_ptr,
    starts_ptr,
    ends_ptr,
    visible_blocks_ptr,
    scores_ptr,
    stride_query_row,
    stride_query_head,
    stride_query_dim,
    stride_key_row,
    stride_key_dim,
    stride_scores_row,
    rows,
    columns,
    key_blocks,
    score_divisor,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dimensions = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    start = tl.load(starts_ptr + row)
    end = tl.load(ends_ptr + row)
    visible = end - start
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)

    tile_start = tl.program_id(1) * TILES_PER_PROGRAM
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROGRAM, tl.cdiv(visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(columns, BLOCK_N))

    query = tl.load(
        query_ptr
        + row * stride_query_row
        + heads[None, :] * stride_query_head
        + dimensions[:, None] * stride_query_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dimensions[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns_in_tile = tile * BLOCK_N + column_offsets
        key_rows = start + columns_in_tile
        live = (columns_in_tile < visible) & (key_rows < key_blocks)
        keys = tl.load(
            key_ptr + key_rows[:, None].to(tl.int64) * stride_key_row + dimensions[None, :] * stride_key_dim,
            mask=live[:, None] & (dimensions[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        head_scores = tl.dot(keys, query, out_dtype=tl.float32)
        head_scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(head_scores, 0.0), 0.0)
        scores = tl.sum(head_scores, axis=1) / score_divisor
        tl.store(
            scores_ptr + row * stride_scores_row + columns_in_tile,
            tl.where(live, scores, -float("inf")),
            mask=columns_in_tile < columns,
        )


def selection_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    columns = key.shape[0]
    scores = torch.empty((query.shape[0], columns), dtype=torch.float32, device=query.device)
    visible_blocks = torch.empty(query.shape[0], dtype=torch.int32, device=query.device)
    if not query.shape[0] or not columns:
        return scores, visible_blocks

    block_n = 64
    block_d = max(16, triton.next_power_of_2(query.shape[2]))
    max_n = max(16, triton.next_power_of_2(query.shape[1]))
    tiles_per_program = 1 if query.shape[0] <= 32 else 8
    _selection_scores_kernel[(query.shape[0], triton.cdiv(columns, block_n * tiles_per_program))](
        query,
        key,
        starts,
        ends,
        visible_blocks,
        scores,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        scores.stride(0),
        query.shape[0],
        columns,
        key.shape[0],
        math.sqrt(query.shape[2]),
        NUM_HEADS=query.shape[1],
        HEAD_DIM=query.shape[2],
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        TILES_PER_PROGRAM=tiles_per_program,
        STAGES=2,
        MAX_N=max_n,
        num_warps=2,
    )
    return scores, visible_blocks


@torch.library.custom_op("prime_kernels::select_indexed_blocks", mutates_args=())
def select_indexed_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    num_blocks = key.shape[0]
    if not num_blocks:
        return torch.zeros(query.shape[0], topk, dtype=torch.int32, device=query.device)

    rows_per_chunk = max(1, SCORE_WORKSPACE_BYTES // (num_blocks * torch.float32.itemsize))
    selected_chunks = []
    workspace = torch.empty(TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device=query.device)
    for query_chunk, start_chunk, end_chunk in zip(
        query.split(rows_per_chunk),
        starts.split(rows_per_chunk),
        ends.split(rows_per_chunk),
        strict=True,
    ):
        scores, visible_blocks = selection_scores(
            query_chunk,
            key,
            start_chunk,
            end_chunk,
        )
        selected = torch.full(
            (query_chunk.shape[0], topk),
            -1,
            dtype=torch.int32,
            device=query.device,
        )
        torch.ops.prime_indexed_attention.persistent_topk(
            scores,
            visible_blocks,
            selected,
            workspace,
            topk,
            num_blocks,
        )
        valid = (selected >= 0) & (selected < visible_blocks[:, None])
        selected.add_(start_chunk[:, None])
        selected.masked_fill_(~valid, num_blocks)
        selected_chunks.append(selected)

    if len(selected_chunks) == 1:
        return selected_chunks[0]
    return torch.cat(selected_chunks)


@select_indexed_blocks.register_fake
def select_indexed_blocks_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    return query.new_empty((query.shape[0], topk), dtype=torch.int32)
