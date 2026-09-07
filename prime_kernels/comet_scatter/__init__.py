from __future__ import annotations

import torch

from . import _C

__all__ = ["scatter_tiles", "wait_tiles", "wait_and_reduce"]


def scatter_tiles(
    src: torch.Tensor,
    hidden_peer_ptrs: torch.Tensor,
    flag_peer_ptrs: torch.Tensor,
    tile_peer_rank: torch.Tensor,
    tile_local_row_start: torch.Tensor,
    tile_peer_row_start: torch.Tensor,
    tile_valid_rows: torch.Tensor,
    tile_flag_index: torch.Tensor,
    n_blocks: int = 132,
) -> None:
    torch.ops.prime_comet_scatter.scatter_tiles(
        src, hidden_peer_ptrs, flag_peer_ptrs, tile_peer_rank, tile_local_row_start,
        tile_peer_row_start, tile_valid_rows, tile_flag_index, n_blocks,
    )


def wait_tiles(local_flag: torch.Tensor, tile_valid: torch.Tensor) -> None:
    torch.ops.prime_comet_scatter.wait_tiles(local_flag, tile_valid)


def wait_and_reduce(
    local_hidden: torch.Tensor,
    local_flag: torch.Tensor,
    routed_scores: torch.Tensor,
    weighted_routed_out: torch.Tensor,
    dispatch_peer_rank: torch.Tensor,
    dispatch_local_row_start: torch.Tensor,
    dispatch_own_tile_ordinal: torch.Tensor,
    dispatch_valid_rows: torch.Tensor,
    block_m: int,
    n_blocks: int = 132,
) -> None:
    torch.ops.prime_comet_scatter.wait_and_reduce(
        local_hidden, local_flag, routed_scores, weighted_routed_out,
        dispatch_peer_rank, dispatch_local_row_start, dispatch_own_tile_ordinal, dispatch_valid_rows,
        block_m, n_blocks,
    )
