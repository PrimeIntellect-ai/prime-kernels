from __future__ import annotations

import torch

from . import _C

__all__ = ["scatter_tiles", "wait_tiles", "wait_and_reduce", "fused_dispatch_ffn", "fused_grad_combine_ffn"]


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
    torch.ops.prime_fine_grained_compute_comm_overlap.scatter_tiles(
        src, hidden_peer_ptrs, flag_peer_ptrs, tile_peer_rank, tile_local_row_start,
        tile_peer_row_start, tile_valid_rows, tile_flag_index, n_blocks,
    )


def wait_tiles(local_flag: torch.Tensor, tile_valid: torch.Tensor) -> None:
    torch.ops.prime_fine_grained_compute_comm_overlap.wait_tiles(local_flag, tile_valid)


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
    torch.ops.prime_fine_grained_compute_comm_overlap.wait_and_reduce(
        local_hidden, local_flag, routed_scores, weighted_routed_out,
        dispatch_peer_rank, dispatch_local_row_start, dispatch_own_tile_ordinal, dispatch_valid_rows,
        block_m, n_blocks,
    )


def fused_dispatch_ffn(
    src: torch.Tensor,
    hidden_peer_ptrs: torch.Tensor,
    flag_peer_ptrs: torch.Tensor,
    tile_peer_rank: torch.Tensor,
    tile_local_row_start: torch.Tensor,
    tile_peer_row_start: torch.Tensor,
    tile_valid_rows: torch.Tensor,
    tile_flag_index: torch.Tensor,
    recv_hidden: torch.Tensor,
    recv_flag: torch.Tensor,
    recv_tile_valid: torch.Tensor,
    recv_tile_to_local_expert: torch.Tensor,
    gate_proj: torch.Tensor | None,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_out: torch.Tensor,
    act_scratch: torch.Tensor,
    gate_scratch: torch.Tensor | None,
    block_start_clock: torch.Tensor,
    block_end_clock: torch.Tensor,
    block_m: int,
    n_producer_blocks: int,
    n_consumer_blocks: int,
) -> None:
    """CTA-specialized, single-kernel-launch fusion of dispatch (producer CTAs, blockIdx.x <
    n_producer_blocks) and the gated/ungated SiLU expert FFN (consumer CTAs, the rest) -- a
    prototype for real intra-kernel compute/communication overlap, replacing the two-CUDA-stream
    approach (which was measured to have zero actual kernel overlap: the comm wait resolves before
    the compute stream even starts). Producer and consumer CTAs are part of the same grid, so the
    CUDA scheduler can run them concurrently on different SMs without any separate-kernel-launch
    overhead. Not cutlass-competitive (plain shared-memory-tiled bf16 GEMM, fp32 accumulate, no
    tensor cores) and only SiLU (gated or ungated) is supported -- see kernels.cu's docstring.
    """
    torch.ops.prime_fine_grained_compute_comm_overlap.fused_dispatch_ffn(
        src, hidden_peer_ptrs, flag_peer_ptrs, tile_peer_rank, tile_local_row_start,
        tile_peer_row_start, tile_valid_rows, tile_flag_index,
        recv_hidden, recv_flag, recv_tile_valid, recv_tile_to_local_expert,
        gate_proj, up_proj, down_proj, expert_out, act_scratch, gate_scratch,
        block_start_clock, block_end_clock, block_m, n_producer_blocks, n_consumer_blocks,
    )


def fused_grad_combine_ffn(
    src: torch.Tensor,
    hidden_peer_ptrs: torch.Tensor,
    flag_peer_ptrs: torch.Tensor,
    tile_peer_rank: torch.Tensor,
    tile_local_row_start: torch.Tensor,
    tile_peer_row_start: torch.Tensor,
    tile_valid_rows: torch.Tensor,
    tile_flag_index: torch.Tensor,
    grad_expert_out_recv: torch.Tensor,
    recv_flag: torch.Tensor,
    recv_tile_valid: torch.Tensor,
    recv_tile_to_local_expert: torch.Tensor,
    hidden_shadow: torch.Tensor,
    gate_proj: torch.Tensor | None,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    grad_dispatch_hidden_out: torch.Tensor,
    up_scratch: torch.Tensor,
    gate_scratch: torch.Tensor | None,
    grad_act_scratch: torch.Tensor,
    act_scratch: torch.Tensor,
    grad_up_proj: torch.Tensor,
    grad_down_proj: torch.Tensor,
    grad_gate_proj: torch.Tensor | None,
    block_m: int,
    n_producer_blocks: int,
    n_consumer_blocks: int,
) -> None:
    """Backward mirror of `fused_dispatch_ffn`: producer CTAs scatter a gradient (`src`, e.g.
    `grad_combine_hidden`) into a peer's symmetric-memory buffer using the *forward* dispatch
    schedule (role-swapped, same trick `OverlappedMoELayerFunction.backward` already uses for the
    unfused path); consumer CTAs wait per-tile then compute both the FFN's *input* gradient
    (`grad_dispatch_hidden_out`) and its *weight* gradients (`grad_up_proj`/`grad_down_proj`/
    `grad_gate_proj`) via real WMMA GEMMs, recomputing `up`/`gate` from the saved `hidden_shadow`
    (not saved anywhere else). Weight gradients are accumulated with fp32 atomics into
    `grad_up_proj`/`grad_down_proj`/`grad_gate_proj` (one expert's tokens span multiple tiles
    across the grid, hence atomics rather than a plain store) -- these three buffers must be
    zeroed by the caller before each call, and must be fp32 (cast to the params' own dtype
    afterwards). Only gated/ungated SiLU is supported. See `kernels.cu`'s
    `fused_grad_combine_ffn_kernel` docstring for the math.
    """
    torch.ops.prime_fine_grained_compute_comm_overlap.fused_grad_combine_ffn(
        src, hidden_peer_ptrs, flag_peer_ptrs, tile_peer_rank, tile_local_row_start,
        tile_peer_row_start, tile_valid_rows, tile_flag_index,
        grad_expert_out_recv, recv_flag, recv_tile_valid, recv_tile_to_local_expert,
        hidden_shadow, gate_proj, up_proj, down_proj,
        grad_dispatch_hidden_out, up_scratch, gate_scratch, grad_act_scratch,
        act_scratch, grad_up_proj, grad_down_proj, grad_gate_proj,
        block_m, n_producer_blocks, n_consumer_blocks,
    )
