"""Small Triton helpers around the grouped GEMMs: the row-tile table and the tail fill."""

import torch
import triton
import triton.language as tl


@triton.jit
def _tile_table_kernel(
    counts_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_row_end_ptr,
    num_tiles_ptr,
    expert_row_start_ptr,
    num_experts,
    num_rows,
    max_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    counts = tl.load(counts_ptr + experts, mask=experts < num_experts, other=0).to(tl.int32)
    tiles = tl.cdiv(counts, BLOCK_M)
    tile_end = tl.cumsum(tiles, axis=0)
    row_end = tl.cumsum(counts, axis=0)
    if tl.program_id(0) == 0:
        tl.store(num_tiles_ptr, tl.sum(tiles))
        tl.store(expert_row_start_ptr + experts, row_end - counts, mask=experts < num_experts)

    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    # The expert of tile t is the number of experts whose tiles all come before t.
    expert = tl.sum((tile_end[None, :] <= t[:, None]).to(tl.int32), axis=1)
    owner = experts[None, :] == expert[:, None]
    first_tile = tl.sum(tl.where(owner, tile_end - tiles, 0), axis=1)
    expert_row_start = tl.sum(tl.where(owner, row_end - counts, 0), axis=1)
    # Counts that overrun the activations are cut at their end, so no tile writes past it.
    expert_row_end = tl.minimum(tl.sum(tl.where(owner, row_end, 0), axis=1), num_rows)
    row_start = expert_row_start + (t - first_tile) * BLOCK_M
    valid = t < max_tiles
    tl.store(tile_expert_ptr + t, expert, mask=valid)
    tl.store(tile_row_start_ptr + t, row_start, mask=valid)
    tl.store(tile_row_end_ptr + t, expert_row_end, mask=valid)


class TileTable:
    """Row tiles of ``block_m`` rows, each inside one expert's group, built on device.

    Expert ``e`` owns rows ``[start_e, start_e + num_tokens_per_expert[e])`` with the groups laid
    out back to back from row 0. Tile ``t < num_tiles`` covers rows ``[row_start[t], row_start[t]
    + block_m)`` of expert ``expert[t]``, of which only those before ``row_end[t]`` are its own.
    """

    def __init__(self, num_tokens_per_expert: torch.Tensor, num_rows: int, block_m: int) -> None:
        num_experts = num_tokens_per_expert.numel()
        device = num_tokens_per_expert.device
        self.block_m = block_m
        self.max_tiles = triton.cdiv(num_rows, block_m) + num_experts
        self.expert = torch.empty(self.max_tiles, dtype=torch.int32, device=device)
        self.row_start = torch.empty_like(self.expert)
        self.row_end = torch.empty_like(self.expert)
        self.num_tiles = torch.empty(1, dtype=torch.int32, device=device)
        self.expert_row_start = torch.empty(num_experts, dtype=torch.int32, device=device)
        self.num_tokens_per_expert = num_tokens_per_expert
        block_t = 64
        _tile_table_kernel[(triton.cdiv(self.max_tiles, block_t),)](
            num_tokens_per_expert,
            self.expert,
            self.row_start,
            self.row_end,
            self.num_tiles,
            self.expert_row_start,
            num_experts,
            num_rows,
            self.max_tiles,
            BLOCK_M=block_m,
            BLOCK_E=triton.next_power_of_2(num_experts),
            BLOCK_T=block_t,
        )


@triton.jit
def _zero_tail_kernel(out_ptr, counts_ptr, num_experts, numel, N, BLOCK_E: tl.constexpr, BLOCK: tl.constexpr):
    experts = tl.arange(0, BLOCK_E)
    used = tl.sum(tl.load(counts_ptr + experts, mask=experts < num_experts, other=0)).to(tl.int64)
    for start in range(used * N + tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        tl.store(out_ptr + offsets, tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty), mask=offsets < numel)


def zero_tail(out: torch.Tensor, num_tokens_per_expert: torch.Tensor, num_programs: int) -> None:
    """Zero the rows of ``out`` after the last expert's group, which no GEMM tile owns."""
    num_rows, N = out.shape
    num_experts = num_tokens_per_expert.numel()
    _zero_tail_kernel[(num_programs,)](
        out,
        num_tokens_per_expert,
        num_experts,
        num_rows * N,
        N,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK=2048,
    )
