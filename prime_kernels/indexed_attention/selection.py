import tilelang
import torch
from tilelang import language as T

SCORE_WORKSPACE_BYTES = 1024**3
RADIX_BITS = 8
RADIX_SIZE = 1 << RADIX_BITS


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def selection_score_kernel(
    num_query_heads: int,
    head_dim: int,
    block_keys: int = 64,
    key_tiles_per_program: int = 8,
    threads: int = 64,
):
    query_tokens = T.dynamic("query_tokens")
    key_blocks = T.dynamic("key_blocks")
    head_tile = max(tilelang.math.next_power_of_2(num_query_heads), 16)

    query_shape = [query_tokens, num_query_heads, head_dim]
    key_shape = [key_blocks, head_dim]
    bounds_shape = [query_tokens]
    scores_shape = [query_tokens, key_blocks]

    @T.prim_func
    def kernel(
        query: T.Tensor(query_shape, T.bfloat16),
        key: T.Tensor(key_shape, T.bfloat16),
        block_starts: T.Tensor(bounds_shape, T.int32),
        block_ends: T.Tensor(bounds_shape, T.int32),
        scores: T.Tensor(scores_shape, T.float32),
        visible_block_counts: T.Tensor(bounds_shape, T.int32),
    ):
        with T.Kernel(
            T.ceildiv(key_blocks, block_keys * key_tiles_per_program),
            query_tokens,
            threads=threads,
        ) as (key_group, query_token):
            query_shared = T.alloc_shared([head_tile, head_dim], T.bfloat16)
            key_shared = T.alloc_shared([block_keys, head_dim], T.bfloat16)
            head_scores = T.alloc_fragment([block_keys, head_tile], T.float32)
            combined_scores = T.alloc_fragment([block_keys], T.float32)

            start = T.alloc_var(T.int32)
            visible_count = T.alloc_var(T.int32)
            start = block_starts[query_token]
            visible_count = block_ends[query_token] - start

            if key_group == 0:
                visible_block_counts[query_token] = visible_count

            for head, dim in T.Parallel(head_tile, head_dim):
                query_shared[head, dim] = T.if_then_else(
                    head < num_query_heads,
                    query[query_token, head, dim],
                    0,
                )

            for key_tile in T.serial(key_tiles_per_program):
                first_local_key = (key_group * key_tiles_per_program + key_tile) * block_keys
                if first_local_key < visible_count:
                    for local_key, dim in T.Parallel(block_keys, head_dim):
                        key_index = start + first_local_key + local_key
                        key_shared[local_key, dim] = T.if_then_else(
                            first_local_key + local_key < visible_count and key_index < key_blocks,
                            key[key_index, dim],
                            0,
                        )

                    T.gemm(
                        key_shared,
                        query_shared,
                        head_scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for local_key, head in T.Parallel(block_keys, head_tile):
                        head_scores[local_key, head] = T.if_then_else(
                            head < num_query_heads,
                            T.max(head_scores[local_key, head], 0),
                            0,
                        )
                    T.reduce_sum(head_scores, combined_scores, dim=1)
                    for local_key in T.Parallel(block_keys):
                        if first_local_key + local_key < visible_count:
                            scores[query_token, first_local_key + local_key] = combined_scores[local_key] * (
                                head_dim**-0.5
                            )

    return kernel


@tilelang.jit(out_idx=[-1])
def radix_select_kernel(topk: int, threads: int = 256):
    rows = T.dynamic("rows")
    columns = T.dynamic("columns")

    @T.prim_func
    def kernel(
        scores: T.Tensor([rows, columns], T.float32),
        visible_block_counts: T.Tensor([rows], T.int32),
        block_starts: T.Tensor([rows], T.int32),
        selected_blocks: T.Tensor([rows, topk], T.int32),
    ):
        with T.Kernel(rows, threads=threads) as row:
            thread = T.get_thread_binding()
            histogram = T.alloc_shared([RADIX_SIZE], T.int32)
            threshold_prefix = T.alloc_shared([1], T.uint32)
            threshold_prefix_mask = T.alloc_shared([1], T.uint32)
            remaining_count = T.alloc_shared([1], T.int32)
            threshold_digit = T.alloc_shared([1], T.int32)
            threshold_count = T.alloc_shared([1], T.int32)
            greater_digit_count = T.alloc_shared([1], T.int32)
            output_count = T.alloc_shared([1], T.int32)

            score_bits = T.alloc_var(T.uint32)
            digit = T.alloc_var(T.int32)
            running_count = T.alloc_var(T.int32)
            bin_count = T.alloc_var(T.int32)
            output_position = T.alloc_var(T.int32)
            visible_count = T.alloc_var(T.int32)
            column = T.alloc_var(T.int32)

            visible_count = T.min(visible_block_counts[row], columns)
            for output_position in T.Parallel(topk):
                selected_blocks[row, output_position] = columns
            if thread == 0:
                threshold_prefix[0] = 0
                threshold_prefix_mask[0] = 0
                remaining_count[0] = T.min(topk, visible_count)
                threshold_count[0] = 0
            T.sync_threads()

            for radix_pass in T.serial(32 // RADIX_BITS):
                if remaining_count[0] > 0:
                    T.fill(histogram, 0)
                    T.sync_threads()
                    for column_group in T.serial(T.ceildiv(columns, threads)):
                        column = column_group * threads + thread
                        if column < visible_count:
                            score_bits = T.reinterpret(scores[row, column], T.uint32)
                            if (score_bits & threshold_prefix_mask[0]) == threshold_prefix[0]:
                                digit = T.cast(
                                    (score_bits >> (32 - RADIX_BITS * (radix_pass + 1))) & (RADIX_SIZE - 1),
                                    T.int32,
                                )
                                T.atomic_add(histogram[digit], 1)
                    T.sync_threads()

                    if thread == 0:
                        running_count = 0
                        threshold_digit[0] = 0
                        threshold_count[0] = 0
                        greater_digit_count[0] = 0
                        for digit_offset in T.serial(RADIX_SIZE):
                            digit = RADIX_SIZE - 1 - digit_offset
                            bin_count = histogram[digit]
                            if running_count < remaining_count[0] and running_count + bin_count >= remaining_count[0]:
                                threshold_digit[0] = digit
                                threshold_count[0] = bin_count
                                greater_digit_count[0] = running_count
                            running_count += bin_count
                        remaining_count[0] -= greater_digit_count[0]
                        threshold_prefix[0] |= T.cast(threshold_digit[0], T.uint32) << (
                            32 - RADIX_BITS * (radix_pass + 1)
                        )
                        threshold_prefix_mask[0] |= T.cast(RADIX_SIZE - 1, T.uint32) << (
                            32 - RADIX_BITS * (radix_pass + 1)
                        )
                    T.sync_threads()

            if thread == 0:
                output_count[0] = 0
            T.sync_threads()
            for column_group in T.serial(T.ceildiv(columns, threads)):
                column = column_group * threads + thread
                if column < visible_count:
                    score_bits = T.reinterpret(scores[row, column], T.uint32)
                    if score_bits > threshold_prefix[0]:
                        output_position = T.atomic_add(output_count[0], 1, return_prev=True)
                        selected_blocks[row, output_position] = block_starts[row] + column
            T.sync_threads()
            if threshold_count[0] == remaining_count[0]:
                for column_group in T.serial(T.ceildiv(columns, threads)):
                    column = column_group * threads + thread
                    if column < visible_count:
                        score_bits = T.reinterpret(scores[row, column], T.uint32)
                        if score_bits == threshold_prefix[0]:
                            output_position = T.atomic_add(output_count[0], 1, return_prev=True)
                            if output_position < topk:
                                selected_blocks[row, output_position] = block_starts[row] + column
            elif thread == 0:
                output_position = output_count[0]
                for tied_column in T.serial(columns):
                    if tied_column < visible_count and output_position < topk:
                        score_bits = T.reinterpret(scores[row, tied_column], T.uint32)
                        if score_bits == threshold_prefix[0]:
                            selected_blocks[row, output_position] = block_starts[row] + tied_column
                            output_position += 1

    return kernel


def compute_selection_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    block_starts: torch.Tensor,
    block_ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not query.shape[0] or not key.shape[0]:
        return (
            torch.empty((query.shape[0], key.shape[0]), dtype=torch.float32, device=query.device),
            torch.empty(query.shape[0], dtype=torch.int32, device=query.device),
        )
    return selection_score_kernel(query.shape[1], query.shape[2])(
        query.contiguous(),
        key.contiguous(),
        block_starts.contiguous(),
        block_ends.contiguous(),
    )


@torch.library.custom_op("prime_kernels::select_indexed_blocks", mutates_args=())
def select_indexed_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    block_starts: torch.Tensor,
    block_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    num_blocks = key.shape[0]
    if not query.shape[0] or not topk:
        return torch.empty((query.shape[0], topk), dtype=torch.int32, device=query.device)
    if not num_blocks:
        return torch.zeros((query.shape[0], topk), dtype=torch.int32, device=query.device)

    rows_per_chunk = max(1, SCORE_WORKSPACE_BYTES // (num_blocks * torch.float32.itemsize))
    selected_chunks = []
    contiguous_key = key.contiguous()
    for query_chunk, start_chunk, end_chunk in zip(
        query.split(rows_per_chunk),
        block_starts.split(rows_per_chunk),
        block_ends.split(rows_per_chunk),
        strict=True,
    ):
        scores, visible_block_counts = selection_score_kernel(query.shape[1], query.shape[2])(
            query_chunk.contiguous(),
            contiguous_key,
            start_chunk.contiguous(),
            end_chunk.contiguous(),
        )
        selected_chunks.append(
            radix_select_kernel(topk)(
                scores,
                visible_block_counts,
                start_chunk.contiguous(),
            )
        )

    if len(selected_chunks) == 1:
        return selected_chunks[0]
    return torch.cat(selected_chunks)


@select_indexed_blocks.register_fake
def select_indexed_blocks_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    block_starts: torch.Tensor,
    block_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    return query.new_empty((query.shape[0], topk), dtype=torch.int32)
