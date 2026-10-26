import tilelang
import torch
from tilelang import language as T

SCORE_WORKSPACE_BYTES = 1024**3
RADIX_BITS = 8
RADIX_SIZE = 1 << RADIX_BITS
SCORE_VECTOR = 4


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def selection_score_kernel(
    num_query_heads: int,
    head_dim: int,
    query_tile: int = 16,
    block_keys: int = 64,
    key_tiles_per_program: int = 8,
    threads: int = 128,
):
    query_tokens = T.dynamic("query_tokens")
    key_blocks = T.dynamic("key_blocks")
    score_columns = T.dynamic("score_columns")
    rows = query_tile * num_query_heads

    query_shape = [query_tokens, num_query_heads, head_dim]
    key_shape = [key_blocks, head_dim]
    bounds_shape = [query_tokens]
    scores_shape = [query_tokens, score_columns]

    def combine_heads(head_scores_shared, local_query, local_key):
        values = [
            T.max(head_scores_shared[local_query * num_query_heads + head, local_key], 0)
            for head in range(num_query_heads)
        ]
        while len(values) > 1:
            values = [values[i] + values[i + 1] for i in range(0, len(values), 2)]
        return values[0]

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
            T.ceildiv(query_tokens, query_tile),
            threads=threads,
        ) as (key_group, query_group):
            query_shared = T.alloc_shared([rows, head_dim], T.bfloat16)
            key_shared = T.alloc_shared([block_keys, head_dim], T.bfloat16)
            head_scores = T.alloc_fragment([rows, block_keys], T.float32)
            head_scores_shared = T.alloc_shared([rows, block_keys], T.float32)
            query_starts = T.alloc_shared([query_tile], T.int32)
            query_counts = T.alloc_shared([query_tile], T.int32)

            tile_start = T.alloc_var(T.int32)
            tile_end = T.alloc_var(T.int32)
            first_query = query_group * query_tile

            for local_query in T.Parallel(query_tile):
                query_starts[local_query] = T.if_then_else(
                    first_query + local_query < query_tokens, block_starts[first_query + local_query], key_blocks
                )
                query_counts[local_query] = T.if_then_else(
                    first_query + local_query < query_tokens,
                    block_ends[first_query + local_query] - block_starts[first_query + local_query],
                    0,
                )
            for row, dim in T.Parallel(rows, head_dim):
                query_shared[row, dim] = T.if_then_else(
                    first_query + row // num_query_heads < query_tokens,
                    query[first_query + row // num_query_heads, row % num_query_heads, dim],
                    0,
                )
            T.sync_threads()

            tile_start = key_blocks
            tile_end = 0
            for local_query in T.serial(query_tile):
                if query_counts[local_query] > 0:
                    tile_start = T.min(tile_start, query_starts[local_query])
                    tile_end = T.max(tile_end, query_starts[local_query] + query_counts[local_query])
            if key_group == 0:
                for local_query in T.Parallel(query_tile):
                    if first_query + local_query < query_tokens:
                        visible_block_counts[first_query + local_query] = query_counts[local_query]

            for key_tile in T.serial(key_tiles_per_program):
                first_key = tile_start + (key_group * key_tiles_per_program + key_tile) * block_keys
                if first_key < tile_end:
                    for local_key, dim in T.Parallel(block_keys, head_dim):
                        key_shared[local_key, dim] = T.if_then_else(
                            first_key + local_key < tile_end and first_key + local_key < key_blocks,
                            key[first_key + local_key, dim],
                            0,
                        )
                    T.gemm(
                        query_shared,
                        key_shared,
                        head_scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(head_scores, head_scores_shared)
                    T.sync_threads()
                    for local_query, local_key in T.Parallel(query_tile, block_keys):
                        if (
                            first_key + local_key >= query_starts[local_query]
                            and first_key + local_key - query_starts[local_query] < query_counts[local_query]
                        ):
                            scores[first_query + local_query, first_key + local_key - query_starts[local_query]] = (
                                combine_heads(head_scores_shared, local_query, local_key) * (head_dim**-0.5)
                            )
                    T.sync_threads()

    return kernel


@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True})
def radix_select_kernel(topk: int, threads: int = 256):
    rows = T.dynamic("rows")
    column_vectors = T.dynamic("column_vectors")

    @T.prim_func
    def kernel(
        scores: T.Tensor([rows, column_vectors, SCORE_VECTOR], T.int32),
        visible_block_counts: T.Tensor([rows], T.int32),
        block_starts: T.Tensor([rows], T.int32),
        num_blocks: T.int32,
        selected_blocks: T.Tensor([rows, topk], T.int32),
    ):
        with T.Kernel(rows, threads=threads) as row:
            thread = T.get_thread_binding()
            histogram = T.alloc_shared([RADIX_SIZE], T.int32)
            suffix_counts = T.alloc_shared([RADIX_SIZE], T.int32)
            thread_greater = T.alloc_shared([threads], T.int32)
            thread_ties = T.alloc_shared([threads], T.int32)
            greater_scan = T.alloc_shared([threads], T.int32)
            tie_scan = T.alloc_shared([threads], T.int32)
            threshold_prefix = T.alloc_shared([1], T.uint32)
            threshold_prefix_mask = T.alloc_shared([1], T.uint32)
            remaining_count = T.alloc_shared([1], T.int32)
            threshold_digit = T.alloc_shared([1], T.int32)
            greater_digit_count = T.alloc_shared([1], T.int32)
            output_count = T.alloc_shared([1], T.int32)
            tie_count = T.alloc_shared([1], T.int32)
            loaded_words = T.alloc_local([SCORE_VECTOR], T.int32)
            loaded_bits = T.alloc_local([SCORE_VECTOR], T.uint32)

            visible_count = T.alloc_var(T.int32)
            prefix_bits = T.alloc_var(T.uint32)
            prefix_mask = T.alloc_var(T.uint32)
            vector = T.alloc_var(T.int32)
            greater_before = T.alloc_var(T.int32)
            ties_before = T.alloc_var(T.int32)
            ties_left = T.alloc_var(T.int32)

            visible_count = T.min(visible_block_counts[row], num_blocks)
            for output_position in T.Parallel(topk):
                selected_blocks[row, output_position] = num_blocks
            if thread == 0:
                threshold_prefix[0] = 0
                threshold_prefix_mask[0] = 0
                remaining_count[0] = T.min(topk, visible_count)
            T.sync_threads()

            for radix_pass in T.serial(32 // RADIX_BITS):
                if remaining_count[0] > 0:
                    T.fill(histogram, 0)
                    prefix_bits = threshold_prefix[0]
                    prefix_mask = threshold_prefix_mask[0]
                    T.sync_threads()
                    for vector_group in T.serial(T.ceildiv(visible_count, threads * SCORE_VECTOR)):
                        vector = vector_group * threads + thread
                        if vector * SCORE_VECTOR < visible_count:
                            for element in T.vectorized(SCORE_VECTOR):
                                loaded_words[element] = scores[row, vector, element]
                            for element in T.unroll(SCORE_VECTOR):
                                loaded_bits[element] = T.cast(loaded_words[element], T.uint32)
                                if (
                                    vector * SCORE_VECTOR + element < visible_count
                                    and (loaded_bits[element] & prefix_mask) == prefix_bits
                                ):
                                    T.atomic_add(
                                        histogram[
                                            T.cast(
                                                (loaded_bits[element] >> (32 - RADIX_BITS * (radix_pass + 1)))
                                                & (RADIX_SIZE - 1),
                                                T.int32,
                                            )
                                        ],
                                        1,
                                    )
                    T.sync_threads()
                    T.cumsum(histogram, suffix_counts, dim=0, reverse=True)
                    T.sync_threads()
                    for bin_digit in T.Parallel(RADIX_SIZE):
                        if (
                            suffix_counts[bin_digit] >= remaining_count[0]
                            and suffix_counts[bin_digit] - histogram[bin_digit] < remaining_count[0]
                        ):
                            threshold_digit[0] = bin_digit
                            greater_digit_count[0] = suffix_counts[bin_digit] - histogram[bin_digit]
                    T.sync_threads()
                    if thread == 0:
                        remaining_count[0] -= greater_digit_count[0]
                        threshold_prefix[0] |= T.cast(threshold_digit[0], T.uint32) << (
                            32 - RADIX_BITS * (radix_pass + 1)
                        )
                        threshold_prefix_mask[0] |= T.cast(RADIX_SIZE - 1, T.uint32) << (
                            32 - RADIX_BITS * (radix_pass + 1)
                        )
                    T.sync_threads()

            prefix_bits = threshold_prefix[0]
            if thread == 0:
                output_count[0] = 0
                tie_count[0] = 0
            for vector_group in T.serial(T.ceildiv(visible_count, threads * SCORE_VECTOR)):
                vector = vector_group * threads + thread
                if vector * SCORE_VECTOR < visible_count:
                    for element in T.vectorized(SCORE_VECTOR):
                        loaded_words[element] = scores[row, vector, element]
                greater_before = 0
                ties_before = 0
                for element in T.unroll(SCORE_VECTOR):
                    loaded_bits[element] = T.cast(loaded_words[element], T.uint32)
                    if vector * SCORE_VECTOR + element < visible_count:
                        greater_before += T.cast(loaded_bits[element] > prefix_bits, T.int32)
                        ties_before += T.cast(loaded_bits[element] == prefix_bits, T.int32)
                thread_greater[thread] = greater_before
                thread_ties[thread] = ties_before
                T.sync_threads()
                T.cumsum(thread_greater, greater_scan, dim=0)
                T.cumsum(thread_ties, tie_scan, dim=0)
                T.sync_threads()
                greater_before = greater_scan[thread] - thread_greater[thread]
                ties_before = tie_scan[thread] - thread_ties[thread]
                ties_left = T.max(remaining_count[0] - tie_count[0], 0)
                for element in T.unroll(SCORE_VECTOR):
                    if vector * SCORE_VECTOR + element < visible_count:
                        if loaded_bits[element] > prefix_bits or (
                            loaded_bits[element] == prefix_bits and ties_before < ties_left
                        ):
                            selected_blocks[row, output_count[0] + greater_before + T.min(ties_before, ties_left)] = (
                                block_starts[row] + vector * SCORE_VECTOR + element
                            )
                        greater_before += T.cast(loaded_bits[element] > prefix_bits, T.int32)
                        ties_before += T.cast(loaded_bits[element] == prefix_bits, T.int32)
                T.sync_threads()
                if thread == 0:
                    output_count[0] += greater_scan[threads - 1] + T.min(tie_scan[threads - 1], ties_left)
                    tie_count[0] += tie_scan[threads - 1]
                T.sync_threads()

    return kernel


def score_workspace(query_tokens: int, num_blocks: int, device: torch.device) -> torch.Tensor:
    return torch.empty(
        (query_tokens, -(-num_blocks // SCORE_VECTOR) * SCORE_VECTOR), dtype=torch.float32, device=device
    )


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
    scores = score_workspace(query.shape[0], key.shape[0], query.device)
    visible_block_counts = selection_score_kernel(query.shape[1], query.shape[2])(
        query.contiguous(),
        key.contiguous(),
        block_starts.contiguous(),
        block_ends.contiguous(),
        scores,
    )
    return scores[:, : key.shape[0]], visible_block_counts


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
    scores = score_workspace(min(rows_per_chunk, query.shape[0]), num_blocks, query.device)
    selected_chunks = []
    contiguous_key = key.contiguous()
    for query_chunk, start_chunk, end_chunk in zip(
        query.split(rows_per_chunk),
        block_starts.split(rows_per_chunk),
        block_ends.split(rows_per_chunk),
        strict=True,
    ):
        chunk_scores = scores[: query_chunk.shape[0]]
        visible_block_counts = selection_score_kernel(query.shape[1], query.shape[2])(
            query_chunk.contiguous(),
            contiguous_key,
            start_chunk.contiguous(),
            end_chunk.contiguous(),
            chunk_scores,
        )
        selected_chunks.append(
            radix_select_kernel(topk)(
                chunk_scores.view(torch.int32).view(query_chunk.shape[0], -1, SCORE_VECTOR),
                visible_block_counts,
                start_chunk.contiguous(),
                num_blocks,
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
