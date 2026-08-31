import tilelang
import torch
import torch.nn.functional as F
from tilelang import language as T

# A 32K-token sequence produces a 1 GiB score matrix and runs in one pass.
SCORE_WORKSPACE_BYTES = 1024**3


@tilelang.jit(out_idx=[-1])
def indexed_selection_scores_kernel(
    num_query_heads: int,
    head_dim: int,
    block_queries: int = 64,
    block_keys: int = 128,
    threads: int = 256,
):
    query_tokens = T.dynamic("query_tokens")
    key_blocks = T.dynamic("key_blocks")

    query_shape = [query_tokens * num_query_heads, head_dim]
    key_shape = [key_blocks, head_dim]
    bounds_shape = [query_tokens]
    scores_shape = [query_tokens, key_blocks]

    @T.prim_func
    def kernel(
        query: T.Tensor(query_shape, T.bfloat16),
        key: T.Tensor(key_shape, T.bfloat16),
        starts: T.Tensor(bounds_shape, T.int32),
        ends: T.Tensor(bounds_shape, T.int32),
        scores: T.Tensor(scores_shape, T.float32),
    ):
        with T.Kernel(
            T.ceildiv(key_blocks, block_keys),
            T.ceildiv(query_tokens, block_queries),
            threads=threads,
        ) as (key_block, query_block):
            query_shared = T.alloc_shared([block_queries, head_dim], T.float32)
            key_shared = T.alloc_shared([block_keys, head_dim], T.float32)
            head_scores = T.alloc_fragment([block_queries, block_keys], T.float32)
            combined_scores = T.alloc_fragment([block_queries, block_keys], T.float32)

            for row, dim in T.Parallel(block_keys, head_dim):
                key_index = key_block * block_keys + row
                key_shared[row, dim] = T.if_then_else(
                    key_index < key_blocks,
                    T.cast(key[key_index, dim], T.float32),
                    0,
                )
            T.clear(combined_scores)
            for head in T.serial(num_query_heads):
                for row, dim in T.Parallel(block_queries, head_dim):
                    query_index = query_block * block_queries + row
                    query_shared[row, dim] = T.if_then_else(
                        query_index < query_tokens,
                        T.cast(query[query_index * num_query_heads + head, dim], T.float32),
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
                for row, column in T.Parallel(block_queries, block_keys):
                    combined_scores[row, column] += T.max(head_scores[row, column], 0)

            for row, column in T.Parallel(block_queries, block_keys):
                query_index = query_block * block_queries + row
                key_index = key_block * block_keys + column
                if query_index < query_tokens and key_index < key_blocks:
                    if key_index < starts[query_index] or key_index >= ends[query_index]:
                        combined_scores[row, column] = -T.infinity(T.float32)

            T.copy(
                combined_scores,
                scores[query_block * block_queries, key_block * block_keys],
            )

    return kernel


def _select_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    num_blocks = key.shape[0]
    if num_blocks == 0:
        return torch.zeros(query.shape[0], topk, dtype=torch.int32, device=query.device)

    chunk_size = max(1, SCORE_WORKSPACE_BYTES // (num_blocks * torch.float32.itemsize))
    selected_chunks = []
    selected_count = min(topk, num_blocks)
    key = key.contiguous()
    for query_chunk, start_chunk, end_chunk in zip(
        query.split(chunk_size),
        starts.split(chunk_size),
        ends.split(chunk_size),
        strict=True,
    ):
        scores = indexed_selection_scores_kernel(
            query.shape[1],
            query.shape[2],
        )(
            query_chunk.flatten(0, 1).contiguous(),
            key,
            start_chunk.contiguous(),
            end_chunk.contiguous(),
        )
        selected = scores.topk(selected_count, dim=-1).indices
        if selected_count < topk:
            selected = F.pad(selected, (0, topk - selected_count), value=num_blocks)
        selected.masked_fill_(
            (selected < start_chunk[:, None]) | (selected >= end_chunk[:, None]),
            num_blocks,
        )
        selected_chunks.append(selected.to(torch.int32))
    if len(selected_chunks) == 1:
        return selected_chunks[0]
    return torch.cat(selected_chunks)


@torch.library.custom_op("prime_kernels::select_indexed_blocks", mutates_args=())
def select_indexed_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    return _select_blocks(query, key, starts, ends, topk)


@select_indexed_blocks.register_fake
def select_indexed_blocks_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    return query.new_empty((query.shape[0], topk), dtype=torch.int32)
