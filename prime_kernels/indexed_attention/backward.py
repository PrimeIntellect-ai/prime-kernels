# Vendored from tile-ai/tilelang (Apache 2.0), modified for indexed GQA.

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(out_idx=[-1])
def attention_delta(
    num_heads: int,
    head_dim: int,
    block_tokens: int = 32,
    num_stages: int = 5,
):
    batch_size = T.dynamic("batch_size")
    query_length = T.dynamic("query_length")
    shape = [batch_size, query_length, num_heads, head_dim]

    @T.prim_func
    def kernel(
        output: T.Tensor(shape, T.bfloat16),
        grad_output: T.Tensor(shape, T.bfloat16),
        delta: T.Tensor([batch_size, query_length, num_heads], T.float32),
    ):
        with T.Kernel(num_heads, T.ceildiv(query_length, block_tokens), batch_size) as (head, token_block, batch):
            output_fragment = T.alloc_fragment([block_tokens, block_tokens], T.float32)
            grad_output_fragment = T.alloc_fragment([block_tokens, block_tokens], T.float32)
            products = T.alloc_fragment([block_tokens, block_tokens], T.float32)
            row_sum = T.alloc_fragment([block_tokens], T.float32)

            T.clear(products)
            for dim_block in T.Pipelined(T.ceildiv(head_dim, block_tokens), num_stages=num_stages):
                T.copy(
                    output[
                        batch,
                        token_block * block_tokens : (token_block + 1) * block_tokens,
                        head,
                        dim_block * block_tokens : (dim_block + 1) * block_tokens,
                    ],
                    output_fragment,
                )
                T.copy(
                    grad_output[
                        batch,
                        token_block * block_tokens : (token_block + 1) * block_tokens,
                        head,
                        dim_block * block_tokens : (dim_block + 1) * block_tokens,
                    ],
                    grad_output_fragment,
                )
                for token, dim in T.Parallel(block_tokens, block_tokens):
                    products[token, dim] += output_fragment[token, dim] * grad_output_fragment[token, dim]

            T.reduce_sum(products, row_sum, dim=1)
            T.copy(
                row_sum,
                delta[
                    batch,
                    token_block * block_tokens : (token_block + 1) * block_tokens,
                    head,
                ],
            )

    return kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # TileLang can otherwise merge buffers used on opposite sides of an atomic update.
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
    },
)
def indexed_attention_backward_kernel(
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    selection_width: int,
    scale: float,
    block_tokens: int = 32,
    threads: int = 128,
):
    batch_size = T.dynamic("batch_size")
    query_length = T.dynamic("query_length")
    kv_length = T.dynamic("kv_length")

    query_heads_per_kv_head = num_query_heads // num_kv_heads
    head_tile = max(tilelang.math.next_power_of_2(query_heads_per_kv_head), 16)
    num_token_blocks = tilelang.cdiv(selection_width, block_tokens)
    log2_scale = scale * 1.44269504

    query_shape = [batch_size, query_length, num_query_heads, head_dim]
    kv_shape = [batch_size, kv_length, num_kv_heads, head_dim]
    indices_shape = [batch_size, query_length, selection_width]
    statistics_shape = [batch_size, query_length, num_query_heads]

    @T.prim_func
    def kernel(
        query: T.Tensor(query_shape, T.bfloat16),
        key: T.Tensor(kv_shape, T.bfloat16),
        value: T.Tensor(kv_shape, T.bfloat16),
        grad_output: T.Tensor(query_shape, T.bfloat16),
        indices: T.Tensor(indices_shape, T.int32),
        logsumexp: T.Tensor(statistics_shape, T.float32),
        delta: T.Tensor(statistics_shape, T.float32),
        grad_query: T.Tensor(query_shape, T.bfloat16),
        grad_key: T.Tensor(kv_shape, T.float32),
        grad_value: T.Tensor(kv_shape, T.float32),
    ):
        with T.Kernel(query_length, batch_size, num_kv_heads, threads=threads) as (query_token, batch, kv_head):
            query_shared = T.alloc_shared([head_tile, head_dim], T.bfloat16)
            key_shared = T.alloc_shared([block_tokens, head_dim], T.bfloat16)
            value_shared = T.alloc_shared([block_tokens, head_dim], T.bfloat16)
            grad_output_shared = T.alloc_shared([head_tile, head_dim], T.bfloat16)
            probability_shared = T.alloc_shared([head_tile, block_tokens], T.bfloat16)
            grad_score_shared = T.alloc_shared([head_tile, block_tokens], T.bfloat16)
            grad_query_shared = T.alloc_shared([head_tile, head_dim], T.bfloat16)
            logsumexp_shared = T.alloc_shared([head_tile], T.float32)
            delta_shared = T.alloc_shared([head_tile], T.float32)
            atomic_store = T.alloc_shared([block_tokens, head_dim], T.float32)

            valid_index = T.alloc_fragment([block_tokens], "bool")
            probability = T.alloc_fragment([head_tile, block_tokens], T.float32)
            grad_probability = T.alloc_fragment([head_tile, block_tokens], T.float32)
            grad_query_accumulator = T.alloc_fragment([head_tile, head_dim], T.float32)
            grad_kv_accumulator = T.alloc_fragment([block_tokens, head_dim], T.float32)

            first_query_head = kv_head * query_heads_per_kv_head
            last_valid_kv = kv_length - 2

            T.copy(
                query[
                    batch,
                    query_token,
                    first_query_head : first_query_head + query_heads_per_kv_head,
                    :,
                ],
                query_shared[:query_heads_per_kv_head, :],
            )
            T.copy(
                grad_output[
                    batch,
                    query_token,
                    first_query_head : first_query_head + query_heads_per_kv_head,
                    :,
                ],
                grad_output_shared[:query_heads_per_kv_head, :],
            )
            T.copy(
                logsumexp[
                    batch,
                    query_token,
                    first_query_head : first_query_head + query_heads_per_kv_head,
                ],
                logsumexp_shared[:query_heads_per_kv_head],
            )
            T.copy(
                delta[
                    batch,
                    query_token,
                    first_query_head : first_query_head + query_heads_per_kv_head,
                ],
                delta_shared[:query_heads_per_kv_head],
            )
            for head, dim in T.Parallel(head_tile - query_heads_per_kv_head, head_dim):
                query_shared[query_heads_per_kv_head + head, dim] = 0
                grad_output_shared[query_heads_per_kv_head + head, dim] = 0
            for head in T.Parallel(head_tile - query_heads_per_kv_head):
                logsumexp_shared[query_heads_per_kv_head + head] = 0
                delta_shared[query_heads_per_kv_head + head] = 0

            T.clear(grad_query_accumulator)
            for selection_block in T.Pipelined(num_token_blocks, num_stages=0):
                for selected_token in T.Parallel(block_tokens):
                    valid_index[selected_token] = (
                        indices[batch, query_token, selection_block * block_tokens + selected_token] <= last_valid_kv
                    )

                for head, selected_token in T.Parallel(head_tile, block_tokens):
                    probability[head, selected_token] = T.if_then_else(
                        valid_index[selected_token],
                        0,
                        -T.infinity(T.float32),
                    )
                for selected_token, dim in T.Parallel(block_tokens, head_dim):
                    key_shared[selected_token, dim] = key[
                        batch,
                        indices[batch, query_token, selection_block * block_tokens + selected_token],
                        kv_head,
                        dim,
                    ]
                T.gemm(
                    query_shared,
                    key_shared,
                    probability,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                for head, selected_token in T.Parallel(head_tile, block_tokens):
                    if head < query_heads_per_kv_head:
                        probability[head, selected_token] = T.exp2(
                            probability[head, selected_token] * log2_scale - logsumexp_shared[head]
                        )
                    else:
                        probability[head, selected_token] = 0
                T.copy(probability, probability_shared)

                for selected_token, dim in T.Parallel(block_tokens, head_dim):
                    value_shared[selected_token, dim] = value[
                        batch,
                        indices[batch, query_token, selection_block * block_tokens + selected_token],
                        kv_head,
                        dim,
                    ]
                T.gemm(
                    grad_output_shared,
                    value_shared,
                    grad_probability,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )

                for head, selected_token in T.Parallel(head_tile, block_tokens):
                    if head < query_heads_per_kv_head:
                        grad_probability[head, selected_token] = (
                            probability[head, selected_token]
                            * (grad_probability[head, selected_token] - delta_shared[head])
                            * scale
                        )
                    else:
                        grad_probability[head, selected_token] = 0
                T.copy(grad_probability, grad_score_shared)

                T.gemm(
                    grad_score_shared,
                    key_shared,
                    grad_query_accumulator,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.gemm(
                    grad_score_shared,
                    query_shared,
                    grad_kv_accumulator,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                T.copy(grad_kv_accumulator, atomic_store)
                for selected_token, dim in T.Parallel(block_tokens, head_dim // 4):
                    T.atomic_addx4(
                        grad_key[
                            batch,
                            indices[
                                batch,
                                query_token,
                                selection_block * block_tokens + selected_token,
                            ],
                            kv_head,
                            dim * 4,
                        ],
                        atomic_store[selected_token, dim * 4],
                    )

                T.gemm(
                    probability_shared,
                    grad_output_shared,
                    grad_kv_accumulator,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                T.copy(grad_kv_accumulator, atomic_store)
                for selected_token, dim in T.Parallel(block_tokens, head_dim // 4):
                    T.atomic_addx4(
                        grad_value[
                            batch,
                            indices[
                                batch,
                                query_token,
                                selection_block * block_tokens + selected_token,
                            ],
                            kv_head,
                            dim * 4,
                        ],
                        atomic_store[selected_token, dim * 4],
                    )

            T.copy(grad_query_accumulator, grad_query_shared)
            T.copy(
                grad_query_shared[:query_heads_per_kv_head, :],
                grad_query[
                    batch,
                    query_token,
                    first_query_head : first_query_head + query_heads_per_kv_head,
                    :,
                ],
            )

    return kernel


@torch.library.custom_op("prime_kernels::indexed_attention_backward", mutates_args=())
def indexed_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    _, _, num_query_heads, head_dim = query.shape
    _, _, num_kv_heads, _ = key.shape
    selection_width = indices.shape[-1]

    delta = attention_delta(num_query_heads, head_dim)(output, grad_output)
    grad_key = torch.zeros_like(key, dtype=torch.float32)
    grad_value = torch.zeros_like(value, dtype=torch.float32)
    grad_query = indexed_attention_backward_kernel(
        num_query_heads,
        num_kv_heads,
        head_dim,
        selection_width,
        scale,
    )(
        query,
        key,
        value,
        grad_output,
        indices,
        logsumexp,
        delta,
        grad_key,
        grad_value,
    )
    return grad_query, grad_key.to(query.dtype), grad_value.to(query.dtype)


@indexed_attention_backward.register_fake
def indexed_attention_backward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)
