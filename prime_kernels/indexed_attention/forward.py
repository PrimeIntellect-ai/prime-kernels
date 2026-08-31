# Vendored from tile-ai/tilelang (Apache 2.0), modified for indexed GQA.

import tilelang
import torch
import torch.nn.functional as F
from tilelang import language as T

from prime_kernels.indexed_attention.backward import indexed_attention_backward

FORWARD_BLOCK_TOKENS = 64
MAX_QUERY_HEADS_PER_KV_HEAD = 16


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def indexed_attention_forward_kernel(
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    selection_width: int,
    scale: float,
    block_tokens: int = FORWARD_BLOCK_TOKENS,
    num_stages: int = 2,
    threads: int = 256,
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
    output_shape = query_shape
    logsumexp_shape = [batch_size, query_length, num_query_heads]

    @T.prim_func
    def kernel(
        query: T.Tensor(query_shape, T.bfloat16),
        key: T.Tensor(kv_shape, T.bfloat16),
        value: T.Tensor(kv_shape, T.bfloat16),
        indices: T.Tensor(indices_shape, T.int32),
        output: T.Tensor(output_shape, T.bfloat16),
        logsumexp: T.Tensor(logsumexp_shape, T.float32),
    ):
        with T.Kernel(query_length, batch_size, num_kv_heads, threads=threads) as (query_token, batch, kv_head):
            query_shared = T.alloc_shared([head_tile, head_dim], T.bfloat16)
            key_shared = T.alloc_shared([block_tokens, head_dim], T.bfloat16)
            value_shared = T.alloc_shared([block_tokens, head_dim], T.bfloat16)
            probability_shared = T.alloc_shared([head_tile, block_tokens], T.bfloat16)
            output_shared = T.alloc_shared([head_tile, head_dim], T.bfloat16)
            logsumexp_shared = T.alloc_shared([head_tile], T.float32)

            valid_index = T.alloc_fragment([block_tokens], "bool")
            scores = T.alloc_fragment([head_tile, block_tokens], T.float32)
            output_accumulator = T.alloc_fragment([head_tile, head_dim], T.float32)
            row_sum = T.alloc_fragment([head_tile], T.float32)
            block_row_sum = T.alloc_fragment([head_tile], T.float32)
            rescale = T.alloc_fragment([head_tile], T.float32)
            row_max = T.alloc_fragment([head_tile], T.float32)
            previous_row_max = T.alloc_fragment([head_tile], T.float32)

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
            for head, dim in T.Parallel(head_tile - query_heads_per_kv_head, head_dim):
                query_shared[query_heads_per_kv_head + head, dim] = 0

            T.clear(output_accumulator)
            T.clear(row_sum)
            T.fill(row_max, -(2**30))

            for selection_block in T.Pipelined(num_token_blocks, num_stages=num_stages):
                for selected_token in T.Parallel(block_tokens):
                    valid_index[selected_token] = (
                        indices[batch, query_token, selection_block * block_tokens + selected_token] <= last_valid_kv
                    )
                for head, selected_token in T.Parallel(head_tile, block_tokens):
                    scores[head, selected_token] = T.if_then_else(
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
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(row_max, previous_row_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)
                for head in T.Parallel(head_tile):
                    row_max[head] = T.max(row_max[head], previous_row_max[head])
                    rescale[head] = T.exp2((previous_row_max[head] - row_max[head]) * log2_scale)
                for head, selected_token in T.Parallel(head_tile, block_tokens):
                    scores[head, selected_token] = T.exp2(
                        scores[head, selected_token] * log2_scale - row_max[head] * log2_scale
                    )
                T.reduce_sum(scores, block_row_sum, dim=1)
                for head in T.Parallel(head_tile):
                    row_sum[head] = row_sum[head] * rescale[head] + block_row_sum[head]
                for head, dim in T.Parallel(head_tile, head_dim):
                    output_accumulator[head, dim] *= rescale[head]

                T.copy(scores, probability_shared)
                for selected_token, dim in T.Parallel(block_tokens, head_dim):
                    value_shared[selected_token, dim] = value[
                        batch,
                        indices[batch, query_token, selection_block * block_tokens + selected_token],
                        kv_head,
                        dim,
                    ]
                T.gemm(
                    probability_shared,
                    value_shared,
                    output_accumulator,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for head, dim in T.Parallel(head_tile, head_dim):
                output_accumulator[head, dim] /= row_sum[head]
            for head in T.Parallel(head_tile):
                row_sum[head] = T.log2(row_sum[head]) + row_max[head] * log2_scale

            T.copy(output_accumulator, output_shared)
            T.copy(row_sum, logsumexp_shared)
            T.copy(
                output_shared[:query_heads_per_kv_head, :],
                output[
                    batch,
                    query_token,
                    first_query_head : first_query_head + query_heads_per_kv_head,
                    :,
                ],
            )
            T.copy(
                logsumexp_shared[:query_heads_per_kv_head],
                logsumexp[
                    batch,
                    query_token,
                    first_query_head : first_query_head + query_heads_per_kv_head,
                ],
            )

    return kernel


def unsupported_shape_reason(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
) -> str | None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return "query, key, and value must have shape [batch, tokens, heads, head_dim]"
    if indices.ndim != 3:
        return "indices must have shape [batch, query_tokens, selected_tokens]"
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        return "query, key, and value must use bfloat16"
    if indices.dtype != torch.int32:
        return "indices must use int32"
    if not query.is_cuda or not key.is_cuda or not value.is_cuda or not indices.is_cuda:
        return "all inputs must be CUDA tensors"
    if not (query.device == key.device == value.device == indices.device):
        return "all inputs must be on the same CUDA device"
    batch_size, query_length, num_query_heads, head_dim = query.shape
    key_batch, kv_length, num_kv_heads, key_head_dim = key.shape
    if value.shape != key.shape:
        return "key and value must have the same shape"
    if key_batch != batch_size or key_head_dim != head_dim:
        return "query, key, and value batch and head dimensions must match"
    if indices.shape[:2] != (batch_size, query_length):
        return "indices batch and query dimensions must match query"
    if query_length == 0 or kv_length == 0 or indices.shape[-1] == 0:
        return "query, key, value, and selection dimensions must be non-empty"
    if num_kv_heads == 0 or num_query_heads % num_kv_heads:
        return "query heads must be divisible by KV heads"
    if num_query_heads // num_kv_heads > MAX_QUERY_HEADS_PER_KV_HEAD:
        return f"at most {MAX_QUERY_HEADS_PER_KV_HEAD} query heads per KV head are supported"
    if head_dim < 16 or head_dim & (head_dim - 1):
        return "head_dim must be a power of two and at least 16"
    return None


@torch.library.custom_op("prime_kernels::indexed_attention_forward", mutates_args=())
def indexed_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, num_query_heads, head_dim = query.shape
    _, _, num_kv_heads, _ = key.shape
    selection_width = indices.shape[-1]
    return indexed_attention_forward_kernel(
        num_query_heads,
        num_kv_heads,
        head_dim,
        selection_width,
        scale,
    )(query, key, value, indices)


@indexed_attention_forward.register_fake
def indexed_attention_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), query.new_empty(query.shape[:-1], dtype=torch.float32)


def indexed_attention_setup_context(ctx, inputs, output) -> None:
    query, key, value, indices, scale = inputs
    attention_output, logsumexp = output
    ctx.save_for_backward(query, key, value, attention_output, indices, logsumexp)
    ctx.scale = scale
    ctx.mark_non_differentiable(logsumexp)


def indexed_attention_autograd_backward(ctx, grad_output: torch.Tensor, _grad_logsumexp: torch.Tensor | None):
    query, key, value, output, indices, logsumexp = ctx.saved_tensors
    grad_query, grad_key, grad_value = indexed_attention_backward(
        query.detach(),
        key.detach(),
        value.detach(),
        output.detach(),
        grad_output,
        indices,
        logsumexp.detach(),
        ctx.scale,
    )
    return grad_query, grad_key, grad_value, None, None


indexed_attention_forward.register_autograd(
    indexed_attention_autograd_backward,
    setup_context=indexed_attention_setup_context,
)


def indexed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Apply GQA over each query's selected tokens.

    Indices are in the key sequence's coordinate space. The value ``key.shape[1]`` is a
    sentinel for unused entries, and every query must select at least one real token.
    """
    reason = unsupported_shape_reason(query, key, value, indices)
    if reason is not None:
        raise ValueError(reason)

    scale = query.shape[-1] ** -0.5 if scale is None else scale
    sentinel = key.shape[1]
    key = torch.cat((key, key.new_zeros((key.shape[0], 1, key.shape[2], key.shape[3]))), dim=1)
    value = torch.cat((value, value.new_zeros((value.shape[0], 1, value.shape[2], value.shape[3]))), dim=1)

    padding = (-indices.shape[-1]) % FORWARD_BLOCK_TOKENS
    if padding:
        indices = F.pad(indices, (0, padding), value=sentinel)

    output, _ = indexed_attention_forward(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        indices.contiguous(),
        scale,
    )
    return output
