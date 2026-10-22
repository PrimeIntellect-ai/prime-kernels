import torch
import triton
import triton.language as tl

from prime_kernels.indexed_attention.backward import indexed_attention_backward
from prime_kernels.indexed_attention.plan import MMA_ROWS, indexed_attention_plan, window_queries

LOG2E = 1.4426950408889634
BLOCK_UNION_KEYS = 64
MAX_QUERY_HEADS_PER_KV_HEAD = 16


@triton.jit
def _forward_kernel(
    query,
    key,
    value,
    output,
    logsumexp,
    window_offsets,
    union_keys,
    union_bits,
    query_length,
    num_heads,
    scale_log2,
    GROUP: tl.constexpr,
    WINDOW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    window = tl.program_id(0)
    kv_head = tl.program_id(1)
    num_kv_heads = tl.num_programs(1)
    rows = tl.arange(0, ROWS)
    row_query = rows // GROUP
    token = window * WINDOW + row_query
    row_valid = (row_query < WINDOW) & (token < query_length)
    head = kv_head * GROUP + rows % GROUP
    dims = tl.arange(0, HEAD_DIM)
    row_offsets = (token[:, None] * num_heads + head[:, None]) * HEAD_DIM + dims[None, :]
    q = tl.load(query + row_offsets, mask=row_valid[:, None], other=0.0)

    row_max = tl.full([ROWS], -1e30, tl.float32)
    row_sum = tl.zeros([ROWS], tl.float32)
    accumulator = tl.zeros([ROWS, HEAD_DIM], tl.float32)
    start = tl.load(window_offsets + window)
    end = tl.load(window_offsets + window + 1)
    for tile_start in range(start, end, BLOCK_N):
        columns = tile_start + tl.arange(0, BLOCK_N)
        column_valid = columns < end
        keys = tl.load(union_keys + columns, mask=column_valid, other=0)
        bits = tl.load(union_bits + columns, mask=column_valid, other=0).to(tl.int32)
        kv_offsets = (keys[:, None] * num_kv_heads + kv_head) * HEAD_DIM + dims[None, :]
        k = tl.load(key + kv_offsets)
        scores = tl.dot(q, tl.trans(k))
        selected = ((bits[None, :] >> row_query[:, None]) & 1) != 0
        scores = tl.where(selected, scores * scale_log2, float("-inf"))
        new_max = tl.maximum(row_max, tl.max(scores, 1))
        rescale = tl.exp2(row_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        row_sum = row_sum * rescale + tl.sum(probabilities, 1)
        v = tl.load(value + kv_offsets)
        accumulator = tl.dot(probabilities.to(tl.bfloat16), v, accumulator * rescale[:, None])
        row_max = new_max

    row_sum = tl.where(row_sum == 0, 1.0, row_sum)
    tl.store(output + row_offsets, (accumulator / row_sum[:, None]).to(tl.bfloat16), mask=row_valid[:, None])
    tl.store(logsumexp + token * num_heads + head, row_max + tl.log2(row_sum), mask=row_valid)


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
    if batch_size * kv_length >= 2**28:
        return "at most 2**28 key tokens are supported"
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
    window_offsets: torch.Tensor,
    union_keys: torch.Tensor,
    union_bits: torch.Tensor,
    tile_offsets: torch.Tensor,
    segment_windows: torch.Tensor,
    segment_masks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_length, num_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    group = num_heads // num_kv_heads
    output = torch.empty_like(query)
    logsumexp = query.new_empty(query.shape[:2], dtype=torch.float32)
    _forward_kernel[(window_offsets.numel() - 1, num_kv_heads)](
        query,
        key,
        value,
        output,
        logsumexp,
        window_offsets,
        union_keys,
        union_bits,
        query_length,
        num_heads,
        scale * LOG2E,
        GROUP=group,
        WINDOW=window_queries(group),
        HEAD_DIM=head_dim,
        ROWS=MMA_ROWS,
        BLOCK_N=BLOCK_UNION_KEYS,
        num_warps=4,
        num_stages=1,
    )
    return output, logsumexp


@indexed_attention_forward.register_fake
def indexed_attention_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_offsets: torch.Tensor,
    union_keys: torch.Tensor,
    union_bits: torch.Tensor,
    tile_offsets: torch.Tensor,
    segment_windows: torch.Tensor,
    segment_masks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), query.new_empty(query.shape[:2], dtype=torch.float32)


def indexed_attention_setup_context(ctx, inputs, output) -> None:
    *tensors, scale = inputs
    attention_output, logsumexp = output
    ctx.save_for_backward(*tensors, attention_output, logsumexp)
    ctx.scale = scale
    ctx.mark_non_differentiable(logsumexp)


def indexed_attention_autograd_backward(ctx, grad_output: torch.Tensor, _grad_logsumexp: torch.Tensor | None):
    query, key, value, *plan, output, logsumexp = ctx.saved_tensors
    grad_query, grad_key, grad_value = indexed_attention_backward(
        query, key, value, output, grad_output, logsumexp, *plan, ctx.scale
    )
    return grad_query, grad_key, grad_value, None, None, None, None, None, None, None


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

    Indices are in the key sequence's coordinate space. Values of at least ``key.shape[1]`` mark
    unused entries, and every query must select at least one real token. The backward pass is
    deterministic.
    """
    reason = unsupported_shape_reason(query, key, value, indices)
    if reason is not None:
        raise ValueError(reason)

    scale = query.shape[-1] ** -0.5 if scale is None else scale
    batch_size, query_length, num_heads, head_dim = query.shape
    kv_length, num_kv_heads = key.shape[1:3]
    if batch_size > 1:
        offsets = torch.arange(batch_size, device=indices.device, dtype=torch.int32)[:, None, None] * kv_length
        indices = torch.where(indices < kv_length, indices + offsets, batch_size * kv_length)
    plan = indexed_attention_plan(
        indices.reshape(batch_size * query_length, -1).contiguous(),
        batch_size * kv_length,
        window_queries(num_heads // num_kv_heads),
    )
    output, _ = indexed_attention_forward(
        query.reshape(batch_size * query_length, num_heads, head_dim).contiguous(),
        key.reshape(batch_size * kv_length, num_kv_heads, head_dim).contiguous(),
        value.reshape(batch_size * kv_length, num_kv_heads, head_dim).contiguous(),
        *plan,
        scale,
    )
    return output.view(batch_size, query_length, num_heads, head_dim)
