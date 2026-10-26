import torch
import triton
import triton.language as tl

from prime_kernels.indexed_attention.plan import BLOCK_KEYS, MMA_ROWS, window_queries

LOG2E = 1.4426950408889634
BLOCK_UNION_KEYS = 64


@triton.jit
def _backward_query_kernel(
    query,
    key,
    value,
    output,
    grad_output,
    logsumexp,
    delta,
    grad_query,
    window_offsets,
    union_keys,
    union_bits,
    query_length,
    num_heads,
    scale,
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
    do = tl.load(grad_output + row_offsets, mask=row_valid[:, None], other=0.0)
    o = tl.load(output + row_offsets, mask=row_valid[:, None], other=0.0)
    row_delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    statistics = token * num_heads + head
    tl.store(delta + statistics, row_delta, mask=row_valid)
    row_logsumexp = tl.load(logsumexp + statistics, mask=row_valid, other=0.0)

    dq = tl.zeros([ROWS, HEAD_DIM], tl.float32)
    start = tl.load(window_offsets + window)
    end = tl.load(window_offsets + window + 1)
    for tile_start in range(start, end, BLOCK_N):
        columns = tile_start + tl.arange(0, BLOCK_N)
        column_valid = columns < end
        keys = tl.load(union_keys + columns, mask=column_valid, other=0)
        bits = tl.load(union_bits + columns, mask=column_valid, other=0).to(tl.int32)
        kv_offsets = (keys[:, None] * num_kv_heads + kv_head) * HEAD_DIM + dims[None, :]
        k = tl.load(key + kv_offsets)
        v = tl.load(value + kv_offsets)
        selected = (((bits[None, :] >> row_query[:, None]) & 1) != 0) & row_valid[:, None]
        p = tl.where(selected, tl.exp2(tl.dot(q, tl.trans(k)) * scale_log2 - row_logsumexp[:, None]), 0.0)
        ds = p * (tl.dot(do, tl.trans(v)) - row_delta[:, None]) * scale
        dq = tl.dot(ds.to(tl.bfloat16), k, dq)
    tl.store(grad_query + row_offsets, dq.to(tl.bfloat16), mask=row_valid[:, None])


@triton.jit
def _backward_key_value_kernel(
    query,
    key,
    value,
    grad_output,
    logsumexp,
    delta,
    grad_key,
    grad_value,
    tile_offsets,
    segment_windows,
    segment_masks,
    query_length,
    kv_length,
    num_heads,
    scale,
    scale_log2,
    GROUP: tl.constexpr,
    WINDOW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    num_kv_heads = tl.num_programs(1)
    local_keys = tl.arange(0, BLOCK_N)
    keys = tile * BLOCK_N + local_keys
    key_valid = keys < kv_length
    dims = tl.arange(0, HEAD_DIM)
    kv_offsets = (keys[:, None] * num_kv_heads + kv_head) * HEAD_DIM + dims[None, :]
    k = tl.load(key + kv_offsets, mask=key_valid[:, None], other=0.0)
    v = tl.load(value + kv_offsets, mask=key_valid[:, None], other=0.0)
    rows = tl.arange(0, ROWS)
    row_query = rows // GROUP
    head = kv_head * GROUP + rows % GROUP

    # Each tile walks its windows in a fixed order and owns its rows of dK/dV: no atomics.
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    for segment in range(tl.load(tile_offsets + tile), tl.load(tile_offsets + tile + 1)):
        window = tl.load(segment_windows + segment)
        key_mask = tl.load(segment_masks + segment * WINDOW + row_query, mask=row_query < WINDOW, other=0)
        token = window * WINDOW + row_query
        row_valid = (row_query < WINDOW) & (token < query_length)
        row_offsets = (token[:, None] * num_heads + head[:, None]) * HEAD_DIM + dims[None, :]
        q = tl.load(query + row_offsets, mask=row_valid[:, None], other=0.0)
        do = tl.load(grad_output + row_offsets, mask=row_valid[:, None], other=0.0)
        statistics = token * num_heads + head
        row_logsumexp = tl.load(logsumexp + statistics, mask=row_valid, other=0.0)
        row_delta = tl.load(delta + statistics, mask=row_valid, other=0.0)

        selected = ((key_mask[None, :] >> local_keys[:, None]) & 1) != 0
        pt = tl.where(selected, tl.exp2(tl.dot(k, tl.trans(q)) * scale_log2 - row_logsumexp[None, :]), 0.0)
        dst = pt * (tl.dot(v, tl.trans(do)) - row_delta[None, :]) * scale
        dv = tl.dot(pt.to(tl.bfloat16), do, dv)
        dk = tl.dot(dst.to(tl.bfloat16), q, dk)
    tl.store(grad_key + kv_offsets, dk.to(tl.bfloat16), mask=key_valid[:, None])
    tl.store(grad_value + kv_offsets, dv.to(tl.bfloat16), mask=key_valid[:, None])


@torch.library.custom_op("prime_kernels::indexed_attention_backward", mutates_args=())
def indexed_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    window_offsets: torch.Tensor,
    union_keys: torch.Tensor,
    union_bits: torch.Tensor,
    tile_offsets: torch.Tensor,
    segment_windows: torch.Tensor,
    segment_masks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    query_length, num_heads, head_dim = query.shape
    kv_length, num_kv_heads, _ = key.shape
    group = num_heads // num_kv_heads
    meta = dict(GROUP=group, WINDOW=window_queries(group), HEAD_DIM=head_dim, ROWS=MMA_ROWS, num_warps=4, num_stages=3)
    delta = torch.empty_like(logsumexp)
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    _backward_query_kernel[(window_offsets.numel() - 1, num_kv_heads)](
        query,
        key,
        value,
        output,
        grad_output,
        logsumexp,
        delta,
        grad_query,
        window_offsets,
        union_keys,
        union_bits,
        query_length,
        num_heads,
        scale,
        scale * LOG2E,
        BLOCK_N=BLOCK_UNION_KEYS,
        **meta,
    )
    _backward_key_value_kernel[(tile_offsets.numel() - 1, num_kv_heads)](
        query,
        key,
        value,
        grad_output,
        logsumexp,
        delta,
        grad_key,
        grad_value,
        tile_offsets,
        segment_windows,
        segment_masks,
        query_length,
        kv_length,
        num_heads,
        scale,
        scale * LOG2E,
        BLOCK_N=BLOCK_KEYS,
        **meta,
    )
    return grad_query, grad_key, grad_value


@indexed_attention_backward.register_fake
def indexed_attention_backward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    window_offsets: torch.Tensor,
    union_keys: torch.Tensor,
    union_bits: torch.Tensor,
    tile_offsets: torch.Tensor,
    segment_windows: torch.Tensor,
    segment_masks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)
