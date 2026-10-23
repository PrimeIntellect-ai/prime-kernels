import torch
import triton
import triton.language as tl

# The query bits of a key are packed below its index, so a window holds at most 8 queries.
MAX_WINDOW_QUERIES = 8
MMA_ROWS = 64
BLOCK_KEYS = 32
SORT_CHUNK_WINDOWS = 1024
_EMPTY = tl.constexpr(0x7FFFFFFF)


def window_queries(query_heads_per_kv_head: int) -> int:
    """Consecutive queries whose heads of one KV group share the 64 MMA rows."""
    return min(MAX_WINDOW_QUERIES, MMA_ROWS // query_heads_per_kv_head)


@triton.jit
def _or_by_key(key_a, a, key_b, b):
    return key_b, tl.where(key_a == key_b, a | b, b)


@triton.jit
def _window_keys_kernel(
    sorted_keys,
    row_width,
    counts,
    window_offsets,
    union_keys,
    union_bits,
    segment_offsets,
    segment_tiles,
    segment_ends,
    WRITE: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_KEYS: tl.constexpr,
):
    window = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    packed = tl.load(sorted_keys + window * row_width + i, mask=i < row_width, other=_EMPTY)
    key = packed >> 3
    following = tl.minimum(i + 1, BLOCK - 1)
    is_last = (packed != _EMPTY) & ((key != tl.gather(key, following, 0)) | (i == BLOCK - 1))
    tile = key // BLOCK_KEYS
    segment_last = is_last & ((tile != tl.gather(tile, following, 0)) | (i == BLOCK - 1))
    if not WRITE:
        tl.store(counts + window * 2, tl.sum(is_last.to(tl.int32)))
        tl.store(counts + window * 2 + 1, tl.sum(segment_last.to(tl.int32)))
    else:
        _, bits = tl.associative_scan((key, 1 << (packed & 7)), 0, _or_by_key)
        position = tl.load(window_offsets + window) + tl.cumsum(is_last.to(tl.int32), 0) - 1
        tl.store(union_keys + position, key, mask=is_last)
        tl.store(union_bits + position, bits.to(tl.int8), mask=is_last)
        segment = tl.load(segment_offsets + window) + tl.cumsum(segment_last.to(tl.int32), 0) - 1
        tl.store(segment_tiles + segment, tile, mask=segment_last)
        tl.store(segment_ends + segment, position + 1, mask=segment_last)


@triton.jit
def _segment_masks_kernel(
    union_keys,
    union_bits,
    segment_ends,
    segment_masks,
    num_segments,
    WINDOW: tl.constexpr,
    BLOCK_KEYS: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
):
    segment = tl.program_id(0) * BLOCK_SEGMENTS + tl.arange(0, BLOCK_SEGMENTS)
    valid = segment < num_segments
    end = tl.load(segment_ends + segment, mask=valid, other=0)
    begin = tl.load(segment_ends + segment - 1, mask=valid & (segment > 0), other=0)
    entry = begin[:, None] + tl.arange(0, BLOCK_KEYS)[None, :]
    entry_valid = entry < end[:, None]
    key = tl.load(union_keys + entry, mask=entry_valid, other=0)
    bits = tl.load(union_bits + entry, mask=entry_valid, other=0).to(tl.int32)
    key_bit = 1 << (key % BLOCK_KEYS)
    for query in tl.static_range(WINDOW):
        tl.store(
            segment_masks + segment * WINDOW + query, tl.sum(tl.where((bits >> query) & 1, key_bit, 0), 1), mask=valid
        )


@torch.library.custom_op("prime_kernels::indexed_attention_plan", mutates_args=())
def indexed_attention_plan(
    indices: torch.Tensor,
    kv_length: int,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group the selections of `window` consecutive queries for the attention kernels.

    Returns the window side, as CSR over windows: the sorted distinct keys the window's queries
    select and, per key, a bitmask of the queries that select it. And the key side, as CSR over
    tiles of 32 keys: the windows that select any key of the tile, in window order, with one
    bitmask over the tile's keys per window query. Indices equal to or above `kv_length` are unused.
    """
    query_length, width = indices.shape
    num_windows = triton.cdiv(query_length, window)
    device = indices.device
    padded = torch.nn.functional.pad(indices, (0, 0, 0, num_windows * window - query_length), value=kv_length)
    padded = padded.view(num_windows, window, width)
    query = torch.arange(window, dtype=torch.int32, device=device)[None, :, None]
    row_width = window * width
    packed = torch.empty(num_windows, row_width, dtype=torch.int32, device=device)
    # In chunks, to bound the sort's temporaries (it also returns int64 positions).
    for start in range(0, num_windows, SORT_CHUNK_WINDOWS):
        chunk = padded[start : start + SORT_CHUNK_WINDOWS]
        chunk = torch.where(chunk < kv_length, chunk * 8 + query, _EMPTY.value).view(chunk.shape[0], -1)
        packed[start : start + SORT_CHUNK_WINDOWS] = chunk.sort(dim=1).values
    block = triton.next_power_of_2(row_width)

    counts = torch.empty(num_windows, 2, dtype=torch.int32, device=device)
    _window_keys_kernel[(num_windows,)](
        packed,
        row_width,
        counts,
        None,
        None,
        None,
        None,
        None,
        None,
        WRITE=False,
        BLOCK=block,
        BLOCK_KEYS=BLOCK_KEYS,
        num_warps=8,
    )
    offsets = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (0, 0, 1, 0))
    num_keys, num_segments = offsets[-1].tolist()
    window_offsets = offsets[:, 0].contiguous()
    union_keys = torch.empty(num_keys, dtype=torch.int32, device=device)
    union_bits = torch.empty(num_keys, dtype=torch.int8, device=device)
    segment_tiles = torch.empty(num_segments, dtype=torch.int32, device=device)
    segment_ends = torch.empty(num_segments, dtype=torch.int32, device=device)
    _window_keys_kernel[(num_windows,)](
        packed,
        row_width,
        counts,
        window_offsets,
        union_keys,
        union_bits,
        offsets[:, 1].contiguous(),
        segment_tiles,
        segment_ends,
        WRITE=True,
        BLOCK=block,
        BLOCK_KEYS=BLOCK_KEYS,
        num_warps=8,
    )
    segment_masks = torch.empty(num_segments, window, dtype=torch.int32, device=device)
    _segment_masks_kernel[(triton.cdiv(num_segments, 128),)](
        union_keys,
        union_bits,
        segment_ends,
        segment_masks,
        num_segments,
        WINDOW=window,
        BLOCK_KEYS=BLOCK_KEYS,
        BLOCK_SEGMENTS=128,
    )

    segment_windows = torch.repeat_interleave(
        torch.arange(num_windows, dtype=torch.int32, device=device), counts[:, 1], output_size=num_segments
    )
    order = segment_tiles.argsort(stable=True)
    tile_counts = torch.bincount(segment_tiles, minlength=triton.cdiv(kv_length, BLOCK_KEYS))
    tile_offsets = torch.nn.functional.pad(tile_counts.cumsum(0, dtype=torch.int32), (1, 0))
    return window_offsets, union_keys, union_bits, tile_offsets, segment_windows[order], segment_masks[order]


@indexed_attention_plan.register_fake
def indexed_attention_plan_fake(indices: torch.Tensor, kv_length: int, window: int):
    ctx = torch.library.get_ctx()
    num_keys = ctx.new_dynamic_size()
    num_segments = ctx.new_dynamic_size()
    num_windows = triton.cdiv(indices.shape[0], window)
    int32 = dict(dtype=torch.int32, device=indices.device)
    return (
        indices.new_empty(num_windows + 1, **int32),
        indices.new_empty(num_keys, **int32),
        indices.new_empty(num_keys, dtype=torch.int8),
        indices.new_empty(triton.cdiv(kv_length, BLOCK_KEYS) + 1, **int32),
        indices.new_empty(num_segments, **int32),
        indices.new_empty(num_segments, window, **int32),
    )
