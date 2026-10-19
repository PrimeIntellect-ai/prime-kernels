"""Triton kernels of the fused Lightning Indexer top-k.

Scores are `score[t, e] = k_scale[e] * sum_h relu(q_fp8[t, h] . k_fp8[e]) * w[t, h]`, where `w` already
carries the per-(token, head) query scale. Three kernels share that formula:

- `_group_max_kernel`: the dense query-block x entry-tile GEMM. It keeps no scores, only the max of
  every doc-local group of `GROUP` entries (`GROUP` = the candidate block size) and of every
  super-group of `SUPER` consecutive groups.
- `_group_topk_kernel`: per query, selects the best groups by their max.
- `_gather_topk_kernel`: per query, scores only the entries of a given list of groups and selects
  the best `TOPK` of them.

Every selection is hierarchical, which is exact for the following reason. Split a row into blocks;
if `tau` is the `k`-th largest block max, at least `k` values are `>= tau`, so every top-`k` value
is `>= tau`. Selecting the `k`-th largest block max, keeping only the values `>= tau` (a few more
than `k` for typical rows, at most `k` blocks' worth) and selecting among those gives the top `k`.
The same argument makes V4.1's candidate filter cheap: the dense pass only needs block maxima, and
the final ranking only rescores the entries of the top `topk` groups.

Selection itself is a 4-pass 8-bit radix select over an order-preserving uint32 image of the fp32
values, followed by a pass that writes the strictly larger values and the first ties in order.
"""

import triton
import triton.language as tl

SUPER = tl.constexpr(4)


@triton.jit
def _ordered_key(v):
    """fp32 -> uint32 with the same order (larger float, larger key)."""
    bits = v.to(tl.uint32, bitcast=True)
    return tl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)


@triton.jit
def _load_keys(row_ptr, cols, n_cols):
    """Keys of `row_ptr[cols]` and which are valid (finite)."""
    v = tl.load(row_ptr + cols, mask=cols < n_cols, other=float("-inf"))
    return _ordered_key(v), v > float("-inf")


@triton.jit
def _digit_choice(hist, prefix, prefix_mask, remaining, SHIFT: tl.constexpr):
    """Given the histogram of the 8-bit digit at `SHIFT` among the keys matching `prefix`, fix that
    digit of the `remaining`-th largest key."""
    bins = tl.arange(0, 256)
    at_or_above = tl.cumsum(hist, 0, reverse=True)
    digit = tl.max(tl.where(at_or_above >= remaining, bins, -1))
    above = tl.sum(tl.where(bins > digit, hist, 0))
    prefix = prefix | (digit.to(tl.uint32) << SHIFT)
    prefix_mask = prefix_mask | (tl.full([], 0xFF, tl.uint32) << SHIFT)
    return prefix, prefix_mask, remaining - above


@triton.jit
def _radix_pass_regs(key, valid, prefix, prefix_mask, remaining, SHIFT: tl.constexpr):
    match = valid & ((key & prefix_mask) == prefix)
    hist = tl.histogram(((key >> SHIFT) & 0xFF).to(tl.int32), 256, mask=match)
    return _digit_choice(hist, prefix, prefix_mask, remaining, SHIFT)


@triton.jit
def _radix_pass_mem(row_ptr, n_cols, prefix, prefix_mask, remaining, SHIFT: tl.constexpr, CHUNK: tl.constexpr):
    hist = tl.zeros([256], dtype=tl.int32)
    for c0 in range(0, n_cols, CHUNK):
        key, valid = _load_keys(row_ptr, c0 + tl.arange(0, CHUNK), n_cols)
        match = valid & ((key & prefix_mask) == prefix)
        hist += tl.histogram(((key >> SHIFT) & 0xFF).to(tl.int32), 256, mask=match)
    return _digit_choice(hist, prefix, prefix_mask, remaining, SHIFT)


@triton.jit
def _kth_key_regs(key, valid, k):
    """Key of the `k`-th largest valid key and how many of the top `k` equal it (needs `k` valid)."""
    prefix = tl.zeros([], dtype=tl.uint32)
    prefix_mask = tl.zeros([], dtype=tl.uint32)
    remaining = tl.zeros([], dtype=tl.int32) + k
    prefix, prefix_mask, remaining = _radix_pass_regs(key, valid, prefix, prefix_mask, remaining, 24)
    prefix, prefix_mask, remaining = _radix_pass_regs(key, valid, prefix, prefix_mask, remaining, 16)
    prefix, prefix_mask, remaining = _radix_pass_regs(key, valid, prefix, prefix_mask, remaining, 8)
    prefix, prefix_mask, remaining = _radix_pass_regs(key, valid, prefix, prefix_mask, remaining, 0)
    return prefix, remaining


@triton.jit
def _kth_key_mem(row_ptr, n_cols, k, CHUNK: tl.constexpr):
    """`_kth_key_regs` over `row_ptr[:n_cols]`, streamed in chunks."""
    prefix = tl.zeros([], dtype=tl.uint32)
    prefix_mask = tl.zeros([], dtype=tl.uint32)
    remaining = tl.zeros([], dtype=tl.int32) + k
    prefix, prefix_mask, remaining = _radix_pass_mem(row_ptr, n_cols, prefix, prefix_mask, remaining, 24, CHUNK)
    prefix, prefix_mask, remaining = _radix_pass_mem(row_ptr, n_cols, prefix, prefix_mask, remaining, 16, CHUNK)
    prefix, prefix_mask, remaining = _radix_pass_mem(row_ptr, n_cols, prefix, prefix_mask, remaining, 8, CHUNK)
    prefix, prefix_mask, remaining = _radix_pass_mem(row_ptr, n_cols, prefix, prefix_mask, remaining, 0, CHUNK)
    return prefix, remaining


@triton.jit
def _block_threshold(bmax_ptr, n_blocks, K: tl.constexpr, CHUNK: tl.constexpr, CAP: tl.constexpr):
    """Key of the `K`-th largest valid block max, or 0 (below every valid key) if fewer are valid."""
    threshold = tl.zeros([], dtype=tl.uint32)
    if n_blocks <= CAP:
        key, valid = _load_keys(bmax_ptr, tl.arange(0, CAP), n_blocks)
        if tl.sum(valid.to(tl.int32)) >= K:
            threshold, _ = _kth_key_regs(key, valid, K)
    else:
        n_valid = tl.zeros([], dtype=tl.int32)
        for c0 in range(0, n_blocks, CHUNK):
            _, valid = _load_keys(bmax_ptr, c0 + tl.arange(0, CHUNK), n_blocks)
            n_valid += tl.sum(valid.to(tl.int32))
        if n_valid >= K:
            threshold, _ = _kth_key_mem(bmax_ptr, n_blocks, K, CHUNK)
    return threshold


@triton.jit
def _store_picks(out_ptr, pos, idx, take, blk_ptr, ks, GROUP: tl.constexpr, GATHERED: tl.constexpr):
    """Write column `idx` (as an entry index when `GATHERED`) to `out_ptr[pos]` where `take`."""
    if GATHERED:
        blk = tl.load(blk_ptr + idx // GROUP, mask=take, other=0)
        value = ks.to(tl.int64) + blk.to(tl.int64) * GROUP + (idx % GROUP)
    else:
        value = idx
    tl.store(out_ptr + pos, value.to(out_ptr.dtype.element_ty), mask=take)


@triton.jit
def _select_compacted(
    cval_ptr,
    cidx_ptr,
    n_kept,
    out_ptr,
    blk_ptr,
    ks,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    CHUNK: tl.constexpr,
    CAP: tl.constexpr,
    GATHERED: tl.constexpr,
):
    """Write the columns `cidx_ptr[c]` of the `K` largest of `cval_ptr[:n_kept]` (all valid), -1 after.

    Ties at the `K`-th value are taken in compaction order. Up to `CAP` values are selected in
    registers, more are streamed. Returns the key of the `K`-th largest value (0 if fewer).
    """
    tl.static_assert(CAP >= K)
    n_take = tl.minimum(n_kept, K)
    if n_kept <= CAP:
        offs = tl.arange(0, CAP)
        valid = offs < n_kept
        key = _ordered_key(tl.load(cval_ptr + offs, mask=valid, other=float("-inf")))
        if n_kept > K:
            threshold, remaining = _kth_key_regs(key, valid, K)
            bound = threshold
        else:
            threshold = tl.zeros([], dtype=tl.uint32)
            remaining = tl.zeros([], dtype=tl.int32)
            bound = tl.where(n_kept == K, tl.min(tl.where(valid, key, 0xFFFFFFFF)), 0).to(tl.uint32)
        greater = (valid & (key > threshold)).to(tl.int32)
        equal = (valid & (key == threshold)).to(tl.int32)
        equal_rank = tl.cumsum(equal, 0) - 1
        take = (greater != 0) | ((equal != 0) & (equal_rank < remaining))
        pos = tl.where(greater != 0, tl.cumsum(greater, 0) - 1, n_take - remaining + equal_rank)
        idx = tl.load(cidx_ptr + offs, mask=take, other=0)
        _store_picks(out_ptr, pos, idx, take, blk_ptr, ks, GROUP, GATHERED)
    else:
        threshold, remaining = _kth_key_mem(cval_ptr, n_kept, K, CHUNK)
        bound = threshold
        seen_greater = tl.zeros([], dtype=tl.int32)
        seen_equal = tl.zeros([], dtype=tl.int32)
        for c0 in range(0, n_kept, CHUNK):
            cols = c0 + tl.arange(0, CHUNK)
            key, valid = _load_keys(cval_ptr, cols, n_kept)
            greater = (valid & (key > threshold)).to(tl.int32)
            equal = (valid & (key == threshold)).to(tl.int32)
            equal_rank = seen_equal + tl.cumsum(equal, 0) - 1
            take = (greater != 0) | ((equal != 0) & (equal_rank < remaining))
            pos = tl.where(greater != 0, seen_greater + tl.cumsum(greater, 0) - 1, K - remaining + equal_rank)
            idx = tl.load(cidx_ptr + cols, mask=take, other=0)
            _store_picks(out_ptr, pos, idx, take, blk_ptr, ks, GROUP, GATHERED)
            seen_greater += tl.sum(greater)
            seen_equal += tl.sum(equal)
    for p0 in range(0, K, CHUNK):
        pos = p0 + tl.arange(0, CHUNK)
        tl.store(out_ptr + pos, tl.full([CHUNK], -1, out_ptr.dtype.element_ty), mask=(pos >= n_take) & (pos < K))
    return bound


@triton.jit
def _key_to_float(key):
    """Inverse of `_ordered_key`, with key 0 (below every valid key) mapped to -inf."""
    bits = tl.where((key >> 31) != 0, key ^ 0x80000000, ~key)
    return tl.where(key == 0, float("-inf"), bits.to(tl.float32, bitcast=True))


@triton.jit
def _compact(row_ptr, n_cols, pin, tau, cval_ptr, cidx_ptr, CHUNK: tl.constexpr):
    """Copy the valid values `>= tau` of `row_ptr[:n_cols]` and their columns to `cval_ptr`/`cidx_ptr`;
    column `pin` (if in range) is always kept, as +inf. Returns the counts kept and valid and the
    largest unpinned value."""
    n_kept = tl.zeros([], dtype=tl.int32)
    n_valid = tl.zeros([], dtype=tl.int32)
    row_max = tl.full([], float("-inf"), tl.float32)
    for c0 in range(0, n_cols, CHUNK):
        cols = c0 + tl.arange(0, CHUNK)
        v = tl.load(row_ptr + cols, mask=cols < n_cols, other=float("-inf"))
        is_pin = (cols == pin) & (cols < n_cols)
        valid = (v > float("-inf")) | is_pin
        keep = (valid & (v >= tau)) | is_pin
        rank = n_kept + tl.cumsum(keep.to(tl.int32), 0) - 1
        tl.store(cval_ptr + rank, tl.where(is_pin, float("inf"), v), mask=keep)
        tl.store(cidx_ptr + rank, cols, mask=keep)
        n_kept += tl.sum(keep.to(tl.int32))
        n_valid += tl.sum(valid.to(tl.int32))
        row_max = tl.maximum(row_max, tl.max(tl.where(is_pin, float("-inf"), v)))
    return n_kept, n_valid, row_max


@triton.jit
def _topk_row(
    row_ptr,
    n_cols,
    pin,
    tau,
    cval_ptr,
    cidx_ptr,
    out_ptr,
    blk_ptr,
    ks,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    CHUNK: tl.constexpr,
    CAP: tl.constexpr,
    GATHERED: tl.constexpr,
):
    """Top-`K` valid columns of `row_ptr[:n_cols]`, column `pin` (if in range) counting as the largest.

    `tau` is a score no top-`K` value is expected below: only values `>= tau` are compacted into
    `cval_ptr`/`cidx_ptr` (capacity `n_cols`) and selected from. Should fewer than `K` (or all the
    valid ones) reach it, every valid value is. Returns the key of the `K`-th largest value (0 if
    fewer) and the largest unpinned value.
    """
    n_kept, n_valid, row_max = _compact(row_ptr, n_cols, pin, tau, cval_ptr, cidx_ptr, CHUNK)
    if n_kept < tl.minimum(n_valid, K):
        n_kept, n_valid, row_max = _compact(row_ptr, n_cols, pin, float("-inf"), cval_ptr, cidx_ptr, CHUNK)
    tl.debug_barrier()
    bound = _select_compacted(cval_ptr, cidx_ptr, n_kept, out_ptr, blk_ptr, ks, K, GROUP, CHUNK, CAP, GATHERED)
    return bound, row_max


@triton.jit
def _rescore_threshold(bound, row_max):
    """Score a rescoring pass may keep entries above: the `K`-th best group max `bound` (key, 0 if
    there is none), lowered by a margin for rounding differences between the two score kernels."""
    tau = _key_to_float(bound)
    return tl.where(bound == 0, float("-inf"), tau - 1e-3 * tl.maximum(tl.abs(tau), tl.abs(row_max)))


@triton.jit
def _score_tile(q, w, ks, ke, blk_row, cols, n_cols, K_BYTES, K_SCALE, stride_ks, offs_d, GROUP: tl.constexpr):
    """Scores of the entries `cols` of a gathered row, -inf where invalid."""
    blk = tl.load(blk_row + cols // GROUP, mask=cols < n_cols, other=-1)
    entry = ks + blk * GROUP + cols % GROUP
    valid = (blk >= 0) & (entry < ke)
    entry_safe = tl.where(valid, entry, 0).to(tl.int64)
    # Masked rather than clamped: unused picks would all read one key row.
    k_tile = tl.load(K_BYTES + entry_safe[:, None] * stride_ks + offs_d[None, :], mask=valid[:, None], other=0)
    s = tl.dot(k_tile.to(tl.float8e4nv, bitcast=True), tl.trans(q))
    score = tl.sum(tl.maximum(s, 0.0) * w[None, :], axis=1)
    score *= tl.load(K_SCALE + entry_safe, mask=valid, other=0.0)
    return tl.where(valid, score, float("-inf"))


@triton.jit
def _group_max_kernel(
    Q,
    K,
    K_SCALE,
    W,
    KS,
    KE,
    GM,
    SGM,
    S_Q,
    S_K,
    N_GROUPS,
    N_SUPER,
    stride_qs,
    stride_qh,
    stride_ks,
    stride_ws,
    stride_gm,
    stride_sgm,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """`GM[t, g]` = max score over doc-local entries `[g * GROUP, (g + 1) * GROUP)` of query `t`, and
    `SGM[t, s]` = max of `GM[t, s * SUPER : (s + 1) * SUPER]`.

    Program `(m, n)` covers queries `[m * BLOCK_M, ...)` and doc-local entries `[n * BLOCK_N, ...)`.
    Queries of different documents read different entries, so the tile is computed once per
    document among the block's queries (one pass unless the block straddles a document start).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S_Q
    offs_m_safe = tl.minimum(offs_m, S_Q - 1).to(tl.int64)
    offs_d = tl.arange(0, D)
    G_PER_TILE: tl.constexpr = BLOCK_N // GROUP
    S_PER_TILE: tl.constexpr = G_PER_TILE // SUPER

    ks = tl.load(KS + offs_m, mask=mask_m, other=0)
    n_valid = tl.load(KE + offs_m, mask=mask_m, other=0) - ks
    local_lo = pid_n * BLOCK_N
    offs_g = pid_n * G_PER_TILE + tl.arange(0, G_PER_TILE)
    offs_s = pid_n * S_PER_TILE + tl.arange(0, S_PER_TILE)
    gm_ptrs = GM + offs_m[:, None].to(tl.int64) * stride_gm + offs_g[None, :]
    sgm_ptrs = SGM + offs_m[:, None].to(tl.int64) * stride_sgm + offs_s[None, :]
    gm_mask = mask_m[:, None] & (offs_g[None, :] < N_GROUPS)
    sgm_mask = mask_m[:, None] & (offs_s[None, :] < N_SUPER)

    needed = mask_m & (local_lo < n_valid)
    tl.store(gm_ptrs, tl.full([BLOCK_M, G_PER_TILE], float("-inf"), tl.float32), mask=gm_mask & ~needed[:, None])
    tl.store(sgm_ptrs, tl.full([BLOCK_M, S_PER_TILE], float("-inf"), tl.float32), mask=sgm_mask & ~needed[:, None])
    done = ~needed
    local_cols = local_lo + tl.arange(0, BLOCK_N)
    while tl.min(done.to(tl.int32)) == 0:
        doc_start = tl.min(tl.where(done, 2147483647, ks))
        active = ~done & (ks == doc_start)
        offs_n = tl.minimum(doc_start + local_cols, S_K - 1).to(tl.int64)
        k_tile = tl.load(K + offs_n[:, None] * stride_ks + offs_d[None, :])
        k_scale = tl.load(K_SCALE + offs_n)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for h in range(H):
            q_h = tl.load(Q + offs_m_safe[:, None] * stride_qs + h * stride_qh + offs_d[None, :])
            w_h = tl.load(W + offs_m_safe * stride_ws + h)
            s = tl.dot(q_h, tl.trans(k_tile))
            acc += tl.maximum(s, 0.0) * w_h[:, None]
        acc = acc * k_scale[None, :]
        acc = tl.where(local_cols[None, :] < n_valid[:, None], acc, float("-inf"))
        group_max = tl.max(tl.reshape(acc, [BLOCK_M, G_PER_TILE, GROUP]), axis=2)
        super_max = tl.max(tl.reshape(group_max, [BLOCK_M, S_PER_TILE, SUPER]), axis=2)
        tl.store(gm_ptrs, group_max, mask=gm_mask & active[:, None])
        tl.store(sgm_ptrs, super_max, mask=sgm_mask & active[:, None])
        done = done | active


@triton.jit
def _group_topk_kernel(
    GM,
    SGM,
    KS,
    KE,
    SCRATCH_V,
    SCRATCH_I,
    OUT,
    TAU,
    S_Q,
    N_GROUPS,
    N_SUPER,
    stride_gm,
    stride_sgm,
    stride_out,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    PIN_NEWEST: tl.constexpr,
    WRITE_TAU: tl.constexpr,
    CHUNK: tl.constexpr,
    CAP: tl.constexpr,
):
    """`OUT[t]` = doc-local indices of query `t`'s `K` best groups, -1 for unused picks.

    With `PIN_NEWEST` the group holding the query's newest entry always counts as the best. With
    `WRITE_TAU`, `TAU[t]` is the score below which no entry of those groups can be in the top `K`.
    Persistent, with one `N_GROUPS`-wide compaction row of scratch per program.
    """
    pid = tl.program_id(0)
    cval = SCRATCH_V + pid.to(tl.int64) * N_GROUPS
    cidx = SCRATCH_I + pid.to(tl.int64) * N_GROUPS
    for row in range(pid, S_Q, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        row_ptr = GM + row64 * stride_gm
        pin = -1
        if PIN_NEWEST:
            pin = tl.maximum(tl.load(KE + row64) - tl.load(KS + row64) - 1, 0) // GROUP
        tau = _key_to_float(_block_threshold(SGM + row64 * stride_sgm, N_SUPER, K, CHUNK, CAP))
        bound, row_max = _topk_row(
            row_ptr,
            N_GROUPS,
            pin,
            tau,
            cval,
            cidx,
            OUT + row64 * stride_out,
            GM,
            0,
            K,
            GROUP,
            CHUNK,
            CAP,
            False,
        )
        if WRITE_TAU:
            tl.store(TAU + row64, _rescore_threshold(bound, row_max))
        tl.debug_barrier()


@triton.jit
def _gather_topk_kernel(
    Q,
    K,
    K_SCALE,
    W,
    KS,
    KE,
    BLOCKS,
    TAU,
    SCRATCH_V,
    SCRATCH_I,
    OUT,
    S_Q,
    S_K,
    N_BLOCKS,
    stride_qs,
    stride_qh,
    stride_ks,
    stride_ws,
    stride_blk,
    stride_out,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
    CHUNK: tl.constexpr,
    CAP: tl.constexpr,
    KNOWN_TAU: tl.constexpr,
    SCORE_STAGES: tl.constexpr = 1,
):
    """`OUT[t]` = the `TOPK` best entries among the doc-local groups `BLOCKS[t]` (-1 unused).

    Persistent: program `p` handles queries `p, p + grid, ...` and stages one query's scores in its
    own scratch rows, which stay in L2. With `KNOWN_TAU` (`BLOCKS` are `_group_topk_kernel`'s picks)
    `TAU[t]` bounds the top scores from below; otherwise the bound is the `TOPK`-th best group max.
    """
    pid = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, TILE)
    G_PER_TILE: tl.constexpr = TILE // GROUP
    offs_tg = tl.arange(0, G_PER_TILE)
    n_cols = N_BLOCKS * GROUP
    K_BYTES = K.to(tl.pointer_type(tl.uint8))
    scores = SCRATCH_V + pid.to(tl.int64) * (2 * n_cols + N_BLOCKS)
    cval = scores + n_cols
    bmax = cval + n_cols
    cidx = SCRATCH_I + pid.to(tl.int64) * n_cols
    for row in range(pid, S_Q, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        q = tl.load(Q + row64 * stride_qs + offs_h[:, None] * stride_qh + offs_d[None, :])
        w = tl.load(W + row64 * stride_ws + offs_h)
        ks = tl.load(KS + row64)
        ke = tl.load(KE + row64)
        blk_row = BLOCKS + row64 * stride_blk
        for c0 in tl.range(0, n_cols, TILE, num_stages=SCORE_STAGES):
            cols = c0 + offs_t
            score = _score_tile(q, w, ks, ke, blk_row, cols, n_cols, K_BYTES, K_SCALE, stride_ks, offs_d, GROUP)
            tl.store(scores + cols, score, mask=cols < n_cols)
            if not KNOWN_TAU:
                g = c0 // GROUP + offs_tg
                tl.store(bmax + g, tl.max(tl.reshape(score, [G_PER_TILE, GROUP]), axis=1), mask=g < N_BLOCKS)
        tl.debug_barrier()
        if KNOWN_TAU:
            tau = tl.load(TAU + row64)
        else:
            tau = _key_to_float(_block_threshold(bmax, N_BLOCKS, TOPK, CHUNK, CAP))
        _topk_row(
            scores, n_cols, -1, tau, cval, cidx, OUT + row64 * stride_out, blk_row, ks, TOPK, GROUP, CHUNK, CAP, True
        )
        tl.debug_barrier()
