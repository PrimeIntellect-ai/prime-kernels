import pytest
import torch

import prime_kernels

REASON = prime_kernels.unavailable_reason("dsa_indexer_topk")
pytestmark = pytest.mark.skipif(REASON is not None, reason=REASON or "")

FP8_MAX = 448.0


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row FP8 e4m3 with power-of-two scales, dequantized back to fp64."""
    x = x.double()
    scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-10) / FP8_MAX)))
    x_q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_q.double(), scale.squeeze(-1)


def reference_scores(q, k, w, ks, ke) -> torch.Tensor:
    """`(S_q, S_k)` fp64 indexer scores of the FP8 operands, -inf outside `[ks, ke)`."""
    q_q, q_scale = quantize(q)
    k_q, k_scale = quantize(k)
    w = w.double() * q_scale
    scores = torch.empty(q.shape[0], k.shape[0], dtype=torch.float64, device=q.device)
    for r0 in range(0, q.shape[0], 256):
        logits = torch.einsum("thd,ed->the", q_q[r0 : r0 + 256], k_q).relu()
        scores[r0 : r0 + 256] = torch.einsum("the,th->te", logits, w[r0 : r0 + 256]) * k_scale
    cols = torch.arange(k.shape[0], device=q.device)
    valid = (cols >= ks[:, None]) & (cols < ke[:, None])
    return scores.masked_fill(~valid, float("-inf"))


def doc_local_blocks(scores, ks, width, block_size) -> torch.Tensor:
    """`(S_q, n_blocks)` best score of every doc-local block, -inf where it holds no readable entry."""
    n_blocks = -(-width // block_size)
    cols = ks[:, None].long() + torch.arange(n_blocks * block_size, device=scores.device)
    local = scores.gather(1, cols.clamp_max(scores.shape[1] - 1))
    local = local.masked_fill(cols >= scores.shape[1], float("-inf"))
    return local.unflatten(1, (n_blocks, block_size)).amax(-1)


def check_picks(picks: torch.Tensor, allowed: torch.Tensor, k: int) -> None:
    """`picks` are `k` best columns of `allowed` (fp64 scores, -inf excluded) up to ties."""
    n_allowed = (allowed > float("-inf")).sum(-1)
    n_taken = (picks >= 0).sum(-1)
    torch.testing.assert_close(n_taken, n_allowed.clamp_max(k))
    # -1 padding is at the tail.
    assert bool(((picks >= 0).long().diff(dim=-1) <= 0).all())
    safe = picks.clamp_min(0)
    picked = allowed.gather(1, safe).masked_fill(picks < 0, float("inf"))
    assert bool((picked > float("-inf")).all()), "a pick is outside the allowed entries"
    sorted_picks = safe.masked_fill(picks < 0, -1).sort(-1).values
    dup = (sorted_picks.diff(dim=-1) == 0) & (sorted_picks[:, 1:] >= 0)
    assert not bool(dup.any()), "duplicate picks"
    kth = allowed.topk(min(k, allowed.shape[1]), dim=-1).values[:, -1:]
    # Hopper FP8 tensor cores accumulate with a reduced-precision mantissa (prime-rl's Triton
    # indexer too), so near-ties can swap.
    tol = 2e-3 * allowed.masked_fill(allowed == float("-inf"), 0).abs().amax(-1, keepdim=True) + 1e-12
    worst = picked.amin(-1, keepdim=True)
    has = n_taken[:, None] > 0
    assert bool(((worst >= kth - tol) | ~has).all()), "a pick scores below the k-th best"


def packed_bounds(doc_lens, ratio, device, n_queries=None, query_offset=0):
    """`ks, ke` in entry coordinates for queries of a packed row of documents (all queries by default)."""
    ks, ke = [], []
    first_entry = 0
    for n in doc_lens:
        pos = torch.arange(n)
        ks.append(torch.full((n,), first_entry))
        ke.append(first_entry + (pos + 1) // ratio)
        first_entry += n // ratio
    ks, ke = torch.cat(ks), torch.cat(ke)
    n_queries = ks.shape[0] - query_offset if n_queries is None else n_queries
    sl = slice(query_offset, query_offset + n_queries)
    return ks[sl].int().to(device), ke[sl].int().to(device), first_entry


def make_inputs(n_queries, n_entries, heads, dim, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(n_queries, heads, dim, generator=gen).to(device, torch.bfloat16)
    k = torch.randn(n_entries, dim, generator=gen).to(device, torch.bfloat16)
    w = torch.randn(n_queries, heads, generator=gen).to(device, torch.bfloat16)
    return q, k, w


@pytest.fixture(scope="module")
def kernel():
    return prime_kernels.load("dsa_indexer_topk")


@pytest.mark.parametrize(
    "doc_lens,ratio,heads,dim,topk,block_size,topk_blocks",
    [
        ([700], 1, 16, 64, 64, 8, 32),
        ([300, 5, 517, 1, 233], 1, 32, 128, 64, 8, 24),
        ([1000, 37, 600], 2, 32, 128, 128, 8, 40),
        ([129, 400, 3], 2, 16, 128, 16, 4, 8),
    ],
)
def test_matches_reference(kernel, doc_lens, ratio, heads, dim, topk, block_size, topk_blocks):
    device = torch.device("cuda")
    ks, ke, n_entries = packed_bounds(doc_lens, ratio, device)
    width = max(n // ratio for n in doc_lens)
    q, k, w = make_inputs(ks.shape[0], n_entries, heads, dim, device)
    scores = reference_scores(q, k, w, ks, ke)

    picks, cand = kernel.dsv41_index_topk(q, k, w, ks, ke, topk, None, True, width, block_size, topk_blocks)
    check_picks(picks, scores, topk)
    picks_plain, empty = kernel.dsv41_index_topk(q, k, w, ks, ke, topk, None, False, width, block_size, topk_blocks)
    assert empty.numel() == 0
    check_picks(picks_plain, scores, topk)

    # Candidate blocks: the newest block is always kept, the rest are the best by block max.
    blocks = doc_local_blocks(scores, ks, width, block_size)
    newest = ((ke - ks - 1).clamp_min(0) // block_size).long()
    assert bool((cand == newest[:, None]).any(-1).all()), "the newest block is not pinned"
    pinned = blocks.scatter(1, newest[:, None], float("inf"))
    check_picks(cand.long(), pinned, topk_blocks)

    # Consumer layer: rank only entries inside the given blocks.
    picks_c, empty = kernel.dsv41_index_topk(q, k, w, ks, ke, topk, cand, False, width, block_size, topk_blocks)
    assert empty.numel() == 0
    allowed = torch.full_like(scores, float("-inf"))
    local = cand.long()[:, :, None] * block_size + torch.arange(block_size, device=device)
    entries = (ks[:, None, None].long() + local).flatten(1)
    ok = (cand.long()[:, :, None] >= 0).expand_as(local).flatten(1) & (entries < ke[:, None])
    rows = torch.arange(ks.shape[0], device=device)[:, None].expand_as(entries)
    allowed[rows[ok], entries[ok]] = scores[rows[ok], entries[ok]]
    check_picks(picks_c, allowed, topk)


def test_v41_shapes_with_binding_candidates(kernel):
    """V4.1 Flash indexer shapes where the candidate filter binds (more than 2048 blocks of 8)."""
    device = torch.device("cuda")
    n_entries, n_queries = 20480, 512
    ks, ke, _ = packed_bounds([n_entries], 1, device, n_queries=n_queries, query_offset=n_entries - n_queries)
    q, k, w = make_inputs(n_queries, n_entries, 32, 128, device, seed=1)
    scores = reference_scores(q, k, w, ks, ke)

    picks, cand = kernel.dsv41_index_topk(q, k, w, ks, ke, 512, None, True, n_entries, 8, 2048)
    check_picks(picks, scores, 512)
    picks_plain, _ = kernel.dsv41_index_topk(q, k, w, ks, ke, 512, None, False, n_entries, 8, 2048)
    check_picks(picks_plain, scores, 512)
    blocks = doc_local_blocks(scores, ks, n_entries, 8)
    newest = ((ke - ks - 1).clamp_min(0) // 8).long()
    check_picks(cand.long(), blocks.scatter(1, newest[:, None], float("inf")), 2048)

    picks_c, _ = kernel.dsv41_index_topk(q, k, w, ks, ke, 512, cand, False, n_entries, 8, 2048)
    allowed = torch.full_like(scores, float("-inf"))
    entries = (cand.long()[:, :, None] * 8 + torch.arange(8, device=device)).flatten(1)
    ok = (cand.long()[:, :, None] >= 0).expand(-1, -1, 8).flatten(1) & (entries < ke[:, None])
    rows = torch.arange(n_queries, device=device)[:, None].expand_as(entries)
    allowed[rows[ok], entries[ok]] = scores[rows[ok], entries[ok]]
    check_picks(picks_c, allowed, 512)


def test_rescoring_falls_back_when_the_threshold_overshoots(kernel):
    """The rescoring pass keeps everything should its threshold leave fewer than `topk` entries."""
    from prime_kernels.dsa_indexer_topk import indexer
    from prime_kernels.dsa_indexer_topk.quant import per_token_quant_fp8

    device = torch.device("cuda")
    ks, ke, n_entries = packed_bounds([900, 300], 1, device)
    q, k, w = make_inputs(ks.shape[0], n_entries, 16, 64, device, seed=2)
    scores = reference_scores(q, k, w, ks, ke)
    group, topk = 4, 32
    n_groups = -(-900 // group)
    blocks = torch.arange(n_groups, device=device, dtype=torch.int32).expand(ks.shape[0], -1).contiguous()
    q_fp8, q_scale = per_token_quant_fp8(q)
    k_fp8, k_scale = per_token_quant_fp8(k)
    w = (w.float() * q_scale).contiguous()
    picks = torch.empty(ks.shape[0], topk, dtype=torch.int64, device=device)
    tau = torch.full((ks.shape[0],), float("inf"), device=device)
    indexer._gather_topk(q_fp8, k_fp8, k_scale, w, ks, ke, blocks, tau, picks, topk, group)
    check_picks(picks, scores, topk)


def test_no_readable_entries(kernel):
    device = torch.device("cuda")
    q, k, w = make_inputs(4, 8, 16, 64, device)
    ks = torch.tensor([0, 0, 4, 4], dtype=torch.int32, device=device)
    ke = torch.tensor([0, 1, 4, 6], dtype=torch.int32, device=device)
    picks, cand = kernel.dsv41_index_topk(q, k, w, ks, ke, 4, None, True, 4, 2, 3)
    assert picks[0].tolist() == [-1] * 4 and picks[2].tolist() == [-1] * 4
    assert picks[1].tolist() == [0, -1, -1, -1]
    assert sorted(picks[3].tolist()) == [-1, -1, 4, 5]
    # A query without entries keeps block 0, as the newest block is pinned.
    assert cand[0].tolist() == [0, -1, -1] and cand[3].tolist() == [0, -1, -1]


def test_fake_tensor(kernel):
    q = torch.empty(5, 32, 128, dtype=torch.bfloat16, device="meta")
    k = torch.empty(9, 128, dtype=torch.bfloat16, device="meta")
    w = torch.empty(5, 32, dtype=torch.bfloat16, device="meta")
    ks = torch.empty(5, dtype=torch.int32, device="meta")
    picks, cand = kernel.dsv41_index_topk(q, k, w, ks, ks, 512, None, True, 9, 8, 2048)
    assert picks.shape == (5, 512) and picks.dtype == torch.int64
    assert cand.shape == (5, 2048) and cand.dtype == torch.int32
