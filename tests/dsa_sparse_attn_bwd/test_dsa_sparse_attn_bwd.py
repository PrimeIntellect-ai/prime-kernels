import math

import pytest
import torch

import prime_kernels

REASON = prime_kernels.unavailable_reason("dsa_sparse_attn_bwd")
pytestmark = pytest.mark.skipif(REASON is not None, reason=REASON or "")

HEADS, DIM = 64, 512
# Relative L2 error vs the fp32 reference. The error is dominated by bf16 rounding of the inputs
# the kernel shares with cuDNN (out, grad_out) and of dq / dkv themselves; cuDNN's SM90 backward
# lands on the same numbers (~2.5e-3).
REF_TOL = 1e-2
CUDNN_TOL = 2e-3


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def dsa():
    return prime_kernels.load("dsa_sparse_attn_bwd")


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30)).item()


def make_indices(seq_lens, window=128, topk=512, ratio=2):
    """DeepSeek-V4.1 slot lists for packed documents: a causal sliding window over the token stream
    (rows [0, t)) followed by `topk` random picks from the query's document-local, causal compressed
    entries (rows [t, t + entries)); -1 marks an empty slot. Returns (indices, number of kv rows)."""
    device = "cuda"
    seq_lens = torch.tensor(seq_lens, device=device)
    t = int(seq_lens.sum())
    cu = torch.cat([seq_lens.new_zeros(1), seq_lens.cumsum(0)])
    tok = torch.arange(t, device=device)
    doc = torch.searchsorted(cu[1:], tok, right=True)
    start = cu[doc]
    slots = torch.maximum(start, tok - window + 1)[:, None] + torch.arange(window, device=device)
    win = torch.where(slots <= tok[:, None], slots, -1)
    ent_cu = torch.cat([seq_lens.new_zeros(1), (seq_lens // ratio).cumsum(0)])
    avail = (tok - start + 1) // ratio
    max_avail = max(int(avail.max()), 1)
    scores = torch.rand(t, max_avail, device=device)
    scores = scores.masked_fill(torch.arange(max_avail, device=device) >= avail[:, None], -1.0)
    val, local = scores.topk(min(topk, max_avail), dim=-1)
    picks = torch.where(val >= 0, local + ent_cu[doc][:, None] + t, -1)
    picks = torch.nn.functional.pad(picks, (0, topk - picks.shape[1]), value=-1)
    return torch.cat([win, picks], dim=-1).int().contiguous(), t + int(ent_cu[-1])


def reference(q, kv, indices, sinks, scale, grad_out, chunk=256):
    """fp32 sink-aware sparse attention forward (out, sink-free lse) and backward (dq, dkv, dsinks)."""
    t, h, d = q.shape
    out = torch.empty(t, h, d, dtype=torch.float32, device=q.device)
    lse = torch.empty(t, h, dtype=torch.float32, device=q.device)
    dq = torch.empty_like(out)
    dkv = torch.zeros(kv.shape[0], d, dtype=torch.float32, device=q.device)
    dsinks = torch.zeros(h, dtype=torch.float32, device=q.device)
    kv32 = kv.float()
    for s in range(0, t, chunk):
        e = min(t, s + chunk)
        qi, gi, ii = q[s:e].float(), grad_out[s:e].float(), indices[s:e].long()
        valid = ii >= 0
        k = kv32[ii.clamp(min=0)]
        logits = torch.einsum("thd,tkd->thk", qi, k) * scale
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        lse[s:e] = torch.logsumexp(logits, -1)
        lse_sink = torch.logaddexp(lse[s:e], sinks[None])
        p = torch.exp(logits - lse_sink[..., None])
        o = torch.einsum("thk,tkd->thd", p, k)
        out[s:e] = o
        delta = (o * gi).sum(-1)
        ds = p * (torch.einsum("thd,tkd->thk", gi, k) - delta[..., None])
        dq[s:e] = torch.einsum("thk,tkd->thd", ds, k) * scale
        dk = torch.einsum("thk,thd->tkd", ds, qi) * scale + torch.einsum("thk,thd->tkd", p, gi)
        dkv.index_add_(0, ii.clamp(min=0).flatten(), (dk * valid[..., None]).flatten(0, 1))
        dsinks -= (torch.exp(sinks[None] - lse_sink) * delta).sum(0)
    return out, lse, dq, dkv, dsinks


def make_problem(indices, n_rows):
    t = indices.shape[0]
    q = torch.randn(t, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(n_rows, DIM, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(t, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(HEADS, device="cuda")
    scale = 1 / math.sqrt(DIM)
    out, lse, dq, dkv, dsinks = reference(q, kv, indices, sinks, scale, grad_out)
    inputs = (q, kv, out.to(torch.bfloat16), grad_out, lse, indices, sinks, scale)
    return inputs, (dq, dkv, dsinks)


def shuffled(indices, frac_empty=0.0):
    """Permute each query's slots and blank a fraction of them (the result must not depend on slot order)."""
    perm = torch.rand(indices.shape, device=indices.device).argsort(-1)
    out = indices.gather(1, perm)
    if frac_empty:
        out = torch.where(torch.rand(out.shape, device=out.device) < frac_empty, -1, out)
    return out.int().contiguous()


CASES = {
    "2 docs, V4.1 slots": lambda: make_indices([300, 212]),
    "ragged slot count (k = 56, padded)": lambda: make_indices([5, 70, 1], window=16, topk=40),
    "shuffled slots, 30% empty": lambda: (lambda i, n: (shuffled(i, 0.3), n))(*make_indices([257, 400])),
    "4k tokens, 3 docs": lambda: make_indices([2000, 1500, 596]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_matches_reference(dsa, case):
    indices, n_rows = CASES[case]()
    inputs, expected = make_problem(indices, n_rows)
    actual = dsa.sparse_attn_backward_flat(*inputs)
    for name, a, e in zip(("dq", "dkv", "dsinks"), actual, expected):
        assert a.isfinite().all(), name
        assert rel_err(a, e) < REF_TOL, f"{name}: {rel_err(a, e):.2e}"


def test_empty_queries_and_duplicate_slots(dsa):
    indices, n_rows = make_indices([90, 60], window=32, topk=64)
    indices[::7] = -1  # queries with nothing to attend to
    indices[3, 5:9] = indices[3, 4]  # the same row in several slots of a query
    inputs, expected = make_problem(indices.contiguous(), n_rows)
    actual = dsa.sparse_attn_backward_flat(*inputs)
    assert (actual[0][::7] == 0).all()
    for name, a, e in zip(("dq", "dkv", "dsinks"), actual, expected):
        assert rel_err(a, e) < REF_TOL, f"{name}: {rel_err(a, e):.2e}"


@pytest.mark.parametrize("seq_lens", [[16384], [6000, 3000, 4384, 3000]], ids=["1 doc", "4 docs"])
def test_v41_shapes(dsa, seq_lens):
    """16k queries, 640 slots, 16k window positions + 8k compressed entries; also against cuDNN's SM90
    backward (prime-rl's current path) when it is installed."""
    indices, n_rows = make_indices(seq_lens)
    inputs, expected = make_problem(indices, n_rows)
    actual = dsa.sparse_attn_backward_flat(*inputs)
    for name, a, e in zip(("dq", "dkv", "dsinks"), actual, expected):
        assert rel_err(a, e) < REF_TOL, f"{name}: {rel_err(a, e):.2e}"
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 import flash_attn_bwd_sm90
    except ImportError:
        return
    q, kv, out, grad_out, lse, idx, sinks, scale = inputs
    theirs = flash_attn_bwd_sm90(
        q, kv, out, grad_out, lse, attn_sink=sinks, softmax_scale=scale, topk_idxs=idx, need_d_sink=True
    )
    for name, a, c in zip(("dq", "dkv", "dsinks"), actual, theirs):
        assert rel_err(a, c) < CUDNN_TOL, f"{name} vs cuDNN: {rel_err(a, c):.2e}"


def test_custom_op(dsa):
    """The torch op takes prime-rl's batched layout and has a fake impl for torch.compile."""
    indices, n_rows = make_indices([100, 28], window=16, topk=48)
    (q, kv, out, grad_out, lse, idx, sinks, scale), _ = make_problem(indices, n_rows)
    t = q.shape[0]
    args = (grad_out[None], q[None], kv[None, :, None], out[None], lse[None], idx[None, :, None], sinks, scale)
    dq, dkv, dsinks = dsa.dsa_sparse_attn_backward(*args)
    flat = dsa.sparse_attn_backward_flat(q, kv, out, grad_out, lse, idx, sinks, scale)
    assert dq.shape == (1, t, HEADS, DIM) and dkv.shape == (1, n_rows, 1, DIM) and dsinks.shape == (HEADS,)
    torch.testing.assert_close(dq[0], flat[0], rtol=0, atol=0)
    torch.testing.assert_close(dkv[0, :, 0], flat[1], rtol=1e-2, atol=1e-2)  # atomics: summation order varies

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
        fdq, fdkv, fds = torch.ops.prime_kernels.dsa_sparse_attn_bwd(*fake)
    assert fdq.shape == dq.shape and fdkv.shape == dkv.shape and fds.shape == dsinks.shape
