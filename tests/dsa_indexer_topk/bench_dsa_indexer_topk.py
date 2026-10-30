"""Benchmark `dsa_indexer_topk` against prime-rl's `dsv41_index_topk` at DeepSeek-V4.1 Flash shapes.

    python tests/dsa_indexer_topk/bench_dsa_indexer_topk.py [path/to/prime-rl/src]

Shapes: 16k queries of one document, 32 index heads x 128, top-512, against 16k entries
(compress ratio 1) and 8k entries (ratio 2) on one GPU, and the last CP rank's 16k queries of a
131k-token document (CP=8) with the candidate filter active (2048 blocks of 8): plain, candidate
source (emit) and candidate consumer. With the prime-rl source path it also times prime-rl's op
and reports how much the two agree (mean |intersection| / |picks| per query).
"""

import argparse
import sys

import torch

import prime_kernels


def bench(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.long(), b.long()
    a_sorted = a.sort(-1).values
    found = torch.searchsorted(a_sorted, b.contiguous())
    hit = (a_sorted.gather(1, found.clamp_max(a.shape[1] - 1)) == b) & (b >= 0)
    return float(hit.sum() / (b >= 0).sum().clamp_min(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prime_rl_src", nargs="?", help="prime-rl's src/ directory, to compare against its op")
    args = parser.parse_args()
    ours = prime_kernels.load("dsa_indexer_topk").dsv41_index_topk
    theirs = None
    if args.prime_rl_src:
        sys.path.insert(0, args.prime_rl_src)
        from prime_rl.trainer.models.kernels.dsv41_indexer import dsv41_index_topk as theirs

    torch.manual_seed(0)
    device = torch.device("cuda")
    n_queries, heads, dim, topk, block, topk_blocks = 16384, 32, 128, 512, 8, 2048
    cases = [("ratio 1, 16k entries", 16384, 1, 0), ("ratio 2, 8k entries", 8192, 2, 0)]
    cases.append(("CP rank 7/8, 131k entries", 131072, 1, 131072 - n_queries))
    for name, n_entries, ratio, query_offset in cases:
        q = torch.randn(n_queries, heads, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(n_entries, dim, device=device, dtype=torch.bfloat16)
        w = torch.randn(n_queries, heads, device=device, dtype=torch.bfloat16)
        pos = torch.arange(query_offset, query_offset + n_queries, device=device)
        ks = torch.zeros(n_queries, dtype=torch.int32, device=device)
        ke = ((pos + 1) // ratio).int()
        modes = [("plain", None, False)]
        if n_entries > topk_blocks * block:
            _, cand = (theirs or ours)(q, k, w, ks, ke, topk, None, True, n_entries, block, topk_blocks)
            modes += [("emit", None, True), ("consume", cand, False)]
        for mode, cand_in, emit in modes:

            def run(op):
                return op(q, k, w, ks, ke, topk, cand_in, emit, n_entries, block, topk_blocks)

            line = f"{name:28s} {mode:8s} fused {bench(lambda: run(ours)):8.2f} ms"
            if theirs is not None:
                line += f" | prime-rl {bench(lambda: run(theirs)):8.2f} ms"
                (p_ours, c_ours), (p_theirs, c_theirs) = run(ours), run(theirs)
                line += f" | top-k agreement {overlap(p_theirs, p_ours):.5f}"
                if emit:
                    line += f", candidate agreement {overlap(c_theirs, c_ours):.5f}"
            print(line, flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
