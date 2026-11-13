"""Benchmark `dsa_sparse_attn_bwd` against the backward prime-rl runs today (cuDNN frontend's SM90 DSA
backward, `flash_attn_bwd_sm90`) at DeepSeek-V4.1 Flash shapes on one GPU, with FlashMLA's sparse
prefill forward (prime-rl's forward) for reference.

    python tests/dsa_sparse_attn_bwd/bench_dsa_sparse_attn_bwd.py

Shapes: 16384 queries x 64 heads x 512, kv = 16384 window positions + 8192 compressed entries,
640 slots per query (128-token causal window + top-512 random picks from the document's causal
compressed entries), for one 16k document and for 4 packed documents. TFLOP/s count every slot
(empty ones too), 4 * 64 * 512 * 640 flops per query forward and 10 * 64 * 512 * 640 backward
(S, dP, dQ, dK, dV), like the training trace's accounting.
"""

import math
import sys
from pathlib import Path

import torch

import prime_kernels

sys.path.insert(0, str(Path(__file__).parent))
from test_dsa_sparse_attn_bwd import make_indices, rel_err  # noqa: E402


def bench(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 import flash_attn_bwd_sm90
    from flash_mla import flash_mla_sparse_fwd

    ours = prime_kernels.load("dsa_sparse_attn_bwd").sparse_attn_backward_flat
    torch.manual_seed(0)
    h, d = 64, 512
    print(
        f"{'case':10s} {'empty':>6s} | {'FlashMLA fwd':>17s} | {'cuDNN bwd':>17s} | {'ours bwd':>17s} | speedup | ours vs cuDNN (rel L2: dq, dkv, dsink)"
    )
    for name, lens in [("1 doc", [16384]), ("4 docs", [6000, 3000, 4384, 3000])]:
        idx, n = make_indices(lens)
        t, k = idx.shape
        q = torch.randn(t, h, d, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(n, d, device="cuda", dtype=torch.bfloat16)
        sinks = torch.randn(h, device="cuda")
        grad_out = torch.randn(t, h, d, device="cuda", dtype=torch.bfloat16)
        scale = 1 / math.sqrt(d)

        def fwd():
            return flash_mla_sparse_fwd(q, kv.view(n, 1, d), idx.view(t, 1, k), scale, d, attn_sink=sinks)

        out, _, lse = fwd()

        def theirs():
            return flash_attn_bwd_sm90(
                q, kv, out, grad_out, lse, attn_sink=sinks, softmax_scale=scale, topk_idxs=idx, need_d_sink=True
            )

        def mine():
            return ours(q, kv, out, grad_out, lse, idx, sinks, scale)

        errs = ", ".join(f"{rel_err(a, b):.1e}" for a, b in zip(mine(), theirs()))
        flops = 2 * t * k * h * d
        t_fwd, t_theirs, t_mine = bench(fwd), bench(theirs), bench(mine)
        empty = (idx < 0).float().mean().item()
        print(
            f"{name:10s} {empty:6.1%} | {t_fwd:6.2f} ms {2 * flops / t_fwd / 1e9:4.0f} TF/s | "
            f"{t_theirs:6.2f} ms {5 * flops / t_theirs / 1e9:4.0f} TF/s | {t_mine:6.2f} ms {5 * flops / t_mine / 1e9:4.0f} TF/s | "
            f"{t_theirs / t_mine:6.2f}x | {errs}",
            flush=True,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
