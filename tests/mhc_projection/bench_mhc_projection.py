"""Benchmark the fused mHC projection against prime-rl's DeepSeek-V4.1 path.

Needs prime-rl importable (run with its venv). Times one sublayer's mHC gates plus the collapse
of its input streams, at V4.1 Flash shapes:

    today          DeepseekV41HyperConnection.forward + collapse_streams, eager
    today+compile  the same under torch.compile (prime-rl compiles the decoder blocks)
    fused          prime_kernels.mhc_projection.hyper_connection, eager and under torch.compile

Usage: python bench_mhc_projection.py [tokens]
"""

import sys
from types import SimpleNamespace

import torch
from triton.testing import do_bench

import prime_kernels
from prime_rl.trainer.models.deepseek_v41.hyperconnections import DeepseekV41HyperConnection, collapse_streams

HIDDEN, HC, ITERS, HC_EPS, RMS_EPS = 5120, 4, 20, 1e-6, 1e-20


def today(module, x, pre_mix):
    pre, post, comb = module(x)
    return pre, post, comb, collapse_streams(x, pre_mix)


def fused(mhc, module, x, pre_mix):
    return mhc.hyper_connection(
        x,
        module.fn,
        module.scale,
        module.base,
        pre_mix,
        rms_eps=RMS_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=ITERS,
    )


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def fp64_reference(module, x, pre_mix):
    hc = HC
    xd = x.double()
    flat = xd.flatten(start_dim=2)
    rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + RMS_EPS)
    mixes = (flat @ module.fn.to(torch.bfloat16).double().t()) * rstd
    scale, base = module.scale.double(), module.base.double()
    pre_m, post_m, comb_m = mixes.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre = torch.sigmoid(pre_m * scale[0] + pre_b) + HC_EPS
    post = 2 * torch.sigmoid(post_m * scale[1] + post_b)
    comb = (comb_m * scale[2] + comb_b).unflatten(-1, (hc, hc)).softmax(-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    collapsed = (pre_mix.double().unsqueeze(-1) * xd).sum(2)
    return pre, post, comb, collapsed


def time_ms(fn, x, pre_mix, params, cotangents, backward):
    """(wall ms from CUDA events, summed GPU kernel ms from the profiler) per step."""

    def step():
        outputs = fn(x, pre_mix)
        if backward:
            torch.autograd.grad(outputs, (x, pre_mix, *params), cotangents)

    wall = do_bench(step, warmup=25, rep=200)
    steps = 10
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
    kernel_us = sum(event.device_time_total for event in prof.key_averages() if event.device_type.name == "CUDA")
    return wall, kernel_us / steps / 1e3


def main():
    tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    mhc = prime_kernels.load("mhc_projection")
    config = SimpleNamespace(
        hc_mult=HC, hc_sinkhorn_iters=ITERS, hc_eps=HC_EPS, rms_norm_eps=RMS_EPS, hidden_size=HIDDEN
    )
    module = DeepseekV41HyperConnection(config).cuda()
    module.init_weights(0.02)
    with torch.no_grad():
        module.base.normal_(0, 0.5)
        module.scale.uniform_(0.5, 1.5)
    params = tuple(module.parameters())

    x = torch.randn(1, tokens, HC, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pre_mix = torch.rand(1, tokens, HC, device="cuda", requires_grad=True)

    paths = {
        "today": lambda x, p: today(module, x, p),
        "today+compile": torch.compile(lambda x, p: today(module, x, p)),
        "fused": lambda x, p: fused(mhc, module, x, p),
        "fused+compile": torch.compile(lambda x, p: fused(mhc, module, x, p)),
    }

    # Accuracy against an fp64 evaluation of the same math (weight rounded to bf16 as both paths do).
    inputs = (x, pre_mix, *params)
    exact = fp64_reference(module, x, pre_mix)
    # Cotangents in the dtypes the paths produce (fp32 gates, bf16 collapse), shared exactly.
    cotangents = [torch.randn_like(t, dtype=dtype) for t, dtype in zip(exact, [torch.float32] * 3 + [x.dtype])]
    exact_all = (*exact, *torch.autograd.grad(exact, inputs, [c.double() for c in cotangents]))
    names = ("pre", "post", "comb", "collapsed", "d_x", "d_pre_mix", "d_fn", "d_base", "d_scale")
    print("relative L2 error vs fp64:")
    print(f"  {'':10s} {'today':>9s} {'fused':>9s}")
    errors = {}
    for name in ("today", "fused"):
        outputs = paths[name](x, pre_mix)
        grads = torch.autograd.grad(outputs, inputs, cotangents)
        errors[name] = [rel_err(a, b) for a, b in zip((*outputs, *grads), exact_all)]
    for i, label in enumerate(names):
        print(f"  {label:10s} {errors['today'][i]:9.2e} {errors['fused'][i]:9.2e}")

    streams_gb = x.numel() * x.element_size() / 1e9
    print(f"\ntokens={tokens}, streams {streams_gb:.3f} GB (bf16)")
    print("wall: CUDA events around one eager step (includes launch gaps); gpu: summed kernel time")
    print(f"{'path':16s} {'fwd wall':>9s} {'fwd gpu':>8s} {'fwd+bwd wall':>13s} {'fwd+bwd gpu':>12s}  (ms)")
    for name, fn in paths.items():
        fwd = time_ms(fn, x, pre_mix, params, cotangents, backward=False)
        both = time_ms(fn, x, pre_mix, params, cotangents, backward=True)
        print(f"{name:16s} {fwd[0]:9.3f} {fwd[1]:8.3f} {both[0]:13.3f} {both[1]:12.3f}")

    # The two fused kernels alone, for their achieved bandwidth.
    flat = x.detach().view(-1, HC, HIDDEN)
    flat_pre = pre_mix.detach().view(-1, HC)
    proj = mhc.projection
    fwd = do_bench(lambda: proj.mhc_projection_forward(flat, module.fn, flat_pre, RMS_EPS), warmup=25, rep=200)
    mixes, rstd, collapsed = proj.mhc_projection_forward(flat, module.fn, flat_pre, RMS_EPS)
    grad_mixes, grad_collapsed = torch.randn_like(mixes), torch.randn_like(collapsed)
    bwd = do_bench(
        lambda: proj.mhc_projection_backward(flat, module.fn, flat_pre, mixes, rstd, grad_mixes, grad_collapsed),
        warmup=25,
        rep=200,
    )
    fwd_bytes = streams_gb * 1.25  # read the streams, write the collapse
    bwd_bytes = streams_gb * 2.25  # read the streams and d(collapse), write d(streams)
    print(f"\nprojection fwd kernel {fwd:.3f} ms ({fwd_bytes / fwd:.2f} TB/s)")
    print(f"projection bwd kernel {bwd:.3f} ms ({bwd_bytes / bwd:.2f} TB/s)")


if __name__ == "__main__":
    main()
