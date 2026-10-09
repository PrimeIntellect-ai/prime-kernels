"""Benchmark `moe_experts` against prime-rl's current expert compute at V4.1 Flash shapes.

prime-rl today (`GroupedGemmExpertCompute` with `BF16ExpertCompute` and V4's `ClampedSwiglu`):
gate and up as two `torch._grouped_mm` calls (or one over the packed `gate_up_proj`), the clamped
SwiGLU in eager PyTorch (or under torch.compile when the decoder block is compiled), and the down
projection as a third `torch._grouped_mm`; autograd does the backward. Its FP8 path
(`DeepGemmFP8ExpertCompute`) is the same with prime-rl's `grouped_fp8_gemm` (DeepGEMM with
standalone quantization kernels) for each GEMM; it is benchmarked when prime-rl is importable
(put its `src` on PYTHONPATH).

Per GPU, 16k tokens routed top-6 give 98304 expert rows: 4 local experts of ~24.5k rows each at
EP=96 and 48 local experts of ~2k rows each at EP=8. Group sizes get +-20% imbalance and are
padded to the kernel's TOKEN_GROUP_ALIGNMENT.

    python tests/moe_experts/bench_moe_experts.py [--breakdown] [--accuracy]
"""

import argparse
import statistics

import torch
import torch.nn.functional as F

import prime_kernels

HIDDEN = 5120
INTERMEDIATE = 2304
LIMIT = 10.0


def clamped_swiglu(gate, up):
    return F.silu(gate.clamp(max=LIMIT)) * up.clamp(min=-LIMIT, max=LIMIT)


compiled_swiglu = torch.compile(clamped_swiglu)


def grouped_mm(x, weight, offsets):
    return torch._grouped_mm(x, weight, offs=offsets)


def today(x, gate_proj, up_proj, down_proj, offsets, activation, gemm=grouped_mm):
    if up_proj is None:
        gate, up = gemm(x, gate_proj.transpose(-2, -1), offsets).chunk(2, dim=-1)
    else:
        gate = gemm(x, gate_proj.transpose(-2, -1), offsets)
        up = gemm(x, up_proj.transpose(-2, -1), offsets)
    return gemm(activation(gate, up), down_proj.transpose(-2, -1), offsets)


def prime_rl_fp8_gemm():
    """prime-rl's `grouped_fp8_gemm`, or None when prime-rl is not importable."""
    try:
        from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm
    except ImportError:
        return None
    return grouped_fp8_gemm


def time_ms(fn, iters):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench(candidates, iters, repeats):
    """Median time per candidate, the candidates interleaved so clock drift hits all alike."""
    for fn in candidates.values():
        fn()
        fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in candidates}
    for _ in range(repeats):
        for name, fn in candidates.items():
            samples[name].append(time_ms(fn, iters))
    return {name: statistics.median(times) for name, times in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--per-op", action="store_true", help="also time each fused GEMM against its unfused ops")
    parser.add_argument("--breakdown", action="store_true", help="also list the kernels of one forward + backward")
    parser.add_argument("--accuracy", action="store_true", help="also report errors against an fp32 reference")
    parser.add_argument("--no-bf16-today", action="store_true", help="skip prime-rl's bf16 variants")
    args = parser.parse_args()
    fp8_gemm = prime_rl_fp8_gemm()
    if fp8_gemm is None:
        print("prime-rl is not importable: skipping its FP8 path")

    moe = prime_kernels.load("moe_experts")
    torch.manual_seed(0)
    rows = 16384 * 6
    print(
        f"{'load':8s} {'variant':26s} {'fwd ms':>8s} {'TFLOP/s':>8s} {'fwd+bwd ms':>11s} {'TFLOP/s':>8s} "
        f"{'saved GiB':>10s}"
    )
    for label, num_experts in (("EP=96", 4), ("EP=8", 48)):
        mean = rows / num_experts
        counts = torch.randint(int(mean * 0.8), int(mean * 1.2) + 1, (num_experts,))
        align = moe.TOKEN_GROUP_ALIGNMENT
        counts = ((counts * rows / counts.sum()).long() + align - 1) // align * align
        counts = counts.cuda()
        num_rows = int(counts.sum())
        offsets = torch.cumsum(counts, 0, dtype=torch.int32)
        flops = 2 * num_rows * HIDDEN * INTERMEDIATE * 3

        x = torch.randn(num_rows, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        scale = 7.0 / HIDDEN**0.5
        gate_up = (torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN, device="cuda") * scale).bfloat16()
        gate_up.requires_grad_()
        gate_proj = gate_up[:, :INTERMEDIATE].detach().clone().requires_grad_()
        up_proj = gate_up[:, INTERMEDIATE:].detach().clone().requires_grad_()
        down_proj = (torch.randn(num_experts, HIDDEN, INTERMEDIATE, device="cuda") / INTERMEDIATE**0.5).bfloat16()
        down_proj.requires_grad_()
        dout = torch.randn(num_rows, HIDDEN, device="cuda", dtype=torch.bfloat16)

        separate = [x, gate_proj, up_proj, down_proj]
        packed = [x, gate_up, down_proj]
        # name -> (forward, the tensors it is differentiated against)
        variants = {}
        if not args.no_bf16_today:
            variants["today (eager act)"] = (
                lambda: today(x, gate_proj, up_proj, down_proj, offsets, clamped_swiglu),
                separate,
            )
            variants["today (compiled act)"] = (
                lambda: today(x, gate_proj, up_proj, down_proj, offsets, compiled_swiglu),
                separate,
            )
            variants["today packed gate_up"] = (
                lambda: today(x, gate_up, None, down_proj, offsets, compiled_swiglu),
                packed,
            )
        variants["moe_experts"] = (lambda: moe.moe_experts(x, gate_proj, up_proj, down_proj, counts, LIMIT), separate)
        variants["moe_experts packed"] = (lambda: moe.moe_experts(x, gate_up, None, down_proj, counts, LIMIT), packed)
        if fp8_gemm is not None:
            variants["today fp8 (DeepGEMM)"] = (
                lambda: today(x, gate_proj, up_proj, down_proj, offsets, compiled_swiglu, fp8_gemm),
                separate,
            )
            variants["today fp8 packed"] = (
                lambda: today(x, gate_up, None, down_proj, offsets, compiled_swiglu, fp8_gemm),
                packed,
            )
        variants["moe_experts fp8"] = (
            lambda: moe.moe_experts(x, gate_proj, up_proj, down_proj, counts, LIMIT, fp8=True),
            separate,
        )
        variants["moe_experts fp8 packed"] = (
            lambda: moe.moe_experts(x, gate_up, None, down_proj, counts, LIMIT, fp8=True),
            packed,
        )

        def forward(fn):
            def run():
                with torch.no_grad():
                    fn()

            return run

        def forward_backward(fn, inputs):
            def run():
                # autograd.grad rather than backward(): no accumulation into .grad in the timing.
                torch.autograd.grad(fn(), inputs, dout)

            return run

        saved = {}
        for name, (fn, _) in variants.items():
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            out = fn()
            saved[name] = (torch.cuda.memory_allocated() - before - out.numel() * out.element_size()) / 2**30
            del out

        fwd = bench({name: forward(fn) for name, (fn, _) in variants.items()}, args.iters, args.repeats)
        fwd_bwd = bench(
            {name: forward_backward(fn, inputs) for name, (fn, inputs) in variants.items()}, args.iters, args.repeats
        )
        for name in variants:
            print(
                f"{label:8s} {name:26s} {fwd[name]:8.3f} {flops / fwd[name] / 1e9:8.0f} "
                f"{fwd_bwd[name]:11.3f} {3 * flops / fwd_bwd[name] / 1e9:8.0f} {saved[name]:10.2f}"
            )
        if args.breakdown:
            for name in ("moe_experts", "today fp8 (DeepGEMM)", "moe_experts fp8"):
                if name in variants:
                    breakdown(label, name, forward_backward(*variants[name]))
        if args.accuracy:
            accuracy(
                label, {name: variants[name] for name in variants if "fp8" in name}, inputs=separate, counts=counts
            )
        if args.per_op:
            per_op(moe, x, gate_proj, up_proj, down_proj, counts, offsets, dout, args)


def breakdown(label, name, fn):
    """GPU time of each kernel of one forward + backward, the longest first."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    iters = 3
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    events.sort(key=lambda e: e.self_device_time_total, reverse=True)
    total = sum(e.self_device_time_total for e in events) / iters / 1e3
    print(f"\n{label} {name}: kernels of one forward + backward ({total:.2f} ms of GPU time)")
    for e in events:
        ms = e.self_device_time_total / iters / 1e3
        if ms < 0.01:
            continue
        print(f"    {ms:8.3f} ms  x{e.count // iters:<3d} {e.key[:90]}")


def accuracy(label, variants, inputs, counts):
    """Relative errors of each variant's output and gradients against an fp32 reference."""
    x, gate_proj, up_proj, down_proj = inputs
    dout = torch.randn_like(x)
    num_rows = int(counts.sum())

    def reference():
        out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        start = 0
        for expert, count in enumerate(counts.tolist()):
            rows = slice(start, start + count)
            xe = x[rows].float()
            h = clamped_swiglu(xe @ gate_proj[expert].float().T, xe @ up_proj[expert].float().T)
            out[rows] = h @ down_proj[expert].float().T
            start += count
        return out

    def errors(out):
        grads = torch.autograd.grad(out, inputs, dout.to(out.dtype))
        # Only the routed rows: prime-rl's FP8 path leaves the rest of dx uninitialized.
        tensors = [out[:num_rows], grads[0][:num_rows], *grads[1:]]
        return [t.float() for t in tensors]

    expected = errors(reference())
    for name, (fn, variant_inputs) in variants.items():
        if variant_inputs is not inputs:
            continue
        actual = errors(fn())
        rel = [((a - e).norm() / e.norm()).item() for a, e in zip(actual, expected)]
        names = ("out", "dx", "dgate", "dup", "ddown")
        print(f"{label:8s} {name:26s} rel err vs fp32: " + " ".join(f"{n} {r:.3e}" for n, r in zip(names, rel)))


def per_op(moe, x, gate_proj, up_proj, down_proj, counts, offsets, dout, args):
    """Each fused GEMM of `moe_experts` next to the unfused ops it replaces (the weight gradients
    are torch._grouped_mm in both)."""
    from prime_kernels.moe_experts import gemm
    from prime_kernels.moe_experts.kernels import TileTable

    x, gate_proj, up_proj, down_proj = (t.detach() for t in (x, gate_proj, up_proj, down_proj))
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    table = TileTable(counts, x.shape[0], gemm.BLOCK_M)
    gate, up, h = gemm.fc1(x, gate_proj, up_proj, table, LIMIT, num_sms=num_sms)
    dgate_dup, _ = gemm.dswiglu(dout, down_proj, gate, up, table, LIMIT, num_sms=num_sms)
    dgate, dup = dgate_dup[:, :INTERMEDIATE], dgate_dup[:, INTERMEDIATE:]
    flops = 2 * x.shape[0] * HIDDEN * INTERMEDIATE

    def mm(a, b):
        return torch._grouped_mm(a, b, offs=offsets)

    def activation_backward():
        with torch.enable_grad():
            gate_ = gate.detach().requires_grad_()
            up_ = up.detach().requires_grad_()
            return torch.autograd.grad(compiled_swiglu(gate_, up_), (gate_, up_), h)

    ops = {
        "fc1 + SwiGLU": (
            2,
            lambda: gemm.fc1(x, gate_proj, up_proj, table, LIMIT, num_sms=num_sms),
            lambda: compiled_swiglu(mm(x, gate_proj.transpose(-2, -1)), mm(x, up_proj.transpose(-2, -1))),
        ),
        "down": (
            1,
            lambda: gemm.down(h, down_proj, table, num_sms=num_sms),
            lambda: mm(h, down_proj.transpose(-2, -1)),
        ),
        "dh + SwiGLU backward": (
            1,
            lambda: gemm.dswiglu(dout, down_proj, gate, up, table, LIMIT, num_sms=num_sms),
            lambda: (mm(dout, down_proj), activation_backward()),
        ),
        "dx": (
            2,
            lambda: gemm.dx(dgate, dup, gate_proj, up_proj, table, num_sms=num_sms),
            lambda: mm(dgate, gate_proj) + mm(dup, up_proj),
        ),
    }
    with torch.no_grad():
        for name, (gemms, fused, unfused) in ops.items():
            times = bench({"fused": fused, "unfused": unfused}, args.iters, args.repeats)
            print(
                f"         {name:26s} moe_experts {times['fused']:7.3f} ms {gemms * flops / times['fused'] / 1e9:4.0f} "
                f"TFLOP/s | today {times['unfused']:7.3f} ms {gemms * flops / times['unfused'] / 1e9:4.0f} TFLOP/s"
            )


if __name__ == "__main__":
    main()
