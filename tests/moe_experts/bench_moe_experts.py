"""Benchmark `moe_experts` against prime-rl's current expert compute at V4.1 Flash shapes.

prime-rl today (`GroupedGemmExpertCompute` with `BF16ExpertCompute` and V4's `ClampedSwiglu`):
gate and up as two `torch._grouped_mm` calls (or one over the packed `gate_up_proj`), the clamped
SwiGLU in eager PyTorch (or under torch.compile when the decoder block is compiled), and the down
projection as a third `torch._grouped_mm`; autograd does the backward.

Per GPU, 16k tokens routed top-6 give 98304 expert rows: 4 local experts of ~24.5k rows each at
EP=96 and 48 local experts of ~2k rows each at EP=8. Group sizes get +-20% imbalance and are
padded to the kernel's TOKEN_GROUP_ALIGNMENT.

    python tests/moe_experts/bench_moe_experts.py
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


def today(x, gate_proj, up_proj, down_proj, offsets, activation):
    if up_proj is None:
        gate, up = torch._grouped_mm(x, gate_proj.transpose(-2, -1), offs=offsets).chunk(2, dim=-1)
    else:
        gate = torch._grouped_mm(x, gate_proj.transpose(-2, -1), offs=offsets)
        up = torch._grouped_mm(x, up_proj.transpose(-2, -1), offs=offsets)
    return torch._grouped_mm(activation(gate, up), down_proj.transpose(-2, -1), offs=offsets)


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
    args = parser.parse_args()

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
        variants = {
            "today (eager act)": (lambda: today(x, gate_proj, up_proj, down_proj, offsets, clamped_swiglu), separate),
            "today (compiled act)": (
                lambda: today(x, gate_proj, up_proj, down_proj, offsets, compiled_swiglu),
                separate,
            ),
            "today packed gate_up": (lambda: today(x, gate_up, None, down_proj, offsets, compiled_swiglu), packed),
            "moe_experts": (lambda: moe.moe_experts(x, gate_proj, up_proj, down_proj, counts, LIMIT), separate),
            "moe_experts packed": (lambda: moe.moe_experts(x, gate_up, None, down_proj, counts, LIMIT), packed),
        }

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
        if args.per_op:
            per_op(moe, x, gate_proj, up_proj, down_proj, counts, offsets, dout, args)


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
