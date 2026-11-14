#!/usr/bin/env python3
"""Time NVFP4 MoE expert GEMMs against BF16 and MXFP8 on one Blackwell GPU, end to end and by component.

For each model shape and tokens per expert, one expert layer (gate_up then down projection) runs forward and
backward with:
- ``bf16``: torch grouped GEMMs;
- ``mxfp8``: prime-kernels' mxfp8_moe;
- ``nvfp4``: nvfp4_moe (NVFP4 forward, MXFP8 backward with stochastic rounding on the dequantized operands),
  quantizing the weights in the GEMM or receiving them pre-quantized (as FSDP's quantized all-gather would).

Components (quantizers, the NVFP4 GEMM, the stochastic-rounding MXFP8 quantizer against round-to-nearest) show
where the time goes.

    python benchmarks/nvfp4_moe.py --json results.json
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

import prime_kernels

SHAPES = {
    # name: (hidden, expert intermediate, local experts)
    "qwen3-30b-a3b": (2048, 768, 128),
    "qwen3-235b-a22b/ep8": (4096, 1536, 16),
    "dsv3/ep8": (7168, 2048, 32),
}
TOKENS_PER_EXPERT = (256, 1024, 4096, 16384)


def timed(fn, warmup: int = 3, iters: int = 10) -> float:
    """Median milliseconds per call."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def layer(gemm, x, gate_up, down, offsets, grad):
    """One expert layer forward and backward: gate_up, SwiGLU, down."""

    def run():
        x.grad = gate_up.grad = down.grad = None
        gate, up = gemm(x, gate_up, offsets).chunk(2, dim=-1)
        out = gemm(F.silu(gate) * up, down, offsets)
        out.backward(grad)

    return run


def bench_shape(name: str, hidden: int, inter: int, experts: int, tokens: int, kernels) -> dict:
    nvfp4, mxfp8 = kernels
    rows = tokens * experts
    offsets = torch.full((experts,), tokens, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    x = (torch.randn(rows, hidden, device="cuda") * 0.5).bfloat16().requires_grad_()
    gate_up = (torch.randn(experts, 2 * inter, hidden, device="cuda") * 0.02).bfloat16()
    gate_up = gate_up.transpose(-2, -1).requires_grad_()  # [E, K, N] logical, as prime-rl passes it
    down = (torch.randn(experts, hidden, inter, device="cuda") * 0.02).bfloat16().transpose(-2, -1).requires_grad_()
    grad = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    result = {"shape": name, "hidden": hidden, "inter": inter, "experts": experts, "tokens_per_expert": tokens}

    def grouped(a, w, o):
        return F.grouped_mm(a, w, offs=o, out_dtype=torch.bfloat16)

    result["bf16"] = timed(layer(grouped, x, gate_up, down, offsets, grad))
    if mxfp8 is not None:
        result["mxfp8"] = timed(layer(lambda a, w, o: mxfp8.grouped_gemm(a, w, o), x, gate_up, down, offsets, grad))
    from prime_kernels.nvfp4_moe.grouped_gemm import quantize_expert_weight

    # Pre-quantized: what FSDP's quantized all-gather hands the GEMM (weights quantized once per gather, reused by
    # every GEMM on them)
    quantized = {id(w): quantize_expert_weight(w.detach().transpose(-2, -1)) for w in (gate_up, down)}
    for prequantized in (False, True):

        def gemm(a, w, o, prequantized=prequantized):
            return nvfp4.grouped_gemm(a, w, offs=o, weight_quantized=quantized[id(w)] if prequantized else None)

        result["nvfp4" + ("+prequantized" if prequantized else "")] = timed(layer(gemm, x, gate_up, down, offsets, grad))

    # Components on the gate_up projection
    with torch.no_grad():
        result["c/quantize_activations"] = timed(lambda: nvfp4.quantize_activations(x, offsets))
        result["c/quantize_weights"] = timed(lambda: nvfp4.quantize_weights(gate_up))
        qx = nvfp4.quantize_activations(x, offsets)
        qw = nvfp4.quantize_weights(gate_up)
        from prime_kernels.nvfp4_moe._ops import _grouped_mm

        result["c/nvfp4_gemm"] = timed(
            lambda: _grouped_mm(
                qx.data,
                qw.data.transpose(-2, -1),
                qx.block_scales,
                qw.block_scales,
                offsets,
                qx.global_scales,
                qw.global_scales,
            )
        )
        result["c/bf16_gemm"] = timed(lambda: grouped(x, gate_up, offsets))
        if mxfp8 is not None:
            from prime_kernels.mxfp8_moe.grouped_gemm import _quantize_dim1, quantize_rows
            from prime_kernels.nvfp4_moe import stochastic as sr

            grad_up = torch.randn(rows, 2 * inter, device="cuda", dtype=torch.bfloat16)
            result["c/grad_quant_nearest"] = timed(lambda: (quantize_rows(grad_up), _quantize_dim1(grad_up)))
            # The fused stochastic-rounding kernels against the PyTorch reference of the same bits
            result["c/grad_quant_stochastic"] = timed(lambda: sr.quantize_grad(grad_up, 0, 1))
            result["c/grad_quant_stochastic_torch"] = timed(
                lambda: (sr.reference_rows(grad_up, 0), sr.reference_columns(grad_up, 1))
            )
            result["c/operand_requant_stochastic"] = timed(
                lambda: (sr.quantize_activation_columns(qx, 2), sr.quantize_weight_for_dgrad(qw, 3))
            )
            result["c/operand_requant_stochastic_torch"] = timed(
                lambda: (
                    sr.reference_columns(qx.dequantize(), 2),
                    sr.reference_rows(qw.dequantize().contiguous(), 3),
                )
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="*", default=list(SHAPES), choices=list(SHAPES))
    parser.add_argument("--tokens", nargs="*", type=int, default=list(TOKENS_PER_EXPERT))
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args()

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise SystemExit(f"needs SM100/SM103, this is SM{capability[0]}{capability[1]}")
    nvfp4 = prime_kernels.load("nvfp4_moe")
    mxfp8 = prime_kernels.load("mxfp8_moe") if "mxfp8_moe" in prime_kernels.KERNELS else None
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}")

    results = []
    for name in args.shapes:
        hidden, inter, experts = SHAPES[name]
        for tokens in args.tokens:
            try:
                result = bench_shape(name, hidden, inter, experts, tokens, (nvfp4, mxfp8))
            except torch.OutOfMemoryError:
                print(f"{name} T={tokens}: out of memory, skipped")
                torch.cuda.empty_cache()
                continue
            results.append(result)
            base = result["bf16"]
            layers = {k: v for k, v in result.items() if k == "mxfp8" or k.startswith("nvfp4")}
            print(
                f"{name} T={tokens}: bf16 {base:.2f} ms | "
                + " | ".join(f"{k} {v:.2f} ({base / v:.2f}x)" for k, v in layers.items())
            )
            print("    " + " | ".join(f"{k[2:]} {v:.3f}" for k, v in result.items() if k.startswith("c/")))
            torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as out:
            json.dump(results, out, indent=1)


if __name__ == "__main__":
    main()
