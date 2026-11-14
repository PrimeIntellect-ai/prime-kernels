# NVFP4 routed-expert GEMM

This SM100 kernel quantizes BF16 inputs and expert weights to E2M1 with
block-16 E4M3 scales. Activations have one FP32 outer scale per token; weights
have one per expert. Gated experts must pass a fused gate/up weight to share
the outer weight scale.

`grouped_gemm(x, weight_t, offs=offsets)` accepts `x[M, K]`, `weight_t[E, K, N]`, and cumulative INT32 row offsets.
The token dispatcher aligns expert row counts to `TOKEN_GROUP_ALIGNMENT`. Expert hidden and intermediate dimensions
must be positive multiples of 32.

One recipe, after Thinking Machines' end-to-end 4-bit training:

- Forward: NVFP4 (E2M1, 1x16 blocks, E4M3 block scales, FP32 outer scale). Activations, both GEMM inputs, take a
  per-token outer scale and choose each block's scale with 4/6 (the lower-error of the max-6 and max-4 scales);
  weights take a per-expert outer scale and never 4/6. All quantizer arithmetic is IEEE (no approximate
  reciprocals).
- Backward: dgrad and wgrad in MXFP8 (E4M3, E8M0 scales per 32 elements) on the dequantized NVFP4 forward operands,
  so the gradients see the forward's rounding. The operands and the output gradient are quantized with stochastic
  rounding (`stochastic.py`), seeded by `stochastic.set_seed`: call it once per training step and a step reproduces
  the same bits.

Weights quantized ahead of the GEMM (`quantize_expert_weight`, e.g. per FSDP shard for a quantized all-gather) and
tokens quantized before dispatch (`dispatch.py`) give the same bytes as quantizing inside the GEMM. Master parameters
and optimizer precision are controlled by the trainer. Use `fullgraph=False` with the trainer.

Build with a CUDA development toolkit matching PyTorch's CUDA major.
Kernel-specific dependencies are declared in `kernels.toml` under
`build-requires`; the build backend installs the dependencies for the selected
kernels into the isolated build environment:

```sh
PRIME_KERNELS=nvfp4_moe PRIME_KERNELS_REQUIRE=1 MAX_JOBS=2 \
  UV_TORCH_BACKEND=cu130 uv build --wheel .
uv run pytest tests/test_nvfp4_moe.py
```

The grouped GEMM derives from MSLK and retains `MSLK_LICENSE`. The quantizer
retains Transformer Engine and FlashInfer attribution in its CUDA source. The build
uses packaged CUTLASS headers; no compiler or CUTLASS package is needed by a
finished wheel at runtime.
