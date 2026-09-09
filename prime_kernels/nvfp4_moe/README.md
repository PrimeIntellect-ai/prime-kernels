# NVFP4 routed-expert GEMM

This SM100 kernel quantizes BF16 inputs and expert weights to E2M1 with
block-16 E4M3 scales. Activations have one FP32 outer scale per token; weights
have one per expert. Gated experts must pass a fused gate/up weight to share
the outer weight scale.

`grouped_gemm(x, weight_t, offs=offsets, backward="dequant_bf16")` accepts
`x[M, K]`, `weight_t[E, K, N]`, and cumulative INT32 row offsets. The token
dispatcher aligns expert row counts to `TOKEN_GROUP_ALIGNMENT`. Expert hidden
and intermediate dimensions must be positive multiples of 32.

The default backward reconstructs the packed forward operands and computes
BF16 dgrad and wgrad. `backward="bf16"` uses the original BF16 operands.
Master parameters and optimizer precision are controlled by the trainer.
There is no adaptive 4/6 scaling. Use `fullgraph=False` with the trainer.

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
retains the Transformer Engine attribution in its CUDA source. The build
uses packaged CUTLASS headers; no compiler or CUTLASS package is needed by a
finished wheel at runtime.
