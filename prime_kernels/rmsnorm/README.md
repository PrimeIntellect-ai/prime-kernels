# Fused RMSNorm + residual -> MXFP8

One kernel: RMSNorm over `branch`, residual added after the norm, output
quantized to MXFP8 (fp8 e4m3 values + e8m0 scales per 32 elements) in the
swizzled layout a tensor-core GEMM expects.

Targets `sm_100a` (B200) and `sm_120a`; needs CUDA >= 12.8 for the e8m0
intrinsics.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100a
cmake --build build -j

./build/test_rmsnorm      # bit-exact against a CPU reference
./run_bench.sh            # build, verify, sweep N x C, write results-*.txt
```

## From Python

The kernel is part of the `prime-kernels` wheel, as `prime_kernels.rmsnorm`. It builds for
`sm_100a` and `sm_120a`, and the registry says which of those this machine can run:

```python
import prime_kernels

if prime_kernels.is_available("rmsnorm"):
    rmsnorm = prime_kernels.load("rmsnorm")
    vals, scales, rrms = rmsnorm.fused_rmsnorm_residual(branch, residual, weight, 1e-6)
```

`branch` and `residual` are bf16 `(N, C)`, `weight` is bf16 `(C,)`, and both N and C must be
multiples of 128. `scales` comes back flat, already in the blocked layout — the same one
`flash_moe.pack_scales_blocked` produces. `csrc/torch_interface.cpp` is the whole binding;
it calls the same `fused_rmsnorm_residual_forward` dispatcher `bench.cu` measures.

## The ladder

| file | variant | idea |
|---|---|---|
| `rmsnorm_00_baseline.cu` | `loop` | one block per token, two passes over C |
| `rmsnorm_01_loop_epi.cu` | `epi` | + half2 epilogue, fp32 scale math |
| `rmsnorm_02_fixed_c.cu` | `fixed_c` | block sized to C, one pass, row in registers |
| `rmsnorm_03_fixed_32pt.cu` | `fixed_32pt` | 32 elements per thread: one MXFP8 group each |
| `rmsnorm_04_pers_32pt.cu` | `pers_32pt` | + persistent grid, weights loaded once |
| `rmsnorm_05_tma.cu` | `tma_simple`, `_swz` | row staged in shared memory via TMA |

`copy` is the ceiling: same byte traffic, no reduction or quantization.

Measurements and the dispatch rule are in [RESULTS.md](RESULTS.md).

## Testing

`test.cu` builds its reference on the CPU using the device conversion
intrinsics, fed the kernel's own `rrms`, so `vals` and `scales` compare
bit-exactly. A mismatch is a real bug, never tolerance noise.

`GenericVector` in `src/include/common.cuh` is Apache-2.0, from the `llmq`
project (IST Austria).
