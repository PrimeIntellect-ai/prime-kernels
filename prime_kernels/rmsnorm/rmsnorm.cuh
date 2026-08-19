// Shared declarations for the fused RMSNorm + residual + MXFP8-quantize kernels.
//
// The public entry point is `fused_rmsnorm_residual_forward`, which picks a
// variant based on C. The individual `launch_*` functions expose each variant
// directly so the test harness can exercise all of them.

#ifndef RMSNORM_CUH
#define RMSNORM_CUH

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// Common signature of every variant launcher.
#define RMSNORM_LAUNCH_PARAMS                                                 \
    __nv_fp8_e4m3 *vals, __nv_fp8_e8m0 *scales, float *rrms,                  \
    const nv_bfloat16 *branch, const nv_bfloat16 *residual,                   \
    const nv_bfloat16 *weight, float epsilon, int N, int C, cudaStream_t stream



// One block per token, strided loop over C. Handles any C that is a multiple
// of 128 * 16 / BLOCK_SIZE. bf16 scale math.
void launch_rmsnorm_loop(RMSNORM_LAUNCH_PARAMS);

// Rung 01: the baseline with the improved quantization epilogue.
void launch_rmsnorm_epi(RMSNORM_LAUNCH_PARAMS);

// Rung 02: block size scaled to C so a thread's one vector covers the row.
// Requires C <= 8192; the launcher exits the process past that, so callers
// that iterate over shapes must respect the limit.
void launch_rmsnorm_fixed_c(RMSNORM_LAUNCH_PARAMS);

// Rung 03: 32 elements per thread, so one thread owns a whole MXFP8 group and
// the group amax needs no cross-lane reduction. Halves the block size against
// rung 02 for the same C. Same C <= 8192 bound, same std::exit past it.
void launch_rmsnorm_32pt(RMSNORM_LAUNCH_PARAMS);

// Rung 04: the same 32-elements-per-thread layout on a persistent grid sized to
// the occupancy, so `weight` is loaded once per block instead of once per token.
// Same C <= 8192 bound.
void launch_rmsnorm_pers32(RMSNORM_LAUNCH_PARAMS);

// Largest C whose per-block staging fits this device's opt-in shared memory,
// for a kernel staging `bytes_per_column` bytes per column. The TMA kernels
// have no other bound on C -- past this they fail with a bare "invalid
// argument" -- so whichever rung brings them back wants this. sm_120's 99 KB
// runs out around C=10112 at 10 bytes/column, B200's 227 KB around C=23168.
inline int max_c_for_smem(int bytes_per_column) {
    int dev = 0, limit = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (limit / bytes_per_column) & ~127;
}


// TMA without warp specialization: every warp runs the whole
// fetch -> reduce -> quantize sequence, separated by __syncthreads(), with the
// bulk copies for the next token issued one iteration ahead. fp32 scale math.
// The copies are issued before the wait, so a block keeps two in flight.
void launch_rmsnorm_tma_simple(RMSNORM_LAUNCH_PARAMS);

// Largest C this device can stage for the tma_simple family. The launchers have
// no other bound on C, so the tables skip rather than launch past it.
int rmsnorm_tma_simple_max_c();


// Same as `tma_simple`, but staged through tensor TMA with a 128B swizzle so the
// 16-wide layout's 2-way bank conflict disappears. Falls back to the flat copy
// when C/64 exceeds the TMA box limit of 256 (i.e. C > 16384).
void launch_rmsnorm_tma_simple_swz(RMSNORM_LAUNCH_PARAMS);



// Bandwidth ceiling for this access pattern: reads branch and residual, writes
// vals, and does nothing else -- no reduction, no barrier, no quantization. The
// byte traffic per element is identical to the real kernels (2 + 2 read, 1
// written), so this is how fast any correct implementation could possibly be.
// Produces garbage output; benchmark only, not in the test table.
void launch_rmsnorm_bandwidth(RMSNORM_LAUNCH_PARAMS);



// Auto-dispatching entry point used by the torch binding and the benchmark.
void fused_rmsnorm_residual_forward(RMSNORM_LAUNCH_PARAMS);

#endif  // RMSNORM_CUH
