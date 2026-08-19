// Correctness tests for the fused RMSNorm + residual + MXFP8-quantize kernels.
//
// Strategy: the CPU reference reuses the *device* conversion intrinsics
// (__nv_cvt_*, __float2bfloat16), all of which are __host__ __device__, and it
// is fed the kernel's own `rrms` output as the normalization factor. Every
// remaining operation is then deterministic, so `vals` and `scales` must match
// bit-exactly -- any difference is a real bug, not reduction-order noise.
//
// `rrms` itself is checked separately against a double-precision sum, with a
// tolerance that covers fp32 accumulation and the rsqrtf approximation.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rmsnorm.cuh"

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t err_ = (expr);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,   \
                    cudaGetErrorString(err_));                                \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// host-side bf16 helpers
// ---------------------------------------------------------------------------

static inline float b2f(nv_bfloat16 x) { return __bfloat162float(x); }
static inline nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }

// A bf16 multiply rounds the (exactly representable) fp32 product to bf16,
// so this reproduces the device `operator*` bit-for-bit.
static inline nv_bfloat16 bmul(nv_bfloat16 a, nv_bfloat16 b) {
    return f2b(b2f(a) * b2f(b));
}

// ---------------------------------------------------------------------------
// reference scale-factor layout
// ---------------------------------------------------------------------------

// Independent re-implementation of get_sf_out_offset using division/modulo
// rather than shifts and ORs, so that a mistake in the bit-twiddling shows up.
// SF layout: [numMTiles, numKTiles, 32, 4, 4]
static int64_t ref_sf_offset(int mIdx, int kIdx, int numKTiles) {
    const int mTileIdx  = mIdx / 128;
    const int outerMIdx = mIdx % 32;
    const int innerMIdx = (mIdx / 32) % 4;
    const int kTileIdx  = kIdx / 4;
    const int innerKIdx = kIdx % 4;
    return ((int64_t)mTileIdx * numKTiles + kTileIdx) * 512
           + outerMIdx * 16 + innerMIdx * 4 + innerKIdx;
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

enum class ScaleMath {
    Bf16,  // nw = bf16(s * branch) * w      (matches the reference transformer)
    Fp32,  // nw = s * (branch * w)          (one_step variant)
};

struct RefResult {
    std::vector<uint8_t> vals;    // fp8 e4m3 bit patterns
    std::vector<uint8_t> scales;  // e8m0 bit patterns
    std::vector<double> rrms;     // high-precision normalization factors
};

static RefResult reference(const std::vector<nv_bfloat16>& branch,
                           const std::vector<nv_bfloat16>& residual,
                           const std::vector<nv_bfloat16>& weight,
                           const std::vector<float>& s_gpu,  // kernel's rrms
                           float epsilon, int N, int C,
                           ScaleMath math, int group_size) {
    const int num_k_tiles = C / 128;
    RefResult out;
    out.vals.assign((size_t)N * C, 0);
    out.scales.assign((size_t)(N / 128) * num_k_tiles * 512, 0);
    out.rrms.assign(N, 0.0);

    std::vector<nv_bfloat16> res(C);

    for (int row = 0; row < N; ++row) {
        // high-precision rrms, for the separate tolerance check
        double ss = 0.0;
        for (int c = 0; c < C; ++c) {
            const double v = b2f(branch[(size_t)row * C + c]);
            ss += v * v;
        }
        out.rrms[row] = 1.0 / std::sqrt(ss / C + (double)epsilon);

        // everything below is driven by the kernel's own s, so it is exact
        const float s = s_gpu[row];

        for (int c = 0; c < C; ++c) {
            const nv_bfloat16 bv = branch[(size_t)row * C + c];
            const nv_bfloat16 wv = weight[c];
            const float rv = b2f(residual[(size_t)row * C + c]);

            float nw;
            if (math == ScaleMath::Bf16) {
                // scale in bf16, then multiply by the weight in bf16
                nw = b2f(bmul(f2b(s * b2f(bv)), wv));
            } else {
                // exact bf16xbf16 product in fp32, then one fp32 multiply by s
                nw = s * (b2f(bv) * b2f(wv));
            }
            // the residual add happens after the norm, in the input precision
            res[c] = f2b(nw + rv);
        }

        for (int g = 0; g < C; g += group_size) {
            float amax = 0.f;
            for (int k = 0; k < group_size; ++k) {
                amax = std::fmax(amax, std::fabs(b2f(res[g + k])));
            }

            const float scale_factor = amax * (1.f / 448.f);
            const __nv_fp8_storage_t sf =
                    __nv_cvt_float_to_e8m0(scale_factor, __NV_SATFINITE, cudaRoundPosInf);
            const nv_bfloat16 inv_sf = nv_bfloat16(__nv_cvt_e8m0_to_bf16raw(254 - sf));

            for (int k = 0; k < group_size; ++k) {
                const nv_bfloat16 scaled = bmul(res[g + k], inv_sf);
                out.vals[(size_t)row * C + g + k] = __nv_cvt_bfloat16raw_to_fp8(
                        static_cast<__nv_bfloat16_raw>(scaled), __NV_SATFINITE, __NV_E4M3);
            }

            // one scale per 32 elements, regardless of the amax group size
            for (int k = 0; k < group_size; k += 32) {
                out.scales[ref_sf_offset(row, (g + k) / 32, num_k_tiles)] = sf;
            }
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------

struct Stats {
    int mismatches = 0;
    int checked = 0;
};

static Stats compare_bytes(const char* what, const std::vector<uint8_t>& got,
                           const std::vector<uint8_t>& want, int max_report) {
    Stats st;
    st.checked = (int)want.size();
    for (size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i]) {
            if (st.mismatches < max_report) {
                printf("      %s[%zu]: got 0x%02x, want 0x%02x\n", what, i, got[i], want[i]);
            }
            ++st.mismatches;
        }
    }
    return st;
}

// ---------------------------------------------------------------------------
// launch with a watchdog, so a deadlocked kernel does not hang the test run
// ---------------------------------------------------------------------------

static bool sync_with_timeout(cudaStream_t stream, double seconds) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const cudaError_t st = cudaStreamQuery(stream);
        if (st == cudaSuccess) return true;
        if (st != cudaErrorNotReady) {
            fprintf(stderr, "      cuda error: %s\n", cudaGetErrorString(st));
            return true;  // surfaced by the caller's error check
        }
        const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        if (dt.count() > seconds) return false;
    }
}

// ---------------------------------------------------------------------------
// variant table
// ---------------------------------------------------------------------------

using LaunchFn = void (*)(RMSNORM_LAUNCH_PARAMS);

struct Variant {
    const char* name;
    LaunchFn launch;
    ScaleMath math;
    int max_c;  // 0 = unbounded
    bool wip;   // only run when named explicitly via --variant
    int (*max_c_fn)();  // device-dependent limit, overrides max_c when set
};

static const Variant kVariants[] = {
    // the ladder
    {"loop",     &launch_rmsnorm_loop,     ScaleMath::Bf16, 0,    false},
    {"epi",      &launch_rmsnorm_epi,      ScaleMath::Fp32, 0,    false},
    {"fixed_c",  &launch_rmsnorm_fixed_c,  ScaleMath::Fp32, 8192, false},
    {"fixed_32pt",     &launch_rmsnorm_32pt,     ScaleMath::Fp32, 8192, false},
    {"pers_32pt", &launch_rmsnorm_pers32,   ScaleMath::Fp32, 8192, false},
    // the shared-memory-staged lineage
    {"tma_simple",      &launch_rmsnorm_tma_simple,      ScaleMath::Fp32, 0, false, &rmsnorm_tma_simple_max_c},
    {"tma_simple_swz",  &launch_rmsnorm_tma_simple_swz,  ScaleMath::Fp32, 0, false, &rmsnorm_tma_simple_max_c},
    // still in rmsnorm.cu
};
static const int kNumVariants = (int)(sizeof(kVariants) / sizeof(kVariants[0]));

// ---------------------------------------------------------------------------

struct Buffers {
    nv_bfloat16 *d_branch = nullptr, *d_residual = nullptr, *d_weight = nullptr;
    __nv_fp8_e4m3* d_vals = nullptr;
    __nv_fp8_e8m0* d_scales = nullptr;
    float* d_rrms = nullptr;
    std::vector<nv_bfloat16> h_branch, h_residual, h_weight;
    size_t n_elem = 0, n_scales = 0;
};

static void make_inputs(Buffers& b, int N, int C) {
    b.n_elem = (size_t)N * C;
    b.n_scales = (size_t)(N / 128) * (C / 128) * 512;

    b.h_branch.resize(b.n_elem);
    b.h_residual.resize(b.n_elem);
    b.h_weight.resize(C);

    unsigned s = 12345u;
    auto rnd = [&]() {
        s = s * 1664525u + 1013904223u;
        return __float2bfloat16(((float)(s >> 8) / 8388608.0f) - 1.0f);
    };
    for (size_t i = 0; i < b.n_elem; ++i) b.h_branch[i] = rnd();
    for (size_t i = 0; i < b.n_elem; ++i) b.h_residual[i] = rnd();
    for (int i = 0; i < C; ++i) b.h_weight[i] = rnd();

    // edge cases: an all-zero row (amax == 0 -> e8m0 of zero), and a row with a
    // residual large enough to push the group scale near the top of the range.
    const nv_bfloat16 zero = __float2bfloat16(0.f);
    for (int c = 0; c < C; ++c) {
        b.h_branch[c] = zero;
        b.h_residual[c] = zero;
    }
    b.h_residual[(size_t)C + 5] = __float2bfloat16(240.f);

    CUDA_CHECK(cudaMalloc(&b.d_branch, b.n_elem * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&b.d_residual, b.n_elem * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&b.d_weight, C * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&b.d_vals, b.n_elem * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&b.d_scales, b.n_scales * sizeof(__nv_fp8_e8m0)));
    CUDA_CHECK(cudaMalloc(&b.d_rrms, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(b.d_branch, b.h_branch.data(), b.n_elem * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_residual, b.h_residual.data(), b.n_elem * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_weight, b.h_weight.data(), C * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
}

static void free_inputs(Buffers& b) {
    cudaFree(b.d_branch); cudaFree(b.d_residual); cudaFree(b.d_weight);
    cudaFree(b.d_vals); cudaFree(b.d_scales); cudaFree(b.d_rrms);
}

// returns true on pass
static bool run_case(const Variant& v, int N, int C, float epsilon,
                     int group_size, double timeout_s, int max_report) {
    printf("  %-9s N=%-5d C=%-5d ", v.name, N, C);
    fflush(stdout);

    const int limit = v.max_c_fn ? v.max_c_fn() : v.max_c;
    if (limit && C > limit) {
        printf("SKIP (C > %d)\n", limit);
        return true;
    }

    Buffers b;
    make_inputs(b, N, C);

    // poison the outputs so that "kernel never wrote here" is a visible failure
    CUDA_CHECK(cudaMemset(b.d_vals, 0xA5, b.n_elem * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMemset(b.d_scales, 0xA5, b.n_scales * sizeof(__nv_fp8_e8m0)));
    CUDA_CHECK(cudaMemset(b.d_rrms, 0xA5, N * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    v.launch(b.d_vals, b.d_scales, b.d_rrms, b.d_branch, b.d_residual,
             b.d_weight, epsilon, N, C, stream);

    const cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("FAIL (launch: %s)\n", cudaGetErrorString(launch_err));
        free_inputs(b);
        return false;
    }

    if (!sync_with_timeout(stream, timeout_s)) {
        printf("FAIL (TIMEOUT after %.0fs)\n", timeout_s);
        printf("      The kernel is not making progress -- most likely a barrier\n"
               "      that is never satisfied. It is still resident on the device,\n"
               "      so the remaining cases cannot run; aborting the test run.\n");
        fflush(stdout);
        std::_Exit(2);
    }

    const cudaError_t run_err = cudaGetLastError();
    if (run_err != cudaSuccess) {
        printf("FAIL (execution: %s)\n", cudaGetErrorString(run_err));
        free_inputs(b);
        return false;
    }

    std::vector<uint8_t> g_vals(b.n_elem), g_scales(b.n_scales);
    std::vector<float> g_rrms(N);
    CUDA_CHECK(cudaMemcpy(g_vals.data(), b.d_vals, b.n_elem, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_scales.data(), b.d_scales, b.n_scales, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_rrms.data(), b.d_rrms, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamDestroy(stream));

    const RefResult ref = reference(b.h_branch, b.h_residual, b.h_weight, g_rrms,
                                   epsilon, N, C, v.math, group_size);

    // 1. rrms against the double-precision reference
    int rrms_bad = 0;
    double worst_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        const double want = ref.rrms[i];
        const double got = g_rrms[i];
        const double rel = std::fabs(got - want) / want;
        if (rel > worst_rel) worst_rel = rel;
        if (!(rel <= 2e-6)) {
            if (rrms_bad < max_report) {
                printf("\n      rrms[%d]: got %.9g, want %.9g (rel %.3g)", i, got, want, rel);
            }
            ++rrms_bad;
        }
    }

    // 2. quantized values and scales, bit-exact
    Stats vs{};
    vs.checked = (int)ref.vals.size();
    for (size_t i = 0; i < ref.vals.size(); ++i) {
        if (g_vals[i] != ref.vals[i]) {
            if (vs.mismatches < max_report) {
                if (vs.mismatches == 0 && rrms_bad == 0) printf("\n");
                printf("      vals[row %zu, col %zu]: got 0x%02x, want 0x%02x\n",
                       i / C, i % C, g_vals[i], ref.vals[i]);
            }
            ++vs.mismatches;
        }
    }
    const Stats ss = compare_bytes("scales", g_scales, ref.scales, max_report);

    const bool pass = (rrms_bad == 0 && vs.mismatches == 0 && ss.mismatches == 0);
    if (pass) {
        printf("ok   (rrms rel %.2g)\n", worst_rel);
    } else {
        printf("      -> FAIL: rrms %d/%d, vals %d/%d, scales %d/%d wrong\n",
               rrms_bad, N, vs.mismatches, vs.checked, ss.mismatches, ss.checked);
    }

    free_inputs(b);
    return pass;
}

int main(int argc, char** argv) {
    std::vector<std::pair<int, int>> shapes;
    std::string only;
    float epsilon = 1e-5f;
    int group_size = 32;
    double timeout_s = 10.0;
    int max_report = 5;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_i = [&]() { return std::atoi(argv[++i]); };
        if (a == "--variant" || a == "-v") only = argv[++i];
        else if (a == "--shape") {
            const int n = next_i();
            const int c = next_i();
            shapes.emplace_back(n, c);
        }
        else if (a == "--group") group_size = next_i();
        else if (a == "--timeout") timeout_s = std::atof(argv[++i]);
        else if (a == "--report") max_report = next_i();
        else {
            fprintf(stderr,
                    "usage: %s [--variant NAME] [--shape N C]... [--group G]\n"
                    "          [--timeout SECONDS] [--report K]\n"
                    "variants: loop tma\n", argv[0]);
            return 1;
        }
    }

    if (shapes.empty()) {
        shapes = {{128, 128}, {128, 512}, {256, 2048}, {128, 3072}, {256, 4096},
                  {128, 8192}, {128, 16384}};
    }

    printf("fused RMSNorm + residual + MXFP8 quantize -- correctness\n");
    printf("reference: CPU, fed the kernel's own rrms; vals/scales must be bit-exact\n\n");

    int failed = 0, total = 0;
    for (int vi = 0; vi < kNumVariants; ++vi) {
        const Variant& v = kVariants[vi];
        if (!only.empty() && only != v.name) continue;
        if (only.empty() && v.wip) {
            printf("variant %s: work in progress, run with --variant %s to test it\n\n",
                   v.name, v.name);
            continue;
        }
        printf("variant %s (%s scale math)\n", v.name,
               v.math == ScaleMath::Bf16 ? "bf16" : "fp32");
        for (const auto& sh : shapes) {
            ++total;
            if (!run_case(v, sh.first, sh.second, epsilon, group_size, timeout_s, max_report)) {
                ++failed;
            }
        }
        printf("\n");
    }

    printf("%d/%d cases passed\n", total - failed, total);
    return failed ? 1 : 0;
}
