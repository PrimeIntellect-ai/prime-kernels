// bench_rmsnorm.cu
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rmsnorm.cuh"

using LaunchFn = void (*)(RMSNORM_LAUNCH_PARAMS);

struct BenchVariant {
    const char* name;
    LaunchFn launch;
    int max_c;          // 0 = unbounded
    int (*max_c_fn)();  // device-dependent limit, overrides max_c when set
};

static const BenchVariant kVariants[] = {
    {"auto",     &fused_rmsnorm_residual_forward, 0},
    // the ladder
    {"loop",     &launch_rmsnorm_loop,            0},
    {"epi",      &launch_rmsnorm_epi,             0},
    {"fixed_c",  &launch_rmsnorm_fixed_c,         8192},
    {"fixed_32pt",     &launch_rmsnorm_32pt,            8192},
    {"pers_32pt", &launch_rmsnorm_pers32,          8192},
    {"tma_simple",      &launch_rmsnorm_tma_simple,      0, &rmsnorm_tma_simple_max_c},
    {"tma_simple_swz",  &launch_rmsnorm_tma_simple_swz,  0, &rmsnorm_tma_simple_max_c},
    // still in rmsnorm.cu
    {"copy",     &launch_rmsnorm_bandwidth,       0},
};
static const int kNumVariants = (int)(sizeof(kVariants) / sizeof(kVariants[0]));

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t err_ = (expr);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,   \
                    cudaGetErrorString(err_));                                \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// 0 = unbounded. Device-dependent limits (tma's shared memory) win when set.
static int variant_limit(const BenchVariant& v) {
    return v.max_c_fn ? v.max_c_fn() : v.max_c;
}

static void print_variant_names(FILE* f) {
    for (int i = 0; i < kNumVariants; ++i) {
        fprintf(f, "%s%s", i ? "," : "", kVariants[i].name);
    }
    fprintf(f, "\n");
}

static const BenchVariant* find_variant(const std::string& name) {
    for (int i = 0; i < kNumVariants; ++i) {
        if (name == kVariants[i].name) return &kVariants[i];
    }
    return nullptr;
}

static std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        if (comma > start) out.push_back(s.substr(start, comma - start));
        start = comma + 1;
    }
    return out;
}

// Bytes the kernel must move for one iteration: read branch + residual (bf16),
// write vals (fp8) + rrms (f32). weight and scales are negligible.
static double traffic_bytes(long long N, long long C) {
    return (double)N * (double)C * (2 + 2 + 1) + (double)N * 4;
}

// ---------------------------------------------------------------------------
// device buffers, allocated once for the largest shape and reused
// ---------------------------------------------------------------------------

struct Buffers {
    nv_bfloat16 *branch = nullptr, *residual = nullptr, *weight = nullptr;
    __nv_fp8_e4m3* vals = nullptr;
    __nv_fp8_e8m0* scales = nullptr;
    float* rrms = nullptr;

    size_t pool_elem = 0;   // rotation pool: several shapes' worth of input

    void alloc(size_t elems, size_t max_c) {
        pool_elem = elems;
        CUDA_CHECK(cudaMalloc(&branch, elems * sizeof(nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&residual, elems * sizeof(nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&weight, max_c * sizeof(nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&vals, elems * sizeof(__nv_fp8_e4m3)));
        CUDA_CHECK(cudaMalloc(&scales, (elems / 32 + 512) * sizeof(__nv_fp8_e8m0)));
        // C >= 128, so a slice never needs more than elems/128 rows
        CUDA_CHECK(cudaMalloc(&rrms, (elems / 128 + 128) * sizeof(float)));
    }

    void free_all() {
        CUDA_CHECK(cudaFree(branch));
        CUDA_CHECK(cudaFree(residual));
        CUDA_CHECK(cudaFree(weight));
        CUDA_CHECK(cudaFree(vals));
        CUDA_CHECK(cudaFree(scales));
        CUDA_CHECK(cudaFree(rrms));
    }
};

// One shape's slot in the pool. Rotating between iterations is what keeps the
// inputs cold: a small shape's own footprint may sit inside L2, but the pool
// does not, so consecutive iterations never hit the same lines. Without this,
// varying N independently of C would silently turn every small-N row into an
// L2 benchmark.
static double time_variant(const BenchVariant& v, const Buffers& b, int N, int C,
                           int iters, int warmup, double min_ms, cudaStream_t stream) {
    const size_t shape_elem = (size_t)N * (size_t)C;
    const int slices = (int)std::max<size_t>(1, b.pool_elem / shape_elem);

    auto launch = [&](int i) {
        const size_t off = (size_t)(i % slices) * shape_elem;
        v.launch(b.vals + off, b.scales + off / 32, b.rrms + (off / C),
                 b.branch + off, b.residual + off, b.weight,
                 1e-5f, N, C, stream);
    };

    // warmup also primes the static grid_size lookup in the launcher
    for (int i = 0; i < warmup; ++i) launch(i);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // A small shape can run in single-digit microseconds, where launch overhead
    // and clock wobble dominate and the spread swamps the effect being measured.
    // Calibrate on one launch and run enough of them to cover min_ms.
    int iters_eff = iters;
    if (min_ms > 0.0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch(0);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float one = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&one, start, stop));
        if (one > 0.f) {
            const long long need = (long long)(min_ms / one) + 1;
            iters_eff = (int)std::min<long long>(std::max<long long>(iters, need), 100000);
        }
    }

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters_eff; ++i) launch(i);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters_eff;
}

static void print_device_info(size_t* l2_bytes_out) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    int clock_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    // DDR: two transfers per clock. Reported clocks are unreliable on some HBM
    // parts, so this is a sanity reference, not a spec figure.
    const double peak_gbs = 2.0 * (double)clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;

    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);

    printf("# device %d: %s  sm_%d%d  %d SMs  L2 %.1f MB  %.1f GB total\n",
           dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize / 1048576.0, prop.totalGlobalMem / 1073741824.0);
    printf("# nominal DRAM peak %.0f GB/s (%d-bit @ %.2f GHz)  "
           "cuda runtime %d.%d driver %d.%d\n",
           peak_gbs, bus_bits, clock_khz / 1e6, rt / 1000, (rt % 1000) / 10,
           drv / 1000, (drv % 1000) / 10);
    *l2_bytes_out = (size_t)prop.l2CacheSize;
}

int main(int argc, char** argv) {
    int N = 8192;
    int C = 2048;
    int iters = 100;
    int warmup = 10;

    std::string variant = "auto";

    // sweep mode
    bool sweep = false;
    int reps = 3;
    double target_gib = 2.0;   // rotation pool; must exceed L2 by a lot
    double min_ms = 20.0;      // minimum wall time per timing call
    int fixed_n = 0;           // 0 = derive N from target_gib
    bool per_c_rows = false;   // one N per C rather than a cross product
    // N straddling the point where the baseline switches block size
    // (blk_64 * SMs < N; ~2.4-3.5k on a 148-SM part). "auto" instead picks
    // one N per C so a single pass moves ~--gib GiB.
    std::string rows = "1024,8192,65536";
    // Span the whole range by default. A new kernel's first run should show
// where it lands, not where it was predicted to land -- the winner has moved
// with C every time, and narrowing the sweep to the expected regime is how
// you miss that.
    std::string cols = "1024,2048,4096,8192,16384";
    std::string variants = "copy,loop,epi,fixed_c,fixed_32pt,pers_32pt,tma_simple,tma_simple_swz";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::atoi(argv[++i]); };
        if (a == "-N") { N = next(); fixed_n = N; }
        else if (a == "-C") C = next();
        else if (a == "-i") iters = next();
        else if (a == "-w") warmup = next();
        else if (a == "-v" || a == "--variant") variant = argv[++i];
        else if (a == "--sweep") sweep = true;
        else if (a == "--reps") reps = next();
        else if (a == "--gib") target_gib = std::atof(argv[++i]);
        else if (a == "--min-ms") min_ms = std::atof(argv[++i]);
        else if (a == "--cols") cols = argv[++i];
        else if (a == "--rows") rows = argv[++i];
        else if (a == "--variants") variants = argv[++i];
        else if (a == "--list-variants") { print_variant_names(stdout); return 0; }
        else {
            fprintf(stderr, "usage: %s [-N rows] [-C cols] [-i iters] [-w warmup]\n"
                            "          [--variant auto|one_step|cancel|loop|epi|tma|tma_simple*|copy|quant*]\n"
                            "\n"
                            "       %s --sweep [--cols 4096,8192,...] [--rows 1024,8192,...]\n"
                            "          [--variants loop,epi,...]\n"
                            "          [--gib 2.0] [--reps 3] [-i iters] [-w warmup] [-N rows]\n"
                            "\n"
                            "  --sweep crosses --rows with --cols. With no --rows it picks one N per C\n"
                            "  so a pass moves ~--gib GiB. Either way iterations rotate through a\n"
                            "  pool of that size, so small shapes are not measured out of L2.\n",
                    argv[0], argv[0]);
            return 1;
        }
    }

    if (!sweep) {
        // ---- single shape, one variant (unchanged behaviour) ----
        const BenchVariant* v = find_variant(variant);
            if (!v) {
            fprintf(stderr, "unknown variant '%s'\nknown variants: ", variant.c_str());
            print_variant_names(stderr);
            return 1;
        }
        if (variant_limit(*v) && C > variant_limit(*v)) {
            printf("%-9s N=%d C=%d  SKIP (C > %d)\n", v->name, N, C, variant_limit(*v));
            return 0;
        }
        assert(N % 128 == 0);
        assert(C % 128 == 0);

        const size_t n_elem = (size_t)N * C;
        std::vector<nv_bfloat16> h_branch(n_elem), h_residual(n_elem), h_weight(C);
        unsigned s = 12345u;
        auto rnd = [&]() {
            s = s * 1664525u + 1013904223u;
            return __float2bfloat16(((float)(s >> 8) / 8388608.0f) - 1.0f);
        };
        for (size_t i = 0; i < n_elem; ++i) h_branch[i] = rnd();
        for (size_t i = 0; i < n_elem; ++i) h_residual[i] = rnd();
        for (int i = 0; i < C; ++i) h_weight[i] = rnd();

        Buffers b;
        b.alloc(n_elem, C);
        CUDA_CHECK(cudaMemcpy(b.branch, h_branch.data(), n_elem * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b.residual, h_residual.data(), n_elem * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b.weight, h_weight.data(), C * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        const double per_iter_ms = time_variant(*v, b, N, C, iters, warmup, min_ms, stream);
        const double gbps = traffic_bytes(N, C) / (per_iter_ms * 1e-3) / 1e9;
        printf("%-9s N=%-6d C=%-5d %.4f ms/iter  %6.1f GB/s\n",
               v->name, N, C, per_iter_ms, gbps);

        b.free_all();
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    }

    // ---- sweep ----
    size_t l2_bytes = 0;
    print_device_info(&l2_bytes);

    std::vector<int> c_list;
    for (const std::string& t : split_commas(cols)) c_list.push_back(std::atoi(t.c_str()));

    std::vector<const BenchVariant*> v_list;
    for (const std::string& t : split_commas(variants)) {
        const BenchVariant* v = find_variant(t);
        if (!v) {
            fprintf(stderr, "unknown variant '%s'\nknown variants: ", t.c_str());
            print_variant_names(stderr);
            return 1;
        }
        v_list.push_back(v);
    }

    // Both axes are swept independently: the launchers pick block size and grid
    // from N as well as C (the baseline switches at blk_64 * SMs < N), so a
    // sweep that derives N from C cannot see those decisions. Default rows
    // straddle that threshold on a 148-SM part.
    std::vector<int> n_list;
    if (fixed_n) {
        n_list.push_back(fixed_n);
    } else if (rows != "auto") {
        for (const std::string& t : split_commas(rows)) n_list.push_back(std::atoi(t.c_str()));
    } else {
        // no rows given: one N per C, sized so a single pass moves ~target_gib
        for (int c : c_list) {
            long long n = (long long)(target_gib * 1073741824.0 / (5.0 * c));
            n = (n / 128) * 128;
            n_list.push_back((int)std::max(128LL, n));
        }
        per_c_rows = true;
    }

    struct Shape { int n, c; };
    std::vector<Shape> shapes;
    size_t max_elem = 0, max_c = 0;
    for (size_t ci = 0; ci < c_list.size(); ++ci) {
        assert(c_list[ci] % 128 == 0);
        if (per_c_rows) {
            shapes.push_back({n_list[ci], c_list[ci]});
        } else {
            for (int n : n_list) shapes.push_back({n, c_list[ci]});
        }
        max_c = std::max(max_c, (size_t)c_list[ci]);
    }
    for (const Shape& s : shapes) {
        assert(s.n % 128 == 0);
        max_elem = std::max(max_elem, (size_t)s.n * (size_t)s.c);
    }

    // The pool holds several of the largest shape, or ~target_gib, whichever is
    // bigger -- it is what the rotation cycles through.
    const size_t target_elem = (size_t)(target_gib * 1073741824.0 / 5.0);
    const size_t pool_elem = std::max(max_elem, target_elem);

    printf("# sweep: %d reps, >=%d iters (>=%.0f ms per timing call), %d warmup; "
           "%.1f GiB rotation pool\n",
           reps, iters, min_ms, warmup, pool_elem * 5.0 / 1073741824.0);
    printf("# cols=%s rows=%s variants=%s\n", cols.c_str(),
           per_c_rows ? "auto (one per C)" : rows.c_str(),
           variants.c_str());

    // The pool can be several GiB, so fill the device from a tiled host block
    // rather than materialising it host-side. Values only have to be finite and
    // non-degenerate -- nothing here checks results.
    Buffers b;
    b.alloc(pool_elem, max_c);
    {
        const size_t tile = std::min<size_t>(pool_elem, 8u << 20);   // 16 MB of bf16
        std::vector<nv_bfloat16> h_tile(tile), h_weight(max_c);
        unsigned s = 12345u;
        auto rnd = [&]() {
            s = s * 1664525u + 1013904223u;
            return __float2bfloat16(((float)(s >> 8) / 8388608.0f) - 1.0f);
        };
        for (size_t i = 0; i < tile; ++i) h_tile[i] = rnd();
        for (size_t i = 0; i < max_c; ++i) h_weight[i] = rnd();
        for (size_t off = 0; off < pool_elem; off += tile) {
            const size_t n = std::min(tile, pool_elem - off);
            CUDA_CHECK(cudaMemcpy(b.branch + off, h_tile.data(), n * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.residual + off, h_tile.data(), n * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(b.weight, h_weight.data(), max_c * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<std::string> csv;

    for (const Shape& shape : shapes) {
        const int c = shape.c, n = shape.n;
        const double footprint_mb = traffic_bytes(n, c) / 1048576.0;
        const int slices = (int)std::max<size_t>(1, pool_elem / ((size_t)n * (size_t)c));

        printf("\nN=%d C=%d  (%.0f MB per pass, %.1fx L2, %d slices)\n", n, c, footprint_mb,
               l2_bytes ? footprint_mb * 1048576.0 / (double)l2_bytes : 0.0, slices);
        printf("  %-16s %10s %10s %10s %8s %8s\n",
               "variant", "ms(min)", "ms(med)", "GB/s", "vs copy", "spread");

        // repeats interleaved across variants, so drift hits every variant alike
        std::vector<std::vector<double>> times(v_list.size());
        for (int r = 0; r < reps; ++r) {
            for (size_t vi = 0; vi < v_list.size(); ++vi) {
                const BenchVariant* v = v_list[vi];
                if (variant_limit(*v) && c > variant_limit(*v)) continue;
                times[vi].push_back(time_variant(*v, b, n, c, iters, warmup, min_ms, stream));
            }
        }

        // The fastest repeat is the least-disturbed one: every repeat does
        // identical work, so anything above the minimum is interference. The
        // spread column is how you tell a real difference from a noisy box --
        // ignore a delta that is not several times the spread.
        double copy_ms = 0.0;
        for (size_t vi = 0; vi < v_list.size(); ++vi) {
            if (std::string(v_list[vi]->name) == "copy" && !times[vi].empty()) {
                copy_ms = *std::min_element(times[vi].begin(), times[vi].end());
            }
        }

        for (size_t vi = 0; vi < v_list.size(); ++vi) {
            const BenchVariant* v = v_list[vi];
            if (times[vi].empty()) {
                printf("  %-16s %10s (C > %d)\n", v->name, "SKIP", variant_limit(*v));
                continue;
            }
            std::vector<double> t = times[vi];
            std::sort(t.begin(), t.end());
            const double best = t.front();
            const double med = t[t.size() / 2];
            const double spread = 100.0 * (t.back() - best) / best;
            const double gbps = traffic_bytes(n, c) / (best * 1e-3) / 1e9;
            if (copy_ms > 0.0) {
                printf("  %-16s %10.4f %10.4f %10.1f %7.2fx %7.1f%%\n",
                       v->name, best, med, gbps, best / copy_ms, spread);
            } else {
                printf("  %-16s %10.4f %10.4f %10.1f %8s %7.1f%%\n",
                       v->name, best, med, gbps, "-", spread);
            }
            char line[256];
            snprintf(line, sizeof(line), "%d,%d,%s,%.6f,%.6f,%.1f,%.1f,%d",
                     n, c, v->name, best, med, gbps, spread, slices);
            csv.push_back(line);
        }
    }

    printf("\n# csv: N,C,variant,ms_min,ms_median,gbps,spread_pct,slices\n");
    for (const std::string& l : csv) printf("%s\n", l.c_str());

    b.free_all();
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
