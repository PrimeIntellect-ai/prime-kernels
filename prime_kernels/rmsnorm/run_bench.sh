#!/usr/bin/env bash
# Build and benchmark this kernel on whatever GPU is in front of it.
#
# Copy the repo to the target machine and run:
#
#     ./run_bench.sh                  # detect arch, build, test, sweep
#     ARCH=100a ./run_bench.sh        # force an arch (B200 = 100a)
#     DEVICE=3 ./run_bench.sh         # pick a GPU (nvidia-smi index)
#     NCU=1 ./run_bench.sh            # also collect instruction counts
#     SKIP_TEST=1 ./run_bench.sh      # skip the correctness gate
#     GIB=4 ./run_bench.sh            # bigger rotation pool
#     ROWS=2048 COLS=8192 ./run_bench.sh   # pin the swept shapes
#
# Everything lands in results-<host>-<gpu>-<arch>.txt. That file is the thing
# worth copying back; it carries the device info needed to interpret it.
#
# The sweep crosses N with C, because the launchers pick block size and grid
# from both -- the baseline switches at blk_64 * SMs < N, fixed_c switches on C.
# Iterations rotate through a GIB-sized pool (default 2) so that a small shape,
# whose own footprint fits in L2, is still measured against cold data; the
# report prints the pass size, the L2 multiple and the number of slices.

set -euo pipefail
cd "$(dirname "$0")"

GIB="${GIB:-2}"
REPS="${REPS:-3}"
ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-10}"
# C <= 2048 is not a shape that gets run on the target; keep the sweep on
# sizes that matter and override COLS for anything else.
# COLS and VARIANTS deliberately have no default here. `bench` owns them, so a
# stale copy of this script cannot ask for a variant the binary dropped -- which
# is exactly what happened once, when a pre-rename script asked for `one_step`.
COLS="${COLS:-}"
ROWS="${ROWS:-}"
VARIANTS="${VARIANTS:-}"

# ---------------------------------------------------------------------------
# toolchain and arch
# ---------------------------------------------------------------------------

command -v cmake >/dev/null || { echo "cmake not found" >&2; exit 1; }
command -v nvcc  >/dev/null || { echo "nvcc not found (is CUDA on PATH?)" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }

# CUDA's default device order is by "speed", not by PCI bus, so on a mixed or
# multi-GPU box `nvidia-smi -i 0` and CUDA device 0 need not be the same card --
# which silently mislabels results. Pin both to the same ordering.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DEVICE="${DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="$DEVICE"

if [[ -z "${ARCH:-}" ]]; then
    # compute_cap comes back as e.g. "10.0" on B200, "12.0" on a 5060 Ti.
    cap="$(nvidia-smi -i "$DEVICE" --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')"
    if [[ -z "$cap" ]]; then
        echo "could not detect compute capability; pass ARCH=100a explicitly" >&2
        exit 1
    fi
    # The 'a' suffix is required: the e8m0 conversion intrinsics are
    # arch-specific and will not compile against the portable target.
    ARCH="${cap}a"
fi

gpu="$(nvidia-smi -i "$DEVICE" --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' | tr -cd '[:alnum:]_-')"
out="results-$(hostname -s)-${gpu}-${ARCH}.txt"
build="build-${ARCH}"

echo "arch=${ARCH}  gpu=${gpu}  build=${build}  results=${out}"

# ---------------------------------------------------------------------------
# everything below is captured
# ---------------------------------------------------------------------------

{
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)  host=$(hostname -s)  arch=${ARCH}"
    echo "# $(nvcc --version | tail -2 | head -1)"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,clocks.max.memory \
               --format=csv | sed 's/^/# /'
    echo "# GIB=${GIB} REPS=${REPS} ITERS=${ITERS} WARMUP=${WARMUP}"
    echo
} > "$out"

echo "==> configuring"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" >/dev/null

echo "==> building"
if ! cmake --build "$build" -j >/dev/null; then
    echo "BUILD FAILED -- anything already in $build is stale; not running it" >&2
    exit 1
fi

# Belt and braces: a failed build that somehow returns 0, or a hand-run sweep
# against an old binary, both look like a successful measurement of the wrong
# code. Refuse if any source is newer than what we are about to run.
stale="$(find . -path "./build*" -prune -o \( -name '*.cu' -o -name '*.cuh' -o -name 'CMakeLists.txt' \) \
         -newer "./$build/bench" -print 2>/dev/null | head -3)"
if [[ -n "$stale" ]]; then
    echo "STALE BINARY -- these are newer than $build/bench:" >&2
    echo "$stale" >&2
    exit 1
fi

# Anything explicitly requested is checked against the binary first, so a bad
# name fails here with the valid list instead of after the build and the gate.
sweep_args=(--sweep --gib "$GIB" --reps "$REPS" -i "$ITERS" -w "$WARMUP")
known="$("./$build/bench" --list-variants)"
if [[ -n "$COLS" ]]; then sweep_args+=(--cols "$COLS"); fi
if [[ -n "$ROWS" ]]; then sweep_args+=(--rows "$ROWS"); fi
if [[ -n "$VARIANTS" ]]; then
    for v in ${VARIANTS//,/ }; do
        if [[ ",$known," != *",$v,"* ]]; then
            echo "unknown variant '$v'; this build has: $known" >&2
            exit 1
        fi
    done
    sweep_args+=(--variants "$VARIANTS")
fi

if [[ "${SKIP_TEST:-0}" != "1" ]]; then
    echo "==> correctness gate"
    if ! "./$build/test_rmsnorm" | tee -a "$out" | tail -1; then
        echo "CORRECTNESS FAILED -- timings below would be meaningless" | tee -a "$out"
        exit 1
    fi
    echo >> "$out"
fi

echo "==> sweep (this allocates ~${GIB} GiB on the device)"
"./$build/bench" "${sweep_args[@]}" | tee -a "$out"

# ---------------------------------------------------------------------------
# optional: instruction counts. Structural, so these are the numbers that
# compare across machines -- unlike the GB/s above.
# ---------------------------------------------------------------------------

if [[ "${NCU:-0}" == "1" ]]; then
    if ! command -v ncu >/dev/null; then
        echo "NCU=1 but ncu is not on PATH -- skipping" | tee -a "$out"
    else
        echo "==> ncu counters"
        {
            echo
            echo "# ncu: N=4096, one launch per variant/shape (cold cache, so the"
            echo "# durations here are not comparable to the sweep above)"
            printf "# %-8s %-6s %14s %6s %10s\n" variant C instructions regs occupancy
        } | tee -a "$out"
        # mirror the shapes and variants the sweep actually reported
        ncu_cols="${COLS:-$(grep -o 'cols=[^ ]*' "$out" | tail -1 | cut -d= -f2)}"
        ncu_variants="${VARIANTS:-$(grep -o 'variants=[^ ]*' "$out" | tail -1 | cut -d= -f2)}"
        for c in ${ncu_cols//,/ }; do
            for v in ${ncu_variants//,/ }; do
                res="$(ncu --metrics smsp__inst_executed.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active \
                           --print-summary per-kernel \
                           "./$build/bench" -N 4096 -C "$c" -i 1 -w 0 --variant "$v" 2>/dev/null || true)"
                inst="$(awk '/smsp__inst_executed.sum/ {print $NF}' <<<"$res" | head -1)"
                regs="$(awk '/launch__registers_per_thread/ {print $NF}' <<<"$res" | head -1)"
                occ="$(awk '/sm__warps_active.avg.pct_of_peak/ {print $NF}' <<<"$res" | head -1)"
                [[ -z "$inst" ]] && continue
                printf "  %-8s %-6s %14s %6s %10s\n" "$v" "$c" "$inst" "$regs" "$occ" | tee -a "$out"
            done
        done
    fi
fi

echo
echo "wrote $out"
