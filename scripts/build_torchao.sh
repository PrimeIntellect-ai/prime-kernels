#!/usr/bin/env bash
# Build our pinned torchao (pytorch/ao) as a wheel.
#
# Pinned to v0.18.0 (5f2baf9d) — the first tag that compiles against torch 2.13 /
# CUDA 13: 0.17.0's `_C_cutlass_90a` extension hits a `STABLE_TORCH_LIBRARY_IMPL`
# compile error there (default arguments on a function parameter). Bonus: 0.18.0
# builds as `cp310-abi3` (torch stable ABI), so it should survive the next torch
# bump without a rebuild.
#
# Usage:
#   bash scripts/build_torchao.sh --wheel-dir dist
#
# Options:
#   --ref REF        pytorch/ao commit hash (default: 5f2baf9d575cf732362594c998c399902942531f)
#   --wheel-dir DIR  Output wheel to DIR (default: ./dist)
#
# Set TORCH_CUDA_ARCH_LIST to control the CUTLASS/MXFP8 arch list (default: "9.0a;10.0a" —
# sm_90a enables the CUTLASS kernels, sm_100a the MXFP8 extension).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TORCHAO_GIT_REF="5f2baf9d575cf732362594c998c399902942531f"
WHEEL_DIR="$REPO_ROOT/dist"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ref)       TORCHAO_GIT_REF="$2"; shift 2 ;;
        --wheel-dir) WHEEL_DIR="$2";       shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a;10.0a}"

echo "================================================================"
echo " Building torchao (${TORCHAO_GIT_REF:0:7})"
echo " Arch list: ${TORCH_CUDA_ARCH_LIST}"
echo "================================================================"

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

git clone https://github.com/pytorch/ao "$TMPDIR/ao"
git -C "$TMPDIR/ao" checkout "$TORCHAO_GIT_REF"
git -C "$TMPDIR/ao" submodule update --init --recursive

mkdir -p "$WHEEL_DIR"
uv build --no-build-isolation --wheel --out-dir "$WHEEL_DIR" "$TMPDIR/ao"
echo ""
echo "Wheel built:"
ls -lh "$WHEEL_DIR"/torchao*.whl
echo "================================================================"
echo " torchao build complete"
echo "================================================================"
