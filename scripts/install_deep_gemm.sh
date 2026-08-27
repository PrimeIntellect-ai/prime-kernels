#!/usr/bin/env bash
# Build DeepGEMM (FP8 blockwise / grouped GEMM kernels) as a wheel.
#
# Ported from prime-rl's scripts/install_deep_gemm.sh: prime-kernels' own CI is the
# canonical build site for this wheel now (see .github/workflows/build_and_release.yaml).
#
# Requires CUDA 12.8+ and a Hopper/Blackwell GPU (cross-compiles fine without one).
#
# Usage:
#   bash scripts/install_deep_gemm.sh --wheel-dir dist
#
# Options:
#   --ref REF        DeepGEMM commit hash (default: 891d57b)
#   --wheel-dir DIR  Output wheel to DIR (default: ./dist)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DEEPGEMM_GIT_REPO="https://github.com/deepseek-ai/DeepGEMM.git"
DEEPGEMM_GIT_REF="891d57b4db1071624b5c8fa0d1e51cb317fa709f"
WHEEL_DIR="$REPO_ROOT/dist"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ref)       DEEPGEMM_GIT_REF="$2"; shift 2 ;;
        --wheel-dir) WHEEL_DIR="$2";        shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SHORT_REF="${DEEPGEMM_GIT_REF:0:7}"

CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "")
if [ -z "$CUDA_VERSION" ]; then
    echo "ERROR: nvcc not found. CUDA toolkit required." >&2
    exit 1
fi

echo "================================================================"
echo " Building DeepGEMM (${SHORT_REF})"
echo " CUDA: ${CUDA_VERSION}"
echo "================================================================"

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

git clone --recurse-submodules "$DEEPGEMM_GIT_REPO" "$TMPDIR/DeepGEMM"
cd "$TMPDIR/DeepGEMM"
git checkout "$DEEPGEMM_GIT_REF"
git submodule update --init --recursive

# Back to the invocation directory: a relative --wheel-dir must resolve there, not
# inside $TMPDIR, which the EXIT trap deletes before the caller ever sees the wheel.
cd "$REPO_ROOT"

mkdir -p "$WHEEL_DIR"
uv build --no-build-isolation --wheel --out-dir "$WHEEL_DIR" "$TMPDIR/DeepGEMM"
echo ""
echo "Wheel built:"
ls -lh "$WHEEL_DIR"/deep_gemm*.whl
echo "================================================================"
echo " DeepGEMM build complete"
echo "================================================================"
