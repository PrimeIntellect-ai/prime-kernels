#!/usr/bin/env bash
# Build flash-attn (FA2) as a wheel.
#
# Pinned to v2.8.3 — the exact version prime-rl's `flash-attn` extra already expects.
# Built here instead of consumed from mjun0812/flash-attention-prebuild-wheels: that
# project's newest build for this flash-attn + torch2.13 combo below cu130 is cu126,
# and CUDA 12.6 predates 12.8 — the floor flash-attn's own setup.py requires before it
# will even emit sm_100/sm_120 gencodes (see FLASH_ATTN_CUDA_ARCHS below) — so that
# prebuilt wheel only ever contained sm_80/sm_90 cubins and no PTX, hard-failing on
# every Blackwell GPU.
#
# Requires CUDA 12.8+ (for Blackwell sm_100/sm_120 codegen) — this repo's containers
# are on 12.9, comfortably above that floor.
#
# Usage:
#   bash scripts/install_flash_attn.sh --wheel-dir dist
#
# Options:
#   --ref REF        flash-attention git tag/commit (default: v2.8.3)
#   --wheel-dir DIR  Output wheel to DIR (default: ./dist)
#
# Set FLASH_ATTN_CUDA_ARCHS to control the -gencode arch list (default: "80;90;100;120"
# — sm_80 Ampere, sm_90 Hopper, sm_100 datacenter Blackwell, sm_120 workstation/consumer
# Blackwell). Set FLASH_ATTN_LOCAL_VERSION to stamp a `+<tag>` suffix onto the wheel
# version (flash-attn's own version string is static, unlike deep-ep/deep-gemm/torchao's
# upstream `+<rev>` naming, so nothing else disambiguates a future CUDA/torch bump).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

FLASH_ATTN_GIT_REPO="https://github.com/Dao-AILab/flash-attention.git"
FLASH_ATTN_GIT_REF="v2.8.3"
WHEEL_DIR="$REPO_ROOT/dist"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ref)       FLASH_ATTN_GIT_REF="$2"; shift 2 ;;
        --wheel-dir) WHEEL_DIR="$2";          shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80;90;100;120}"

CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "")
if [ -z "$CUDA_VERSION" ]; then
    echo "ERROR: nvcc not found. CUDA toolkit required." >&2
    exit 1
fi

echo "================================================================"
echo " Building flash-attn (${FLASH_ATTN_GIT_REF})"
echo " CUDA: ${CUDA_VERSION}"
echo " Arch list: ${FLASH_ATTN_CUDA_ARCHS}"
echo "================================================================"

# setup.py's setup_requires (packaging, psutil) and install_requires (einops) aren't
# installed automatically under --no-build-isolation.
uv pip install packaging psutil einops

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

git clone --recurse-submodules "$FLASH_ATTN_GIT_REPO" "$TMPDIR/flash-attention"
git -C "$TMPDIR/flash-attention" checkout "$FLASH_ATTN_GIT_REF"
git -C "$TMPDIR/flash-attention" submodule update --init --recursive

mkdir -p "$WHEEL_DIR"
uv build --no-build-isolation --wheel --out-dir "$WHEEL_DIR" "$TMPDIR/flash-attention"
echo ""
echo "Wheel built:"
ls -lh "$WHEEL_DIR"/flash_attn-*.whl
echo "================================================================"
echo " flash-attn build complete"
echo "================================================================"
