#!/usr/bin/env bash
# Build DeepEP (NVSHMEM-backed expert-parallel all-to-all kernels) as a wheel.
#
# Ported from prime-rl's scripts/install_ep_kernels.sh: prime-kernels' own CI is the
# canonical build site for this wheel now (see .github/workflows/build_and_release.yaml),
# so this script is self-contained rather than reaching into a sibling repo.
#
# Usage:
#   bash scripts/install_ep_kernels.sh --wheel-dir dist
#
# Options:
#   --workspace DIR       Build directory (default: ./ep_kernels_workspace)
#   --wheel-dir DIR       Wheel output directory (default: ./dist)
#   --deepep-ref REF      DeepEP commit hash (default: 29d31c0)
#   --nvshmem-ver VER     NVSHMEM version (default: 3.3.24)
#
# Set TORCH_CUDA_ARCH_LIST to cross compile without a GPU (e.g. "9.0;10.0;10.3" in CI);
# unset, the arch is detected from the GPU nvidia-smi reports.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DEEPEP_COMMIT_HASH="29d31c0"
NVSHMEM_VER="3.3.24"
WORKSPACE="$REPO_ROOT/ep_kernels_workspace"
WHEEL_DIR="$REPO_ROOT/dist"

while [[ $# -gt 0 ]]; do
    case $1 in
        --workspace)  WORKSPACE="$2";          shift 2 ;;
        --wheel-dir)  WHEEL_DIR="$2";          shift 2 ;;
        --deepep-ref) DEEPEP_COMMIT_HASH="$2"; shift 2 ;;
        --nvshmem-ver) NVSHMEM_VER="$2";       shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Auto-detect CUDA toolkit matching torch ───────────────────────────────────
TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
CUDA_MAJOR_MINOR=$(echo "$TORCH_CUDA_VER" | grep -oP '^\d+\.\d+')
CUDA_MAJOR=$(echo "$CUDA_MAJOR_MINOR" | cut -d. -f1)

CUDA_HOME="/usr/local/cuda-${CUDA_MAJOR_MINOR}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: Could not find CUDA toolkit matching torch (cuda ${TORCH_CUDA_VER}) at ${CUDA_HOME}" >&2
    exit 1
fi
export CUDA_HOME
# CUDA 13 moved the CCCL headers (cuda/std/...) out of the main include dir.
if [ -d "$CUDA_HOME/include/cccl" ]; then
    export CPATH="$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
fi

# DeepEP links -lcuda (the driver stub, not the runtime). NVIDIA's own `devel` Ubuntu
# images conveniently symlink it onto $CUDA_HOME/lib64/stubs, but PyTorch's manylinux
# builder images don't — the stub only exists under
# $CUDA_HOME/targets/<arch>-linux/lib/stubs (arch-linux name varies: x86_64-linux,
# sbsa-linux, ...). Find it rather than hardcode the arch directory name.
STUBS_DIR=$(find "$CUDA_HOME" -maxdepth 4 -type d -name stubs 2>/dev/null | head -1)
if [ -n "$STUBS_DIR" ]; then
    export LIBRARY_PATH="${STUBS_DIR}${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

NVCC_VER=$("$CUDA_HOME/bin/nvcc" --version | grep -oP 'release \K[\d.]+')
echo "Torch CUDA: ${TORCH_CUDA_VER}, nvcc: ${NVCC_VER} (${CUDA_HOME})"

# ── Auto-detect GPU architecture (honor a preset TORCH_CUDA_ARCH_LIST) ────────
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i 0 2>/dev/null)
    case "$GPU_NAME" in
        *H100*|*H200*|*H800*)   TORCH_CUDA_ARCH_LIST="9.0" ;;
        *B100*|*B200*|*GB200*)  TORCH_CUDA_ARCH_LIST="10.0" ;;
        *)
            echo "Could not auto-detect GPU arch from '$GPU_NAME'. Set TORCH_CUDA_ARCH_LIST manually." >&2
            exit 1
            ;;
    esac
fi
export TORCH_CUDA_ARCH_LIST

echo "================================================================"
echo " Building DeepEP kernels"
echo " CUDA:       ${CUDA_HOME} (${NVCC_VER})"
echo " GPU:        ${GPU_NAME:-cross-compile} (arch ${TORCH_CUDA_ARCH_LIST})"
echo " DeepEP:     ${DEEPEP_COMMIT_HASH}"
echo " NVSHMEM:    ${NVSHMEM_VER}"
echo " Workspace:  ${WORKSPACE}"
echo "================================================================"

mkdir -p "$WORKSPACE"

echo ""
echo "--- Installing build dependencies ---"
uv pip install cmake ninja

echo ""
echo "--- Setting up NVSHMEM ${NVSHMEM_VER} ---"

ARCH=$(uname -m)
case "${ARCH,,}" in
    x86_64|amd64)  NVSHMEM_SUBDIR="linux-x86_64" ;;
    aarch64|arm64) NVSHMEM_SUBDIR="linux-sbsa" ;;
    *) echo "Unsupported architecture: ${ARCH}" >&2; exit 1 ;;
esac

NVSHMEM_DIR="$WORKSPACE/nvshmem"
if [ ! -d "$NVSHMEM_DIR/lib" ]; then
    NVSHMEM_FILE="libnvshmem-${NVSHMEM_SUBDIR}-${NVSHMEM_VER}_cuda${CUDA_MAJOR}-archive.tar.xz"
    NVSHMEM_URL="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/${NVSHMEM_SUBDIR}/${NVSHMEM_FILE}"

    echo "Downloading ${NVSHMEM_URL}"
    curl -fSL "${NVSHMEM_URL}" -o "$WORKSPACE/${NVSHMEM_FILE}"
    tar -xf "$WORKSPACE/${NVSHMEM_FILE}" -C "$WORKSPACE"
    mv "$WORKSPACE/${NVSHMEM_FILE%.tar.xz}" "$NVSHMEM_DIR"
    rm -f "$WORKSPACE/${NVSHMEM_FILE}"
    rm -rf "$NVSHMEM_DIR/lib/bin" "$NVSHMEM_DIR/lib/share"
    echo "NVSHMEM extracted to ${NVSHMEM_DIR}"
else
    echo "NVSHMEM already present at ${NVSHMEM_DIR}, skipping download"
fi

export CMAKE_PREFIX_PATH="${NVSHMEM_DIR}/lib/cmake:${CMAKE_PREFIX_PATH:-}"
export NVSHMEM_DIR

echo ""
echo "--- Building DeepEP (${DEEPEP_COMMIT_HASH}) ---"

DEEPEP_DIR="$WORKSPACE/DeepEP"
if [ ! -d "$DEEPEP_DIR/.git" ]; then
    git clone https://github.com/deepseek-ai/DeepEP "$DEEPEP_DIR"
fi

cd "$DEEPEP_DIR"
git fetch origin
git checkout "$DEEPEP_COMMIT_HASH"

mkdir -p "$WHEEL_DIR"
python setup.py bdist_wheel --dist-dir "$WHEEL_DIR"

WHEEL=$(ls "$WHEEL_DIR"/deep_ep*.whl | head -1)
echo ""
echo "--- DeepEP wheel built at: $WHEEL ---"
echo "================================================================"
echo " DeepEP build complete"
echo "================================================================"
