#!/usr/bin/env bash
# Build DeepSelect (top-k kernels for the DSA lightning indexer) as a wheel.
#
# Upstream compiles only sm_100a/sm_103a. This build adds sm_90a (its gencode line is
# commented out upstream only to save compile time), so the wheel covers Hopper and
# datacenter Blackwell. There are no sm_120 kernels.
#
# Requires CUDA 12.9+ (cross-compiles without a GPU).
#
# Usage:
#   bash scripts/install_deep_select.sh --wheel-dir dist
#
# Options:
#   --ref REF        DeepSelect commit hash (default: bfa4507)
#   --wheel-dir DIR  Output wheel to DIR (default: ./dist)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DEEPSELECT_GIT_REPO="https://github.com/deepseek-ai/DeepSelect.git"
DEEPSELECT_GIT_REF="bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18"
WHEEL_DIR="$REPO_ROOT/dist"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ref)       DEEPSELECT_GIT_REF="$2"; shift 2 ;;
        --wheel-dir) WHEEL_DIR="$2";          shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

git clone "$DEEPSELECT_GIT_REPO" "$TMPDIR/DeepSelect"
cd "$TMPDIR/DeepSelect"
git checkout "$DEEPSELECT_GIT_REF"
git submodule update --init --recursive

sed -i 's|# "-gencode", "arch=compute_90a,code=sm_90a",|"-gencode", "arch=compute_90a,code=sm_90a",|' setup.py
grep -q '^ *"-gencode", "arch=compute_90a,code=sm_90a",' setup.py

# setup.py probes the platform with `lspci` before it reads DEEP_SELECT_BUILD_TARGET_PLATFORM,
# and build containers have neither lspci nor a GPU.
mkdir -p "$TMPDIR/bin"
printf '#!/bin/sh\necho "3D controller: NVIDIA Corporation Device"\n' > "$TMPDIR/bin/lspci"
chmod +x "$TMPDIR/bin/lspci"
export PATH="$TMPDIR/bin:$PATH"
export DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA

# Back to the invocation directory: a relative --wheel-dir must resolve there, not
# inside $TMPDIR, which the EXIT trap deletes before the caller ever sees the wheel.
cd "$REPO_ROOT"

mkdir -p "$WHEEL_DIR"
uv build --no-build-isolation --wheel --out-dir "$WHEEL_DIR" "$TMPDIR/DeepSelect"
ls -lh "$WHEEL_DIR"/deep_select*.whl
