#!/usr/bin/env bash
# Build DeepEP V2.5 (NCCL-backed expert-parallel kernels) as the wheel `deep-ep-v2`.
#
# The Python package is renamed `deep_ep` -> `deep_ep_v2` (extension `deep_ep_v2._C`) so
# it installs next to the V1 `deep_ep` wheel from install_ep_kernels.sh. The C++ header
# dir `include/deep_ep/` and the C++ namespaces stay as upstream: the JIT includes
# `<deep_ep/...>` relative to `<package>/include`.
#
# The extension is host C++ only. Every CUDA kernel is JIT-compiled at runtime by
# DeepJIT, so the wheel has no per-arch cubins. At runtime it needs a CUDA >= 13.1
# toolkit at CUDA_HOME and nvidia-nccl-cu13 >= 2.32.3 (NCCL device/GIN headers), which
# also has to be the NCCL torch loads: `import deep_ep_v2` checks that.
#
# Build requirements: CUDA 13 toolkit at CUDA_HOME (13.0 is enough for the host code),
# elfutils headers (elfutils/libdwfl.h; libdw is dlopened at runtime), uv, and a venv
# with torch active (VIRTUAL_ENV).
#
# Usage:
#   bash scripts/install_deep_ep_v2.sh --wheel-dir dist
#
# Options:
#   --ref REF        DeepEP commit hash (default: 93eb6eb, V2.5)
#   --wheel-dir DIR  Output wheel to DIR (default: ./dist)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DEEPEP_GIT_REPO="https://github.com/deepseek-ai/DeepEP.git"
DEEPEP_GIT_REF="93eb6eb238127e96c6d7a4a625a6dad158348509"
NCCL_VER="2.32.3"
WHEEL_DIR="$REPO_ROOT/dist"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ref)       DEEPEP_GIT_REF="$2"; shift 2 ;;
        --wheel-dir) WHEEL_DIR="$2";      shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
WHEEL_DIR="$(mkdir -p "$WHEEL_DIR" && cd "$WHEEL_DIR" && pwd)"

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

git clone "$DEEPEP_GIT_REPO" "$TMPDIR/DeepEP"
cd "$TMPDIR/DeepEP"
git checkout "$DEEPEP_GIT_REF"
# .gitmodules points DeepJIT at an ssh URL.
git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive
SHORT_REF=$(git rev-parse --short=7 HEAD)

# setup.py finds NCCL through the pip package (deep_ep/utils/find_pkgs.py). torch pins
# an older nvidia-nccl-cu13; V2.5 needs the device/GIN API of >= 2.32.3.
uv pip install "nvidia-nccl-cu13==${NCCL_VER}"

# Rename the Python package. setup.py would otherwise version the dirty tree `+local`.
mv deep_ep deep_ep_v2
grep -rlZ 'deep_ep\._C' deep_ep_v2 --include='*.py' | xargs -0 sed -i 's/\bdeep_ep\._C\b/deep_ep_v2._C/g'
sed -i -e "s|'deep_ep|'deep_ep_v2|g" \
       -e "s|/deep_ep/include|/deep_ep_v2/include|" \
       -e "s|name='deep_ep_v2'|name='deep-ep-v2'|" \
       -e "s|revision = '+local'|revision = '+${SHORT_REF}'|" setup.py
if grep -nE "'deep_ep['.]|/deep_ep/include" setup.py; then
    echo "ERROR: setup.py still references the deep_ep package" >&2; exit 1
fi
if grep -rn '\bdeep_ep\._C\b\|^ *\(import\|from\) deep_ep\b' deep_ep_v2 --include='*.py'; then
    echo "ERROR: deep_ep_v2 still imports deep_ep" >&2; exit 1
fi

# The build image's glibc headers predate pidfd_open/pidfd_getfd (same numbers on x86_64 and aarch64).
sed -i '\|#include <sys/syscall.h>|a\
#ifndef SYS_pidfd_open\
#define SYS_pidfd_open 434\
#endif\
#ifndef SYS_pidfd_getfd\
#define SYS_pidfd_getfd 438\
#endif' csrc/kernels/comm/symmetric.hpp

# setup.py hard-codes /usr/local/cuda/include/cccl.
export CUDA_HOME
export CPATH="$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"

cd "$REPO_ROOT"
uv build --no-build-isolation --wheel --out-dir "$TMPDIR/dist" "$TMPDIR/DeepEP"

# setup.py bakes in an absolute RPATH to the build venv's NCCL. Point it at the
# nvidia-nccl-cu13 pip package instead: site-packages/nvidia/nccl/lib, next to
# site-packages/deep_ep_v2/_C*.so. That is also the libnccl torch loads, which
# `import deep_ep_v2` (check_nccl_so) requires.
echo "--- Patching deep_ep_v2._C's RPATH to the nccl pip package ---"
python -m wheel unpack "$TMPDIR"/dist/deep_ep_v2-*.whl --dest "$TMPDIR/unpacked"
SO_FILE=$(find "$TMPDIR/unpacked" -path '*/deep_ep_v2/_C*.so')
patchelf --set-rpath '$ORIGIN/../nvidia/nccl/lib' "$SO_FILE"
echo "New RPATH: $(patchelf --print-rpath "$SO_FILE")"
python -m wheel pack "$TMPDIR"/unpacked/deep_ep_v2-* --dest-dir "$WHEEL_DIR"
ls -lh "$WHEEL_DIR"/deep_ep_v2-*.whl
