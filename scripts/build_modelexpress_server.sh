#!/usr/bin/env bash
# Build the modelexpress-server wheel: the ModelExpress server binary and the Valkey server it
# stores its metadata in, packed as executables that install into the environment's bin directory.
#
# ModelExpress (ai-dynamo/modelexpress) ships its Python clients on PyPI but its server only as a
# container image. prime-rl starts a run-scoped server next to the trainer, so it needs the binary
# in its environment. The server's in-memory backend does not implement the weight-version (Refit)
# API that RL weight updates use, so the wheel also ships Valkey, the BSD-licensed Redis fork.
#
# Pins: the MX rev must match the `modelexpress` client prime-rl pins (MX_REF is the v0.8.0 tag),
# and the wheel takes that client's version.
#
# Usage:
#   bash scripts/build_modelexpress_server.sh --wheel-dir dist
#
# Requires cargo, protoc, a C compiler, make, strip and python3 on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MX_REF="fd606d605178db96706bfd48f50851c744501a1d"
WHEEL_VERSION="0.8.0"
VALKEY_VERSION="8.1.12"
VALKEY_SHA256="675bd143a4c42022f7ce10fb012a7a61e0a70b139ff166e3e63fca2bdf05027b"
WHEEL_DIR="$REPO_ROOT/dist"
PLATFORM="manylinux_2_28_$(uname -m)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --wheel-dir) WHEEL_DIR="$2"; shift 2 ;;
        --platform)  PLATFORM="$2";  shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for tool in cargo protoc make strip python3; do
    command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 1; }
done

echo "================================================================"
echo " Building modelexpress-server ${WHEEL_VERSION} (MX ${MX_REF:0:7}, Valkey ${VALKEY_VERSION})"
echo " Platform tag: ${PLATFORM}"
echo "================================================================"

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

git clone https://github.com/ai-dynamo/modelexpress.git "$TMPDIR/modelexpress"
git -C "$TMPDIR/modelexpress" checkout "$MX_REF"
cargo build --release --locked \
    --manifest-path "$TMPDIR/modelexpress/Cargo.toml" \
    -p modelexpress-server --bin modelexpress-server

curl --fail --location --silent --show-error \
    --output "$TMPDIR/valkey.tar.gz" \
    "https://github.com/valkey-io/valkey/archive/refs/tags/${VALKEY_VERSION}.tar.gz"
echo "$VALKEY_SHA256  $TMPDIR/valkey.tar.gz" | sha256sum --check
tar -xzf "$TMPDIR/valkey.tar.gz" -C "$TMPDIR"
make -C "$TMPDIR/valkey-${VALKEY_VERSION}" -j"$(nproc)" valkey-server BUILD_TLS=no MALLOC=libc

STAGE="$TMPDIR/stage"
mkdir -p "$STAGE"
strip -o "$STAGE/modelexpress-server" "${CARGO_TARGET_DIR:-$TMPDIR/modelexpress/target}/release/modelexpress-server"
strip -o "$STAGE/valkey-server" "$TMPDIR/valkey-${VALKEY_VERSION}/src/valkey-server"
cp "$TMPDIR/modelexpress/LICENSE" "$STAGE/LICENSE-modelexpress"
cp "$TMPDIR/valkey-${VALKEY_VERSION}/COPYING" "$STAGE/COPYING-valkey"

python3 "$SCRIPT_DIR/pack_bin_wheel.py" \
    --name modelexpress-server \
    --version "$WHEEL_VERSION" \
    --platform "$PLATFORM" \
    --summary "ModelExpress server (ai-dynamo/modelexpress ${MX_REF:0:7}) and Valkey ${VALKEY_VERSION}" \
    --license "Apache-2.0 AND BSD-3-Clause" \
    --bin "$STAGE/modelexpress-server" \
    --bin "$STAGE/valkey-server" \
    --license-file "$STAGE/LICENSE-modelexpress" \
    --license-file "$STAGE/COPYING-valkey" \
    --out-dir "$WHEEL_DIR"

echo "================================================================"
echo " modelexpress-server build complete"
echo "================================================================"
