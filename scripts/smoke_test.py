#!/usr/bin/env python3
"""GPU smoke tests run against a just-built kernel wheel in CI.

Each check does real device work (not just an import) so a wheel built for the wrong
arch, or one that links against the wrong torch ABI, fails here instead of silently
shipping. Not a correctness suite — that lives in each kernel's own repo (or
prime-rl's tests/unit/train/models/test_fused_moe.py for flash_moe); this only
guards the thing cross-compilation can't verify on its own: does the compiled
extension actually load and run on this GPU.

Usage: python smoke_test.py <prime-kernels|deep-ep|deep-gemm|torchao|flash-attn>
"""

import sys

import torch

assert torch.cuda.is_available(), "smoke tests require a GPU"

CAP = torch.cuda.get_device_capability()
CAP_STR = f"sm_{CAP[0]}{CAP[1]}"
print(f"Device: {torch.cuda.get_device_name()} ({CAP_STR}), torch {torch.__version__}")


def smoke_prime_kernels():
    import prime_kernels

    status = prime_kernels.status()
    print("prime_kernels.status():", status)
    if CAP < (10, 0):
        print(f"flash_moe is Blackwell-only (sm_100a); skipping on {CAP_STR}")
        return
    assert status.get("flash_moe") == "available", status
    flash_moe = prime_kernels.load("flash_moe")
    print("flash_moe loaded:", flash_moe)


def smoke_deep_gemm():
    import deep_gemm

    print("deep_gemm", deep_gemm.__version__)
    m, k, n = 128, 512, 512
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    deep_gemm.bf16_gemm_nt(a, b, out)
    err = (out.float() - (a @ b.T).float()).abs().max().item()
    print("bf16 gemm max err:", err)
    assert err < 1.0, f"deep_gemm output diverges: max err {err}"


def smoke_deep_ep():
    import deep_ep

    print("deep_ep", deep_ep.__file__)
    # Real dispatch/combine needs a multi-GPU NVSHMEM group; out of scope for a
    # single-process smoke test. Import + extension load is what cross-compilation
    # can get wrong (wrong arch, wrong torch ABI), so that's what this checks.


def smoke_torchao():
    import torchao  # noqa: F401
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    print("torchao", torchao.__version__)
    if CAP < (10, 0):
        print(f"MXFP8 is Blackwell-only (sm_100a); skipping on {CAP_STR}")
        return
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    mx = MXTensor.to_mx(x, elem_dtype=torch.float8_e4m3fn, block_size=32)
    print("MXTensor.to_mx OK:", mx.qdata.shape, mx.qdata.dtype)


def smoke_flash_attn():
    from flash_attn import __version__, flash_attn_func

    print("flash_attn", __version__)
    b, s, h, d = 2, 128, 8, 64
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, k, v, causal=True)
    assert out.shape == q.shape, out.shape
    assert torch.isfinite(out).all(), "flash_attn output contains non-finite values"


CHECKS = {
    "prime-kernels": smoke_prime_kernels,
    "deep-gemm": smoke_deep_gemm,
    "deep-ep": smoke_deep_ep,
    "torchao": smoke_torchao,
    "flash-attn": smoke_flash_attn,
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(CHECKS)}>", file=sys.stderr)
        sys.exit(1)
    CHECKS[sys.argv[1]]()
    print("OK")


if __name__ == "__main__":
    main()
