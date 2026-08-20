# prime-kernels

CUDA kernels for Prime Intellect training stacks, shipped as one wheel, `prime-kernels`.

```
.
├── setup.py                  # builds what the manifest declares; no edit needed to add a kernel
└── prime_kernels/
    ├── kernels.toml          # the manifest: one table per kernel
    ├── __init__.py           # registry: is_available / load / status
    ├── _spec.py              # manifest parser (build time + runtime)
    ├── flash_moe/            # one folder per kernel
    │   ├── __init__.py       # Python surface: op wrappers, fake tensors
    │   ├── mxfp8.py
    │   └── csrc/             # the C++/CUDA sources compiled into prime_kernels.flash_moe._C
    └── rmsnorm/
        ├── __init__.py
        ├── csrc/             # the torch binding
        ├── src/              # the kernel variants
        └── CMakeLists.txt    # + bench.cu, test.cu: a standalone harness, see its README
```

The repo root is the wheel: `setup.py` and `pyproject.toml` sit here, and `prime_kernels/`
is the package you import. A kernel folder holds both halves of one kernel — its Python
surface and, under `csrc/`, the sources compiled into `prime_kernels.<name>._C`.

This repo is consumed as a git submodule at `kernels/` in
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), which builds and publishes the
prebuilt wheels with its releases.

## Using a kernel

Kernels are compiled for specific compute capabilities and may not be built at all, so
never import one directly from application code:

```python
import prime_kernels

if prime_kernels.is_available("flash_moe"):
    flash_moe = prime_kernels.load("flash_moe")
    out = flash_moe.fused_moe_bf16(...)
```

`prime_kernels.status()` maps every kernel to `"available"` or the reason it is not.

`rmsnorm` fuses RMSNorm, the residual add and the MXFP8 quantization of the result into one
kernel, and returns the scales already in the blocked layout a tensor core GEMM reads. Only
its sources are committed for now — its table in `kernels.toml` is commented out, so it is
neither built nor shipped in the wheel, and the registry does not list it.

`flash_moe` is used by prime-rl's MoE layers under `model.moe_fused_kernel=true`, which
resolves the kernel during model setup so an unusable install fails before training starts.
It picks `fused_moe_mxfp8` when the run also quantizes the experts to MXFP8 and
`fused_moe_bf16` otherwise.

## Installing

prime-rl's `uv sync --extra kernels` installs the prebuilt wheels attached to a prime-rl
release, pinned in its root `[tool.uv.sources]`. Building from source is manual and always
explicit — no `uv sync` compiles CUDA:

```bash
uv pip install --no-build-isolation -e .
```

The build needs `nvcc` (`CUDA_HOME`) whose CUDA major matches torch's. Kernels whose toolkit
is unsuitable are skipped with a message rather than failing the build; the registry then
reports them unavailable. `PRIME_KERNELS=a,b` builds a subset, `PRIME_KERNELS_REQUIRE=1`
turns a skip into an error (prime-rl's release workflow sets it).

## Adding a kernel

1. Commit the sources under `prime_kernels/<name>/csrc/`. (Paths in the manifest are free
   form, so a kernel that comes with its own dev harness may keep that harness's layout —
   `rmsnorm` does, and puts only the torch binding in `csrc/`.)
2. Add a table to `prime_kernels/kernels.toml` (paths relative to the kernel folder):

```toml
[<name>]
description = "..."
ops = "<torch.ops namespace the extension registers>"
sources = ["csrc/foo.cu", "csrc/torch_interface.cpp"]
include-dirs = ["csrc"]
arch = ["10.0a"]       # compute capabilities to compile for; exact match at runtime
cxx-std = 20
```

3. Add `prime_kernels/<name>/__init__.py`: `from . import _C` plus wrappers and
   `torch.library.register_fake` for each op.

The extension is always named `prime_kernels.<name>._C`, so the C++ side must define
`PYBIND11_MODULE(_C, m)` (ops themselves should be registered with `TORCH_LIBRARY*`).
Two installs registering the same `torch.ops` namespace collide — if a kernel's sources are
also installed as a standalone package (e.g. `prime_moe`), uninstall it.
