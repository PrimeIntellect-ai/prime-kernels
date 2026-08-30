# prime-kernels

CUDA kernels for Prime Intellect training stacks, shipped as one wheel, `prime-kernels`.

```
.
├── setup.py                  # builds what the manifest declares; no edit needed to add a kernel
└── prime_kernels/
    ├── kernels.toml          # the manifest: one table per kernel
    ├── __init__.py           # registry: is_available / load / status
    ├── _spec.py              # manifest parser (build time + runtime)
    ├── flash_moe/            # compiled kernel
    │   ├── __init__.py       # Python surface: op wrappers, fake tensors
    │   ├── mxfp8.py
    │   └── csrc/             # the C++/CUDA sources compiled into prime_kernels.flash_moe._C
    ├── indexed_attention/    # Python-only TileLang indexed GQA forward + backward
    ├── mxfp8_moe/            # Python-only MXFP8 MoE runtime kernels
    └── rmsnorm/
        ├── __init__.py
        ├── csrc/             # the torch binding
        ├── src/              # the kernel variants
        └── CMakeLists.txt    # + bench.cu, test.cu: a standalone harness, see its README
```

The repo root is the wheel: `setup.py` and `pyproject.toml` sit here, and `prime_kernels/`
is the package you import. A kernel folder holds both halves of one kernel — its Python
surface and, for compiled kernels, the sources under `csrc/` compiled into
`prime_kernels.<name>._C`.

This repo is consumed as a git submodule at `deps/prime-kernels/` in
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

`flash_moe` is currently dormant in prime-rl. `mxfp8_moe` provides differentiable MXFP8
grouped GEMM and MXFP8 expert-parallel transport. It is registered as Python-only because
it orchestrates PyTorch and torchao kernels rather than compiling a `_C` extension here.
`indexed_attention` provides differentiable grouped-query attention over an explicit token
selection for each query. Its TileLang kernels accept different query and KV lengths so the
caller can gather KV for context parallelism without gathering queries.

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

3. Add `prime_kernels/<name>/__init__.py`: `from . import _C` plus, per op, a wrapper
   calling `torch.ops.<ns>.<op>` and a `torch.library.register_fake`. No
   `torch.library.custom_op` decorator — that is how a *Python* op is defined, and
   `TORCH_LIBRARY` has already defined these ops C++ side; only the fake (meta) kernel is
   missing, since C++ registers no meta implementation. An op used in training also needs
   `torch.library.register_autograd`: a schema carries no backward, so without it autograd
   treats the op as non-differentiable. `flash_moe` is the exception — it is forward only,
   and prime-rl wraps it in its own `autograd.Function`.

For a Python-only kernel, set `python-only = true`, omit `ops` and `sources`, and expose
the differentiable Python surface from `__init__.py`. Optional import requirements belong
in the manifest's `requires` list so `is_available()` fails during setup. Python-only ops
may use `torch.library.custom_op`; register fake and autograd implementations so they remain
visible to `torch.compile` and training.

Whatever the kernel requires of its inputs — block sizes, alignments, layouts — belongs
here, not in the caller: `TORCH_CHECK` it in the binding, and export the constants
(e.g. `flash_moe.BLOCK_M`) and any setup-time predicate (`unsupported_shape_reason`) from
the kernel's `__init__.py`. A caller hardcoding `128` means every requirement change is a
two-repo change.

The extension is always named `prime_kernels.<name>._C`, so the C++ side must define
`PYBIND11_MODULE(_C, m)` (ops themselves should be registered with `TORCH_LIBRARY*`).
Two installs registering the same `torch.ops` namespace collide — if a kernel's sources are
also installed as a standalone package (e.g. `prime_moe`), uninstall it.
