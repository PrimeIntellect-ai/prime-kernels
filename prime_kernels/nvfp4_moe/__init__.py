from . import _C  # noqa: F401
from .grouped_gemm import grouped_gemm
from .quantize import quantize_activations, quantize_weights

__all__ = [
    "TOKEN_GROUP_ALIGNMENT",
    "grouped_gemm",
    "quantize_activations",
    "quantize_weights",
    "unsupported_shape_reason",
]

TOKEN_GROUP_ALIGNMENT = 32


def unsupported_shape_reason(dim: int, hidden_dim: int) -> str | None:
    if dim <= 0 or hidden_dim <= 0 or dim % 32 or hidden_dim % 32:
        return "NVFP4 expert dimensions must be positive multiples of 32"
    return None
