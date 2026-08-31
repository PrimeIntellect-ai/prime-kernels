from __future__ import annotations

import torch

from . import _C  # noqa: F401
from prime_kernels.indexed_attention.forward import indexed_attention, unsupported_shape_reason
from prime_kernels.indexed_attention.selection import select_indexed_blocks

__all__ = ["indexed_attention", "select_indexed_blocks", "unsupported_shape_reason"]


@torch.library.register_fake("prime_indexed_attention::persistent_topk")
def _persistent_topk_fake(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    k: int,
    max_seq_len: int,
) -> None:
    return None
