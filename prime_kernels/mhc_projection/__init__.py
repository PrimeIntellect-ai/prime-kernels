import torch

from prime_kernels.mhc_projection.gates import mhc_gates
from prime_kernels.mhc_projection.projection import mhc_projection


def hyper_connection(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    pre_mix: torch.Tensor | None = None,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """DeepSeek-V4.1 mHC gates for one sublayer, plus the collapse of its input streams.

    `x`: `(..., hc, d)` bf16 streams. `weight`: `((2 + hc) * hc, hc * d)`, `scale`: `(3,)`,
    `base`: `((2 + hc) * hc,)`. `pre_mix`: `(..., hc)`, the gate collapsing these streams (V4.1
    computes it one sublayer earlier). Returns fp32 `pre`, `post`, `comb` and the collapsed
    `(..., d)` sequence in `x.dtype` (None without `pre_mix`).
    """
    mixes, collapsed = mhc_projection(x, weight, pre_mix, rms_eps)
    pre, post, comb = mhc_gates(mixes, scale, base, x.shape[-2], sinkhorn_iters, hc_eps)
    return pre, post, comb, collapsed


__all__ = ["hyper_connection", "mhc_gates", "mhc_projection"]
