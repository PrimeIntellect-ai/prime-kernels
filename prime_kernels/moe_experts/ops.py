"""Differentiable expert MLP of a rank's local experts: ``down(clamped_swiglu(gate(x), up(x)))``.

Forward: one fused GEMM computes gate and up and applies the clamped SwiGLU, a second GEMM
the down projection. Only ``x``, ``gate`` and ``up`` are saved. Backward: one GEMM computes
``dh = dout @ down`` and turns it into ``dgate`` / ``dup`` (and recomputes ``h``), one GEMM
computes ``dx`` over both products. The weight gradients have nothing to fuse and stay with
``torch._grouped_mm`` (CUTLASS), which was faster than a Gluon version of them.
"""

import torch

from prime_kernels.moe_experts import gemm
from prime_kernels.moe_experts.kernels import TileTable, zero_tail

# Every expert's group of rows must be a multiple of this many rows long (prime-rl's dispatcher
# pads the groups to its experts' token_group_alignment): torch._grouped_mm needs 16-byte aligned
# group starts along its K dimension for the weight gradients.
TOKEN_GROUP_ALIGNMENT = 8

_NUM_SMS: dict[int, int] = {}


def _num_sms(device: torch.device) -> int:
    if device.index not in _NUM_SMS:
        _NUM_SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS[device.index]


def unsupported_shape_reason(hidden_size: int, intermediate_size: int) -> str | None:
    """Why these expert sizes cannot run, or None. Rows must be 16-byte aligned for TMA."""
    if hidden_size % 8 or intermediate_size % 8:
        return f"hidden ({hidden_size}) and intermediate ({intermediate_size}) sizes must be multiples of 8"
    return None


def _gate_up(w1: torch.Tensor, w3: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if w3 is not None:
        return w1, w3
    intermediate = w1.shape[1] // 2
    return w1[:, :intermediate], w1[:, intermediate:]


def _check_groups(num_tokens_per_expert: torch.Tensor, num_rows: int) -> None:
    # Asynchronous device asserts: the group sizes are device data, and reading them on the host
    # would synchronize every MoE layer.
    torch._assert_async((num_tokens_per_expert % TOKEN_GROUP_ALIGNMENT == 0).all())
    torch._assert_async(num_tokens_per_expert.sum() <= num_rows)


@torch.library.custom_op("prime_kernels::moe_experts_forward", mutates_args=())
def _moe_experts_fwd(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor | None,
    w2: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_proj, up_proj = _gate_up(w1, w3)
    _check_groups(num_tokens_per_expert, x.shape[0])
    if x.shape[0] == 0:
        empty = x.new_empty(0, gate_proj.shape[1])
        return x.new_empty(x.shape), empty, empty.clone()
    with torch.cuda.device(x.device):
        num_sms = _num_sms(x.device)
        table = TileTable(num_tokens_per_expert, x.shape[0], gemm.BLOCK_M)
        gate, up, h = gemm.fc1(x, gate_proj, up_proj, table, swiglu_limit, num_sms=num_sms)
        out = gemm.down(h, w2, table, num_sms=num_sms)
        zero_tail(out, num_tokens_per_expert, num_sms)
    return out, gate, up


@_moe_experts_fwd.register_fake
def _(x, w1, w3, w2, num_tokens_per_expert, swiglu_limit):
    intermediate = w2.shape[2]
    return x.new_empty(x.shape), x.new_empty(x.shape[0], intermediate), x.new_empty(x.shape[0], intermediate)


@torch.library.custom_op("prime_kernels::moe_experts_backward", mutates_args=())
def _moe_experts_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor | None,
    w2: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    swiglu_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """dx, dw1, dw3 (empty when w1 packs gate and up) and dw2."""
    gate_proj, up_proj = _gate_up(w1, w3)
    dout = dout.contiguous()
    intermediate = gate.shape[1]
    if x.shape[0] == 0:
        dw3 = x.new_empty(0) if w3 is None else x.new_zeros(w3.shape)
        return x.new_empty(x.shape), x.new_zeros(w1.shape), dw3, x.new_zeros(w2.shape)
    with torch.cuda.device(x.device):
        num_sms = _num_sms(x.device)
        table = TileTable(num_tokens_per_expert, x.shape[0], gemm.BLOCK_M)
        dgate_dup, h = gemm.dswiglu(dout, w2, gate, up, table, swiglu_limit, num_sms=num_sms)
        dgate, dup = dgate_dup[:, :intermediate], dgate_dup[:, intermediate:]
        dx = gemm.dx(dgate, dup, gate_proj, up_proj, table, num_sms=num_sms)
        zero_tail(dx, num_tokens_per_expert, num_sms)
        offsets = torch.cumsum(num_tokens_per_expert, 0, dtype=torch.int32)
        dw2 = torch._grouped_mm(dout.t(), h, offs=offsets)
        if w3 is None:
            # [E, 2 * I, H] straight from the two halves of dgate_dup: the packed weight's layout.
            dw1 = torch._grouped_mm(dgate_dup.t(), x, offs=offsets)
            dw3 = x.new_empty(0)
        else:
            dw1 = torch._grouped_mm(dgate.t(), x, offs=offsets)
            dw3 = torch._grouped_mm(dup.t(), x, offs=offsets)
    return dx, dw1, dw3, dw2


@_moe_experts_bwd.register_fake
def _(dout, x, w1, w3, w2, num_tokens_per_expert, gate, up, swiglu_limit):
    dw3 = x.new_empty(0) if w3 is None else x.new_empty(w3.shape)
    return x.new_empty(x.shape), x.new_empty(w1.shape), dw3, x.new_empty(w2.shape)


def _setup_context(ctx, inputs, output):
    x, w1, w3, w2, num_tokens_per_expert, swiglu_limit = inputs
    _, gate, up = output
    ctx.save_for_backward(x, w1, w3, w2, num_tokens_per_expert, gate, up)
    ctx.swiglu_limit = swiglu_limit
    # gate and up are only returned to be saved; without this autograd would fill zero gradients
    # for them on every backward.
    ctx.set_materialize_grads(False)


def _backward(ctx, dout, _dgate, _dup):
    x, w1, w3, w2, num_tokens_per_expert, gate, up = ctx.saved_tensors
    dx, dw1, dw3, dw2 = _moe_experts_bwd(dout, x, w1, w3, w2, num_tokens_per_expert, gate, up, ctx.swiglu_limit)
    return dx, dw1, dw3 if w3 is not None else None, dw2, None, None


_moe_experts_fwd.register_autograd(_backward, setup_context=_setup_context)


def moe_experts(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor | None,
    down_proj: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Expert MLP over tokens grouped by expert, differentiable in ``x`` and the three weights.

    For expert ``e`` and its rows ``x_e``: ``down_proj[e] @ (silu(min(gate, l)) * clamp(up, -l,
    l))`` with ``gate = gate_proj[e] @ x_e`` and ``up = up_proj[e] @ x_e``, both rounded to bf16
    first; routing scores are applied by the caller.

    Args:
        x: ``[rows, hidden]`` bf16, contiguous. Expert ``e`` owns the ``num_tokens_per_expert[e]``
            rows after those of experts ``< e``; rows after the last group come out as zeros.
        gate_proj, up_proj: ``[E, intermediate, hidden]`` bf16. Pass ``up_proj=None`` and the
            packed ``[E, 2 * intermediate, hidden]`` weight (gate rows first) as ``gate_proj``.
        down_proj: ``[E, hidden, intermediate]`` bf16, contiguous.
        num_tokens_per_expert: ``[E]`` integer group sizes on the device, each a multiple of
            ``TOKEN_GROUP_ALIGNMENT``, summing to at most ``rows`` (checked asynchronously).
        swiglu_limit: the clamp ``l``.

    Returns:
        ``[rows, hidden]`` bf16.
    """
    for name, tensor in (("x", x), ("gate_proj", gate_proj), ("up_proj", up_proj), ("down_proj", down_proj)):
        if tensor is None:
            continue
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
            raise ValueError(f"{name} must be a bfloat16 CUDA tensor, got {tensor.dtype} on {tensor.device}")
    if not x.is_contiguous() or not down_proj.is_contiguous():
        raise ValueError("x and down_proj must be contiguous")
    reason = unsupported_shape_reason(x.shape[1], down_proj.shape[2])
    if reason is not None:
        raise ValueError(reason)
    out, _, _ = _moe_experts_fwd(x, gate_proj, up_proj, down_proj, num_tokens_per_expert, float(swiglu_limit))
    return out
