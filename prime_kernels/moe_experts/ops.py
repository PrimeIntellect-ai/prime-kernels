"""Differentiable expert MLP of a rank's local experts: ``down(clamped_swiglu(gate(x), up(x)))``.

Forward: one fused GEMM computes gate and up and applies the clamped SwiGLU, a second GEMM
the down projection. Only ``x``, ``gate`` and ``up`` are saved. Backward: one GEMM computes
``dh = dout @ down`` and turns it into ``dgate`` / ``dup`` (and recomputes ``h``), one GEMM
computes ``dx`` over both products. The weight gradients have nothing to fuse and stay with
``torch._grouped_mm`` (CUTLASS), which was faster than a Gluon version of them.

With ``fp8=True`` the GEMMs are DeepGEMM's blockwise FP8 ones, with the quantization fused into
the passes around them (see `fp8`); the forward saves the padded gate/up GEMM output, ``x``
quantized for the weight gradient and the transposed FP8 weights.
"""

import importlib.util

import torch

from prime_kernels.moe_experts import fp8 as fp8_impl
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


def unsupported_shape_reason(hidden_size: int, intermediate_size: int, fp8: bool = False) -> str | None:
    """Why these expert sizes cannot run, or None. Rows must be 16-byte aligned for TMA."""
    if fp8:
        return fp8_impl.unsupported_shape_reason(hidden_size, intermediate_size)
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


@torch.library.custom_op("prime_kernels::moe_experts_fp8_forward", mutates_args=())
def _moe_experts_fp8_fwd(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor | None,
    w2: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float,
    save_for_backward: bool,
) -> list[torch.Tensor]:
    """out, then what the backward reads (empty without ``save_for_backward``): the padded
    gate/up GEMM output, ``x`` quantized per column group and the transposed FP8 weights."""
    _check_groups(num_tokens_per_expert, x.shape[0])
    gate_proj, up_proj = _gate_up(w1, w3)
    intermediate = gate_proj.shape[1]
    if x.shape[0] == 0:
        return [x.new_empty(x.shape), *_fp8_saved(x, w2, save_for_backward)]
    with torch.cuda.device(x.device):
        layout = fp8_impl.Layout(num_tokens_per_expert, x.shape[0])
        x_q, x_sf, x_t, x_t_sf = fp8_impl.quantize_activation(x, layout, columns=save_for_backward)
        w13_q, w13_sf, w13_t, w13_t_sf = fp8_impl.quantize_weight(gate_proj, up_proj, transposed=save_for_backward)
        w2_q, w2_sf, w2_t, w2_t_sf = fp8_impl.quantize_weight(w2, None, transposed=save_for_backward)
        gate_up = fp8_impl.padded_gemm(x_q, x_sf, w13_q, w13_sf, layout, 2 * intermediate)
        del x_q, x_sf
        h_q, h_sf, h_head, h_head_sf = fp8_impl.swiglu_quantize(gate_up, layout, swiglu_limit)
        out = fp8_impl.token_gemm(h_q, h_sf, h_head, h_head_sf, w2_q, w2_sf, layout, x.shape[1])
    if not save_for_backward:
        return [out, *_fp8_saved(x, w2, save_for_backward)]
    return [out, gate_up, x_t, x_t_sf, w13_t, w13_t_sf, w2_t, w2_t_sf]


def _fp8_saved(x, w2, save_for_backward):
    """Zeros shaped like what the FP8 forward saves for backward (all empty without saving)."""
    num_experts, hidden, intermediate = w2.shape
    Mp = fp8_impl.padded_rows(x.shape[0], num_experts) if save_for_backward else 0
    G = fp8_impl.GROUP
    E = num_experts if save_for_backward else 0
    return [
        x.new_zeros(Mp, 2 * intermediate),
        x.new_zeros(Mp * hidden, dtype=torch.float8_e4m3fn),
        x.new_zeros(Mp // G, hidden, dtype=torch.float32),
        x.new_zeros(E, hidden, 2 * intermediate, dtype=torch.float8_e4m3fn),
        x.new_zeros(E, hidden // G, 2 * intermediate // G, dtype=torch.float32),
        x.new_zeros(E, intermediate, hidden, dtype=torch.float8_e4m3fn),
        x.new_zeros(E, intermediate // G, hidden // G, dtype=torch.float32),
    ]


@_moe_experts_fp8_fwd.register_fake
def _(x, w1, w3, w2, num_tokens_per_expert, swiglu_limit, save_for_backward):
    return [x.new_empty(x.shape), *_fp8_saved(x, w2, save_for_backward)]


@torch.library.custom_op("prime_kernels::moe_experts_fp8_backward", mutates_args=())
def _moe_experts_fp8_bwd(
    dout: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    gate_up: torch.Tensor,
    x_t: torch.Tensor,
    x_t_sf: torch.Tensor,
    w13_t: torch.Tensor,
    w13_t_sf: torch.Tensor,
    w2_t: torch.Tensor,
    w2_t_sf: torch.Tensor,
    swiglu_limit: float,
    packed: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """dx, dw1, dw3 (empty when w1 packs gate and up) and dw2."""
    dout = dout.contiguous()
    num_experts, hidden, two_intermediate = w13_t.shape
    intermediate = two_intermediate // 2
    if dout.shape[0] == 0:
        dw3 = dout.new_empty(0) if packed else dout.new_zeros(num_experts, intermediate, hidden)
        dw1 = dout.new_zeros(num_experts, intermediate * (2 if packed else 1), hidden)
        return dout.new_empty(dout.shape), dw1, dw3, dout.new_zeros(num_experts, hidden, intermediate)
    with torch.cuda.device(dout.device):
        layout = fp8_impl.Layout(num_tokens_per_expert, dout.shape[0])
        dout_q, dout_sf, dout_t, dout_t_sf = fp8_impl.quantize_activation(dout, layout, columns=True)
        dh = fp8_impl.padded_gemm(dout_q, dout_sf, w2_t, w2_t_sf, layout, intermediate)
        del dout_q, dout_sf
        dgu, dgu_t, h_t = fp8_impl.swiglu_backward_quantize(dh, gate_up, layout, swiglu_limit)
        del dh
        dx = fp8_impl.token_gemm(*dgu, w13_t, w13_t_sf, layout, hidden)
        del dgu
        dw2, _ = fp8_impl.weight_grad(dout_t, dout_t_sf, *h_t, layout, hidden, intermediate)
        dw1, dw3 = fp8_impl.weight_grad(
            *dgu_t, x_t, x_t_sf, layout, two_intermediate, hidden, split=None if packed else intermediate
        )
    if packed:
        dw3 = dx.new_empty(0)
    return dx, dw1, dw3, dw2


@_moe_experts_fp8_bwd.register_fake
def _(dout, num_tokens_per_expert, gate_up, x_t, x_t_sf, w13_t, w13_t_sf, w2_t, w2_t_sf, swiglu_limit, packed):
    num_experts, hidden, two_intermediate = w13_t.shape
    intermediate = two_intermediate // 2
    dx = dout.new_empty(dout.shape)
    dw2 = dout.new_empty(num_experts, hidden, intermediate)
    if packed:
        return dx, dout.new_empty(num_experts, two_intermediate, hidden), dout.new_empty(0), dw2
    dw = dout.new_empty(num_experts, intermediate, hidden)
    return dx, dw, dw.clone(), dw2


def _fp8_setup_context(ctx, inputs, output):
    _, w1, w3, _, num_tokens_per_expert, swiglu_limit, save_for_backward = inputs
    ctx.saved_for_backward = save_for_backward
    ctx.save_for_backward(num_tokens_per_expert, *output[1:])
    ctx.swiglu_limit = swiglu_limit
    ctx.packed = w3 is None
    ctx.set_materialize_grads(False)


def _fp8_backward(ctx, grads):
    if not ctx.saved_for_backward:
        raise RuntimeError("the FP8 expert forward ran without saving for backward but is being differentiated")
    num_tokens_per_expert, *saved = ctx.saved_tensors
    dx, dw1, dw3, dw2 = _moe_experts_fp8_bwd(grads[0], num_tokens_per_expert, *saved, ctx.swiglu_limit, ctx.packed)
    return dx, dw1, None if ctx.packed else dw3, dw2, None, None, None


_moe_experts_fp8_fwd.register_autograd(_fp8_backward, setup_context=_fp8_setup_context)


def moe_experts(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor | None,
    down_proj: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float,
    fp8: bool = False,
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
        fp8: run the GEMMs in blockwise FP8 (DeepSeek-V3 recipe, through DeepGEMM) for the
            forward, the data gradients and the weight gradients; needs hidden and intermediate
            sizes that are multiples of 128.

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
    reason = unsupported_shape_reason(x.shape[1], down_proj.shape[2], fp8=fp8)
    if reason is not None:
        raise ValueError(reason)
    if fp8:
        if importlib.util.find_spec("deep_gemm") is None:
            raise RuntimeError("moe_experts(fp8=True) requires DeepGEMM (deep_gemm), which is not installed")
        save = torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in (x, gate_proj, up_proj, down_proj)
        )
        out, *_ = _moe_experts_fp8_fwd(
            x, gate_proj, up_proj, down_proj, num_tokens_per_expert, float(swiglu_limit), save
        )
        return out
    out, _, _ = _moe_experts_fwd(x, gate_proj, up_proj, down_proj, num_tokens_per_expert, float(swiglu_limit))
    return out
