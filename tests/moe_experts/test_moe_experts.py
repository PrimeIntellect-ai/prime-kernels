import pytest
import torch
import torch.nn.functional as F

import prime_kernels

LIMIT = 10.0


@pytest.fixture(scope="module")
def moe_experts():
    reason = prime_kernels.unavailable_reason("moe_experts")
    if reason is not None:
        pytest.skip(reason)
    return prime_kernels.load("moe_experts")


def make_problem(num_experts, tokens_per_expert, hidden, intermediate, *, tail_rows, packed, seed, alignment):
    """Grouped tokens and weights scaled so that roughly 10% of the gates exceed the clamp."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    counts = torch.randint(
        int(tokens_per_expert * 0.75),
        int(tokens_per_expert * 1.25) + 1,
        (num_experts,),
        device="cuda",
        generator=generator,
    )
    counts = (counts + alignment - 1) // alignment * alignment
    if num_experts > 2:
        counts[1] = 0
    rows = int(counts.sum()) + tail_rows
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    x[rows - tail_rows :] = 0
    scale = 7.0 / hidden**0.5
    if packed:
        gate_up = torch.randn(num_experts, 2 * intermediate, hidden, device="cuda", generator=generator) * scale
        gate_proj = gate_up.bfloat16().requires_grad_()
        up_proj = None
    else:
        gate_proj = torch.randn(num_experts, intermediate, hidden, device="cuda", generator=generator) * scale
        up_proj = torch.randn(num_experts, intermediate, hidden, device="cuda", generator=generator) * scale
        gate_proj = gate_proj.bfloat16().requires_grad_()
        up_proj = up_proj.bfloat16().requires_grad_()
    down_proj = torch.randn(num_experts, hidden, intermediate, device="cuda", generator=generator) / intermediate**0.5
    down_proj = down_proj.bfloat16().requires_grad_()
    dout = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    return x.requires_grad_(), gate_proj, up_proj, down_proj, counts, dout


def reference(x, gate_proj, up_proj, down_proj, counts, dtype):
    """Per-expert loop in ``dtype``; rows after the last group give zeros."""
    if up_proj is None:
        gate_proj, up_proj = gate_proj.chunk(2, dim=1)
    out = torch.zeros(x.shape, device=x.device, dtype=dtype)
    start = 0
    for expert, count in enumerate(counts.tolist()):
        rows = slice(start, start + count)
        xe = x[rows].to(dtype)
        gate = xe @ gate_proj[expert].to(dtype).T
        up = xe @ up_proj[expert].to(dtype).T
        h = F.silu(gate.clamp(max=LIMIT)) * up.clamp(min=-LIMIT, max=LIMIT)
        out[rows] = h @ down_proj[expert].to(dtype).T
        start += count
    return out


def grads(out, dout, tensors):
    return torch.autograd.grad(out, tensors, dout.to(out.dtype))


def rel_err(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item()


CASES = {
    "small": dict(num_experts=3, tokens_per_expert=300, hidden=256, intermediate=192, tail_rows=40),
    "odd": dict(num_experts=5, tokens_per_expert=130, hidden=136, intermediate=264, tail_rows=0),
    "v41_ep96": dict(num_experts=4, tokens_per_expert=2048, hidden=5120, intermediate=2304, tail_rows=64),
    "v41_ep8": dict(num_experts=48, tokens_per_expert=256, hidden=5120, intermediate=2304, tail_rows=0),
}


@pytest.mark.parametrize("packed", [False, True], ids=["separate", "packed"])
@pytest.mark.parametrize("case", list(CASES))
def test_matches_fp32_reference(moe_experts, case, packed):
    x, gate_proj, up_proj, down_proj, counts, dout = make_problem(
        **CASES[case], packed=packed, seed=0, alignment=moe_experts.TOKEN_GROUP_ALIGNMENT
    )
    weights = [w for w in (gate_proj, up_proj, down_proj) if w is not None]
    tensors = [x, *weights]

    out = moe_experts.moe_experts(x, gate_proj, up_proj, down_proj, counts, LIMIT)
    actual = [out, *grads(out, dout, tensors)]
    expected_out = reference(x, gate_proj, up_proj, down_proj, counts, torch.float32)
    expected = [expected_out, *grads(expected_out, dout, tensors)]
    # The unfused bf16 computation (today's path), which rounds gate, up, h and dh to bf16 too.
    bf16_out = reference(x, gate_proj, up_proj, down_proj, counts, torch.bfloat16)
    bf16 = [bf16_out, *grads(bf16_out, dout, tensors)]

    names = ["out", "dx"] + [
        f"d{name}" for name, w in zip(("gate", "up", "down"), (gate_proj, up_proj, down_proj)) if w is not None
    ]
    for name, a, e, b in zip(names, actual, expected, bf16):
        assert a.shape == e.shape and a.dtype == torch.bfloat16, name
        assert torch.isfinite(a).all(), name
        err, bf16_err = rel_err(a, e), rel_err(b, e)
        # Gradients through the gate and up projections pass the clamps, whose masks flip for the
        # values bf16 rounding moves across the limit: with ~10% of values clamped that alone is a
        # ~4% relative difference from fp32, for the unfused bf16 computation as well.
        tolerance = 1e-2 if name in ("out", "ddown") else 6e-2
        assert err < tolerance, f"{name}: rel err {err:.2e} vs fp32"
        assert err < 1.05 * bf16_err + 1e-3, f"{name}: rel err {err:.2e}, unfused bf16 {bf16_err:.2e}"
    tail = int(counts.sum())
    assert (out[tail:] == 0).all() and (actual[1][tail:] == 0).all()


def test_clamp_is_exercised(moe_experts):
    x, gate_proj, up_proj, _, counts, _ = make_problem(
        **CASES["v41_ep96"], packed=False, seed=0, alignment=moe_experts.TOKEN_GROUP_ALIGNMENT
    )
    gate = x[: counts[0]].float() @ gate_proj[0].float().T
    up = x[: counts[0]].float() @ up_proj[0].float().T
    assert 0.02 < (gate > LIMIT).float().mean() < 0.5
    assert 0.02 < (up.abs() > LIMIT).float().mean() < 0.5


def test_torch_compile(moe_experts):
    x, gate_proj, up_proj, down_proj, counts, dout = make_problem(
        **CASES["small"], packed=False, seed=1, alignment=moe_experts.TOKEN_GROUP_ALIGNMENT
    )

    def block(x, gate_proj, up_proj, down_proj, counts):
        return moe_experts.moe_experts(x * 2, gate_proj, up_proj, down_proj, counts, LIMIT) + 1

    compiled = torch.compile(block, fullgraph=True)
    tensors = [x, gate_proj, up_proj, down_proj]
    out = compiled(x, gate_proj, up_proj, down_proj, counts)
    eager = block(x, gate_proj, up_proj, down_proj, counts)
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
    for a, e in zip(grads(out, dout, tensors), grads(eager, dout, tensors)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_opcheck(moe_experts):
    x, gate_proj, up_proj, down_proj, counts, _ = make_problem(
        **CASES["small"], packed=False, seed=2, alignment=moe_experts.TOKEN_GROUP_ALIGNMENT
    )
    torch.library.opcheck(
        torch.ops.prime_kernels.moe_experts_forward.default,
        (x, gate_proj, up_proj, down_proj, counts, LIMIT),
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )


def test_unsupported_shape_reason(moe_experts):
    assert moe_experts.unsupported_shape_reason(5120, 2304) is None
    assert moe_experts.unsupported_shape_reason(5120, 2300) is not None


@pytest.mark.parametrize("rows", [0, 64])
def test_no_routed_tokens(moe_experts, rows):
    num_experts, hidden, intermediate = 3, 256, 192
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate_proj, up_proj = (
        torch.randn(num_experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    )
    down_proj = torch.randn(num_experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.zeros(num_experts, dtype=torch.int64, device="cuda")
    out = moe_experts.moe_experts(x, gate_proj, up_proj, down_proj, counts, LIMIT)
    assert out.shape == x.shape and (out == 0).all()
    dx, *dws = grads(out, torch.randn_like(out), [x, gate_proj, up_proj, down_proj])
    assert (dx == 0).all() and all((dw == 0).all() for dw in dws)


# FP8 (moe_experts(..., fp8=True)): blockwise e4m3, DeepSeek-V3 recipe through DeepGEMM.


@pytest.fixture(scope="module")
def moe_experts_fp8(moe_experts):
    pytest.importorskip("deep_gemm")
    return moe_experts


FP8_CASES = {
    "small": dict(num_experts=3, tokens_per_expert=300, hidden=256, intermediate=384, tail_rows=40),
    "odd": dict(num_experts=5, tokens_per_expert=90, hidden=384, intermediate=256, tail_rows=24),
    "v41_ep96": CASES["v41_ep96"],
    "v41_ep8": CASES["v41_ep8"],
}


def _fp8(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).float()


def quant_groups(x: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize per 1 x 128 group of the last dimension (zero-padded groups)."""
    rows, cols = x.shape
    groups = F.pad(x.float(), (0, -cols % 128)).view(rows, -1, 128)
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
    return (_fp8(groups / scale) * scale).view(rows, -1)[:, :cols]


def quant_blocks(w: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize per 128 x 128 block of the last two dimensions."""
    E, R, C = w.shape
    blocks = w.float().view(E, R // 128, 128, C // 128, 128)
    scale = blocks.abs().amax((2, 4), keepdim=True).clamp_min(1e-4) / 448
    return (_fp8(blocks / scale) * scale).view(E, R, C)


def fp8_emulation(x, gate_proj, up_proj, down_proj, counts, dout):
    """out and the gradients of x and the weights, computed the way the FP8 path does: FP8
    operands (per-token groups along K for activations and gradients, per 128-token groups of
    each channel for the weight-gradient operands, 128 x 128 blocks for weights), exact products
    with fp32 accumulation, GEMM outputs rounded to bf16, SwiGLU math in fp32."""
    if up_proj is None:
        gate_proj, up_proj = gate_proj.chunk(2, dim=1)
    w13 = quant_blocks(torch.cat([gate_proj, up_proj], dim=1).detach())
    w2 = quant_blocks(down_proj.detach())
    intermediate = gate_proj.shape[1]
    out, dx = torch.zeros_like(x, dtype=torch.float32), torch.zeros_like(x, dtype=torch.float32)
    dw13, dw2 = torch.zeros_like(w13), torch.zeros_like(w2)
    start = 0
    for e, count in enumerate(counts.tolist()):
        rows = slice(start, start + count)
        start += count
        if count == 0:
            continue
        xe, de = x[rows].detach(), dout[rows]
        gate_up = (quant_groups(xe) @ w13[e].T).bfloat16().float()
        gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
        gate_c, up_c = gate.clamp(max=LIMIT), up.clamp(-LIMIT, LIMIT)
        sig = torch.sigmoid(gate_c)
        h = gate_c * sig * up_c
        out[rows] = (quant_groups(h) @ w2[e].T).bfloat16().float()
        dh = (quant_groups(de) @ w2[e]).bfloat16().float()
        dgate = torch.where(gate <= LIMIT, dh * up_c * sig * (1 + gate_c * (1 - sig)), 0.0)
        dup = torch.where(up.abs() <= LIMIT, dh * gate_c * sig, 0.0)
        dgu = torch.cat([dgate, dup], dim=1)
        dx[rows] = (quant_groups(dgu) @ w13[e]).bfloat16().float()
        dw2[e] = quant_groups(de.T.float()) @ quant_groups(h.T).T
        dw13[e] = quant_groups(dgu.T) @ quant_groups(xe.T.float()).T
    grads = [dx, dw13[:, :intermediate], dw13[:, intermediate:], dw2]
    return out, grads


@pytest.mark.parametrize("packed", [False, True], ids=["separate", "packed"])
@pytest.mark.parametrize("case", list(FP8_CASES))
def test_fp8_matches_reference(moe_experts_fp8, case, packed):
    x, gate_proj, up_proj, down_proj, counts, dout = make_problem(
        **FP8_CASES[case], packed=packed, seed=0, alignment=moe_experts_fp8.TOKEN_GROUP_ALIGNMENT
    )
    weights = [w for w in (gate_proj, up_proj, down_proj) if w is not None]
    tensors = [x, *weights]
    out = moe_experts_fp8.moe_experts(x, gate_proj, up_proj, down_proj, counts, LIMIT, fp8=True)
    actual = [out, *grads(out, dout, tensors)]
    expected_out = reference(x, gate_proj, up_proj, down_proj, counts, torch.float32)
    expected = [expected_out, *grads(expected_out, dout, tensors)]
    emulated_out, emulated_grads = fp8_emulation(x, gate_proj, up_proj, down_proj, counts, dout)
    dx, dgate, dup, ddown = emulated_grads
    emulated = [emulated_out, dx, torch.cat([dgate, dup], 1) if packed else dgate, *([] if packed else [dup]), ddown]

    names = ["out", "dx"] + [
        f"d{name}" for name, w in zip(("gate", "up", "down"), (gate_proj, up_proj, down_proj)) if w is not None
    ]
    for name, a, e, m in zip(names, actual, expected, emulated):
        assert a.shape == e.shape and a.dtype == torch.bfloat16, name
        assert torch.isfinite(a).all(), name
        err, emulated_err = rel_err(a, e), rel_err(m, e)
        # FP8 rounding (~6% per GEMM operand pair here) moves more values across the clamps than
        # bf16 does, so the gradients through gate and up are further from fp32 (see the bf16 test).
        tolerance = 8e-2 if name in ("out", "ddown") else 1.6e-1
        assert err < tolerance, f"{name}: rel err {err:.2e} vs fp32"
        assert err < 1.02 * emulated_err + 1e-3, f"{name}: rel err {err:.2e}, FP8 emulation {emulated_err:.2e}"
        # Against the emulation of the same recipe only rounding-order differences remain (and the
        # clamp decisions they flip).
        assert rel_err(a, m) < 2e-2, f"{name}: rel err {rel_err(a, m):.2e} vs FP8 emulation"
    tail = int(counts.sum())
    assert (out[tail:] == 0).all() and (actual[1][tail:] == 0).all()


def test_fp8_without_grad_matches(moe_experts_fp8):
    x, gate_proj, up_proj, down_proj, counts, _ = make_problem(
        **FP8_CASES["small"], packed=False, seed=3, alignment=moe_experts_fp8.TOKEN_GROUP_ALIGNMENT
    )
    with_grad = moe_experts_fp8.moe_experts(x, gate_proj, up_proj, down_proj, counts, LIMIT, fp8=True)
    with torch.no_grad():
        without = moe_experts_fp8.moe_experts(x, gate_proj, up_proj, down_proj, counts, LIMIT, fp8=True)
    torch.testing.assert_close(with_grad, without, rtol=0, atol=0)


def test_fp8_torch_compile(moe_experts_fp8):
    x, gate_proj, up_proj, down_proj, counts, dout = make_problem(
        **FP8_CASES["small"], packed=False, seed=1, alignment=moe_experts_fp8.TOKEN_GROUP_ALIGNMENT
    )

    def block(x, gate_proj, up_proj, down_proj, counts):
        return moe_experts_fp8.moe_experts(x * 2, gate_proj, up_proj, down_proj, counts, LIMIT, fp8=True) + 1

    compiled = torch.compile(block, fullgraph=True)
    tensors = [x, gate_proj, up_proj, down_proj]
    out = compiled(x, gate_proj, up_proj, down_proj, counts)
    eager = block(x, gate_proj, up_proj, down_proj, counts)
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
    for a, e in zip(grads(out, dout, tensors), grads(eager, dout, tensors)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("save_for_backward", [True, False])
def test_fp8_opcheck(moe_experts_fp8, save_for_backward):
    x, gate_proj, up_proj, down_proj, counts, _ = make_problem(
        **FP8_CASES["small"], packed=False, seed=2, alignment=moe_experts_fp8.TOKEN_GROUP_ALIGNMENT
    )
    # No test_schema: it does arithmetic on the outputs, which torch lacks for FP8 tensors.
    torch.library.opcheck(
        torch.ops.prime_kernels.moe_experts_fp8_forward.default,
        (x, gate_proj, up_proj, down_proj, counts, LIMIT, save_for_backward),
        test_utils=("test_faketensor", "test_autograd_registration"),
    )


def test_fp8_unsupported_shape_reason(moe_experts_fp8):
    assert moe_experts_fp8.unsupported_shape_reason(5120, 2304, fp8=True) is None
    assert moe_experts_fp8.unsupported_shape_reason(5120, 2312, fp8=True) is not None


@pytest.mark.parametrize("rows", [0, 64])
def test_fp8_no_routed_tokens(moe_experts_fp8, rows):
    num_experts, hidden, intermediate = 3, 256, 128
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate_proj, up_proj = (
        torch.randn(num_experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    )
    down_proj = torch.randn(num_experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.zeros(num_experts, dtype=torch.int64, device="cuda")
    out = moe_experts_fp8.moe_experts(x, gate_proj, up_proj, down_proj, counts, LIMIT, fp8=True)
    assert out.shape == x.shape and (out == 0).all()
    dx, *dws = grads(out, torch.randn_like(out), [x, gate_proj, up_proj, down_proj])
    assert (dx == 0).all() and all((dw == 0).all() for dw in dws)
