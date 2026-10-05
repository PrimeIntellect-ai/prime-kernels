import pytest
import torch

import prime_kernels

REASON = prime_kernels.unavailable_reason("mhc_projection")
pytestmark = pytest.mark.skipif(REASON is not None, reason=REASON or "")

RMS_EPS = 1e-20
HC_EPS = 1e-6
ITERS = 20

# (tokens, hc, d): V4.1 Flash, then small shapes including ragged token and d tails.
SHAPES = [(4096, 4, 5120), (1, 4, 64), (37, 4, 320), (130, 4, 200), (77, 2, 192)]


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def mhc():
    return prime_kernels.load("mhc_projection")


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30)).item()


def reference_projection(x, weight, pre_mix, eps):
    """Plain torch in fp64, so it is the exact value the fp32 kernel approximates. The weight is
    rounded to bf16 first, which is the kernel's (and prime-rl's) contract."""
    xf = x.double()
    flat = xf.flatten(-2)
    rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    mixes = (flat @ weight.to(torch.bfloat16).double().t()) * rstd
    collapsed = (pre_mix.double().unsqueeze(-1) * xf).sum(-2) if pre_mix is not None else None
    return mixes.float(), collapsed.float() if collapsed is not None else None


def reference_gates(mixes, scale, base, hc, iters, eps):
    pre_m, post_m, comb_m = mixes.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre = torch.sigmoid(pre_m * scale[0] + pre_b) + eps
    post = 2 * torch.sigmoid(post_m * scale[1] + post_b)
    comb = (comb_m * scale[2] + comb_b).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def make_inputs(tokens, hc, d, with_pre=True):
    n = (2 + hc) * hc
    x = torch.randn(tokens, hc, d, device="cuda").to(torch.bfloat16).requires_grad_()
    weight = (torch.randn(n, hc * d, device="cuda") * (hc * d) ** -0.5).requires_grad_()
    pre_mix = (torch.rand(tokens, hc, device="cuda") + 0.1).requires_grad_() if with_pre else None
    scale = (torch.rand(3, device="cuda") + 0.5).requires_grad_()
    base = (torch.randn(n, device="cuda") * 0.5).requires_grad_()
    return x, weight, pre_mix, scale, base


def grads(outputs, inputs, cotangents):
    tensors = [t for t in inputs if t is not None]
    found = torch.autograd.grad(outputs, tensors, cotangents)
    return iter(found)


@pytest.mark.parametrize("with_pre", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
def test_projection_matches_reference(mhc, shape, with_pre):
    x, weight, pre_mix, _, _ = make_inputs(*shape, with_pre=with_pre)
    mixes, collapsed = mhc.mhc_projection(x, weight, pre_mix, RMS_EPS)
    ref_mixes, ref_collapsed = reference_projection(x, weight, pre_mix, RMS_EPS)

    # bf16 x bf16 products are exact in fp32; only the accumulation order differs.
    assert rel_err(mixes, ref_mixes) < 1e-5
    outputs, cotangents = [mixes], [torch.randn_like(mixes)]
    ref_outputs = [ref_mixes]
    if with_pre:
        # Only the final bf16 rounding separates the two.
        assert rel_err(collapsed.float(), ref_collapsed) < 3e-3
        outputs.append(collapsed)
        cotangents.append(torch.randn_like(collapsed))
        ref_outputs.append(ref_collapsed)

    inputs = (x, weight, pre_mix)
    got = grads(outputs, inputs, cotangents)
    ref = grads(ref_outputs, inputs, cotangents)
    # The kernel rounds d(projection) to bf16 for the tensor-core GEMMs (as prime-rl's eager
    # path does); dx itself is stored in bf16.
    assert rel_err(next(got), next(ref)) < 5e-3  # x
    assert rel_err(next(got), next(ref)) < 5e-3  # weight
    if with_pre:
        assert rel_err(next(got), next(ref)) < 1e-5  # pre_mix


@pytest.mark.parametrize("tokens", [1, 333, 16384])
def test_gates_match_reference(mhc, tokens):
    hc = 4
    mixes = (torch.randn(tokens, (2 + hc) * hc, device="cuda") * 2).requires_grad_()
    scale = (torch.rand(3, device="cuda") + 0.5).requires_grad_()
    base = (torch.randn((2 + hc) * hc, device="cuda") * 0.5).requires_grad_()
    got = mhc.mhc_gates(mixes, scale, base, hc, ITERS, HC_EPS)
    ref = reference_gates(mixes, scale, base, hc, ITERS, HC_EPS)
    for actual, expected in zip(got, ref):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    cotangents = [torch.randn_like(t) for t in ref]
    got_grads = torch.autograd.grad(got, (mixes, scale, base), cotangents)
    ref_grads = torch.autograd.grad(ref, (mixes, scale, base), cotangents)
    for actual, expected in zip(got_grads, ref_grads):
        assert rel_err(actual, expected) < 1e-4


def test_hyper_connection_matches_reference_under_compile(mhc):
    tokens, hc, d = 256, 4, 512
    x, weight, pre_mix, scale, base = make_inputs(tokens, hc, d)
    x3 = x.detach().view(2, tokens // 2, hc, d).requires_grad_()
    pre3 = pre_mix.detach().view(2, tokens // 2, hc).requires_grad_()
    kwargs = dict(rms_eps=RMS_EPS, hc_eps=HC_EPS, sinkhorn_iters=ITERS)

    compiled = torch.compile(mhc.hyper_connection, fullgraph=True)
    got = compiled(x3, weight, scale, base, pre3, **kwargs)
    mixes, collapsed = reference_projection(x3, weight, pre3, RMS_EPS)
    ref = (*reference_gates(mixes, scale, base, hc, ITERS, HC_EPS), collapsed)
    assert [t.shape for t in got] == [t.shape for t in ref]
    for actual, expected in zip(got, ref):
        assert rel_err(actual.float(), expected) < 3e-3  # collapsed is bf16

    inputs = (x3, weight, scale, base, pre3)
    cotangents = [torch.randn_like(t) for t in ref]
    got_grads = torch.autograd.grad(got, inputs, cotangents)
    ref_grads = torch.autograd.grad(ref, inputs, cotangents)
    for actual, expected in zip(got_grads, ref_grads):
        assert rel_err(actual, expected) < 5e-3


def test_backward_is_deterministic(mhc):
    x, weight, pre_mix, _, _ = make_inputs(4096, 4, 5120)

    def run():
        mixes, collapsed = mhc.mhc_projection(x, weight, pre_mix, RMS_EPS)
        torch.manual_seed(1)
        cotangents = [torch.randn_like(mixes), torch.randn_like(collapsed)]
        return torch.autograd.grad([mixes, collapsed], (x, weight, pre_mix), cotangents)

    for first, second in zip(run(), run()):
        assert torch.equal(first, second)
