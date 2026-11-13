"""Contract tests for the user's toy/optim.py (S0-7). Skipped while it is a stub."""

import pytest
import torch

from conftest import impl
from toy.optim import AdamWNegDecay


def _pair(gamma, seed=0):
    torch.manual_seed(seed)
    W0 = torch.randn(30, 6) / 6 ** 0.5
    b0 = torch.randn(30)
    W1, b1 = W0.clone().requires_grad_(), b0.clone().requires_grad_()
    W2, b2 = W0.clone().requires_grad_(), b0.clone().requires_grad_()
    groups = lambda W, b: [{"params": [W], "weight_decay": gamma}, {"params": [b], "weight_decay": 0.0}]
    ours = impl(AdamWNegDecay, groups(W1, b1), lr=1e-2)
    ref = torch.optim.AdamW(groups(W2, b2), lr=1e-2)
    return (W1, b1, ours), (W2, b2, ref)


@pytest.mark.parametrize("gamma", [0.0, 0.1, 0.55, 1.0])
def test_matches_torch_adamw_for_nonnegative_decay(gamma):
    (W1, b1, ours), (W2, b2, ref) = _pair(gamma)
    gen = torch.Generator().manual_seed(1)
    for step in range(200):
        lr = 1e-2 * (0.5 + 0.5 * (step % 7) / 7)     # the runner changes lr every step
        x = torch.rand(64, 30, generator=gen)
        for W, b, opt in ((W1, b1, ours), (W2, b2, ref)):
            for gr in opt.param_groups:
                gr["lr"] = lr
            loss = ((torch.relu(x @ W @ W.T + b) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            impl(opt.step) if opt is ours else opt.step()
    assert torch.allclose(W1, W2, atol=1e-5), (W1 - W2).abs().max()
    assert torch.allclose(b1, b2, atol=1e-5)


@pytest.mark.parametrize("gamma", [-1.0, -0.55, -0.1])
def test_negative_decay_single_step_formula(gamma):
    torch.manual_seed(0)
    W = (torch.randn(20, 5) * torch.linspace(0.1, 3.0, 20).unsqueeze(1)).requires_grad_()
    W_before = W.detach().clone()
    opt = impl(AdamWNegDecay, [{"params": [W], "weight_decay": gamma}], lr=0.05, eps=1e-8)
    W.grad = torch.zeros_like(W)          # zero gradient: only the decay moves W
    impl(opt.step)
    r = W_before.norm(dim=1, keepdim=True) + 1e-8
    expected = W_before + 0.05 * gamma * W_before * (1 - 1 / r)
    assert torch.allclose(W.detach(), expected, atol=1e-6)


def test_negative_decay_pulls_row_norms_toward_one():
    torch.manual_seed(0)
    W = (torch.randn(50, 8) * torch.linspace(0.2, 2.5, 50).unsqueeze(1)).requires_grad_()
    opt = impl(AdamWNegDecay, [{"params": [W], "weight_decay": -1.0}], lr=0.05)
    for _ in range(400):
        W.grad = torch.zeros_like(W)
        impl(opt.step)
    norms = W.detach().norm(dim=1)
    assert (norms - 1).abs().max() < 1e-2, norms


def test_parameters_without_grad_are_untouched():
    a = torch.randn(4, 3, requires_grad=True)
    b = torch.randn(4, 3, requires_grad=True)
    b_before = b.detach().clone()
    opt = impl(AdamWNegDecay, [{"params": [a, b], "weight_decay": -1.0}], lr=0.1)
    a.grad = torch.ones_like(a)
    impl(opt.step)
    assert torch.equal(b.detach(), b_before)
