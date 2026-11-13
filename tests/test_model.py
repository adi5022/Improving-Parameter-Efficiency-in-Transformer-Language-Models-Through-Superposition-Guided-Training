"""Contract tests for the user's toy/model.py (S0-7). Skipped while it is a stub."""

import math

import torch

from conftest import impl
from toy.model import ToyModel


def make(n=1000, m=25, seed=0):
    torch.manual_seed(seed)
    return impl(ToyModel, n, m)


def test_parameter_names_and_shapes():
    model = make(1000, 25)
    assert model.W.shape == (1000, 25)
    assert model.b.shape == (1000,)
    names = {name for name, _ in model.named_parameters()}
    assert names == {"W", "b"}, f"expected exactly W and b, got {names}"


def test_init_statistics():
    model = make(1000, 25)
    # W ~ N(0,1)/sqrt(m): entry s.d. ~ 1/5 = 0.2; b ~ N(0,1)
    assert abs(model.W.std().item() - 1 / math.sqrt(25)) < 0.01
    assert abs(model.W.mean().item()) < 0.01
    assert abs(model.b.std().item() - 1.0) < 0.1, "b should be N(0,1), not zeros"


def test_init_is_seeded_by_global_rng():
    a, b = make(seed=3), make(seed=3)
    assert torch.equal(a.W, b.W) and torch.equal(a.b, b.b)


def test_forward_matches_definition():
    model = make(50, 7)
    x = torch.rand(16, 50)
    expected = torch.relu(x @ model.W @ model.W.T + model.b)
    y = impl(model, x)
    assert y.shape == (16, 50)
    assert torch.allclose(y, expected, atol=1e-5)


def test_output_nonnegative_and_differentiable():
    model = make(50, 7)
    y = impl(model, torch.rand(8, 50))
    assert (y >= 0).all()
    y.sum().backward()
    assert model.W.grad is not None and model.b.grad is not None
