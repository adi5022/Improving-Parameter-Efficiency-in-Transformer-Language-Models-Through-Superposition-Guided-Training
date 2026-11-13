"""Contract tests for the user's toy/data.py (S0-7). Skipped while it is a stub."""

import torch

from conftest import impl
from toy.data import feature_probs, sample_batch


def test_probs_power_law_and_normalised():
    p = impl(feature_probs, 1000, 1.0)
    assert p.shape == (1000,) and p.dtype == torch.float32
    assert abs(p.sum().item() - 1.0) < 1e-5
    assert (p[:-1] >= p[1:]).all(), "feature 0 must be the most frequent"
    # p_i proportional to i^-alpha  =>  p_1 / p_k = k^alpha
    assert abs((p[0] / p[9]).item() - 10.0) < 1e-3
    p2 = impl(feature_probs, 1000, 2.0)
    assert abs((p2[0] / p2[9]).item() - 100.0) < 1e-2


def test_sample_shape_range_device():
    p = impl(feature_probs, 200, 1.0)
    g = torch.Generator().manual_seed(0)
    x = impl(sample_batch, 512, p, g)
    assert x.shape == (512, 200) and x.dtype == torch.float32
    assert x.device == p.device
    assert (x >= 0).all() and (x <= 2).all()


def test_sample_statistics():
    n, alpha, batch = 100, 1.0, 200_000
    p = impl(feature_probs, n, alpha)
    g = torch.Generator().manual_seed(1)
    x = impl(sample_batch, batch, p, g)
    active = (x > 0).float().mean(0)
    # activation frequency of each feature ~ p_i (binomial s.e. tolerance, 5 sigma)
    se = (p * (1 - p) / batch).sqrt()
    assert ((active - p).abs() <= 5 * se + 1e-4).all()
    # active values ~ U(0,2): mean 1, variance 1/3
    vals = x[x > 0]
    assert abs(vals.mean().item() - 1.0) < 0.01
    assert abs(vals.var().item() - 1 / 3) < 0.01
    # about one active feature per sample on average
    assert abs((x > 0).float().sum(1).mean().item() - 1.0) < 0.02


def test_sampling_reproducible_from_generator():
    p = impl(feature_probs, 100, 1.0)
    a = impl(sample_batch, 64, p, torch.Generator().manual_seed(5))
    b = impl(sample_batch, 64, p, torch.Generator().manual_seed(5))
    c = impl(sample_batch, 64, p, torch.Generator().manual_seed(6))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_sampling_does_not_use_global_rng():
    p = impl(feature_probs, 100, 1.0)
    torch.manual_seed(0)
    a = impl(sample_batch, 64, p, torch.Generator().manual_seed(5))
    torch.manual_seed(999)
    b = impl(sample_batch, 64, p, torch.Generator().manual_seed(5))
    assert torch.equal(a, b), "all randomness must come from the generator argument"
