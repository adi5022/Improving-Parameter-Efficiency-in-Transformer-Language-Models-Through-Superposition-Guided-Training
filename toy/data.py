"""Synthetic sparse-feature data.

Feature i (i = 1..n) is active with probability p_i proportional to i^(-alpha), scaled so
that sum(p) = 1, i.e. one active feature per sample on average. When a feature is
active its value is Uniform(0, 2). Feature 0 in the code is the most frequent.

alpha here is the paper's alpha (the reference scripts write the exponent as
1 + alpha_code, so our alpha = 1 is their alpha_code = 0).
"""

import torch


def feature_probs(n_features: int, alpha: float, device=None) -> torch.Tensor:
    ranks = torch.arange(1, n_features + 1, dtype=torch.float64, device=device)
    p = ranks.pow(-alpha)
    return (p / p.sum()).to(torch.float32)


def sample_batch(batch_size: int, probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    shape = (batch_size, probs.numel())
    active = torch.rand(shape, generator=generator, device=probs.device) < probs
    values = torch.rand(shape, generator=generator, device=probs.device) * 2
    return active.float() * values
