"""Measurements of the trained toy model.

All functions take the weight matrix W of shape [n_features, width]; row W_i is the
direction feature i is written to. Definitions follow Liu, Liu & Gore 2025 (SUP_2025_002):
    represented fraction  phi_t = #{i : ||W_i|| > t} / n      (t = 1/2 and t = 1, Eq. 3 / App. A.1)
    squared overlap       (W_i_hat . W_j_hat)^2 for unit rows, i != j
For random unit vectors in m dimensions E[(u.v)^2] = 1/m, so in strong superposition
the mean squared overlap of represented rows should scale like 1/m.
"""

import torch


@torch.no_grad()
def row_norms(W: torch.Tensor) -> torch.Tensor:
    return W.norm(dim=1)


@torch.no_grad()
def represented_fraction(W: torch.Tensor, threshold: float = 0.5) -> float:
    return (row_norms(W) > threshold).float().mean().item()


@torch.no_grad()
def mean_squared_overlap(W: torch.Tensor, norm_threshold: float | None = None) -> float:
    """Mean of (W_i_hat . W_j_hat)^2 over pairs i != j.

    If norm_threshold is given, only rows with ||W_i|| > norm_threshold are included
    (returns nan if fewer than two rows qualify).
    """
    norms = row_norms(W)
    if norm_threshold is not None:
        W = W[norms > norm_threshold]
        norms = norms[norms > norm_threshold]
    k = W.shape[0]
    if k < 2:
        return float("nan")
    U = W / norms.clamp_min(1e-12).unsqueeze(1)
    G = (U @ U.T) ** 2
    off_diag_sum = G.sum() - G.diagonal().sum()
    return (off_diag_sum / (k * (k - 1))).item()


@torch.no_grad()
def test_loss(model, probs: torch.Tensor, sample_batch, n_samples: int, chunk: int, seed: int) -> float:
    """MSE on n_samples fresh samples, same reduction as training (mean over batch and features).

    Uses its own generator seeded with `seed`, so every run at the same alpha is scored
    on the same test draws (as long as sample_batch consumes the generator the same way).
    """
    g = torch.Generator(device=probs.device)
    g.manual_seed(seed)
    total, count = 0.0, 0
    remaining = n_samples
    while remaining > 0:
        bs = min(chunk, remaining)
        x = sample_batch(bs, probs, g)
        y = model(x)
        total += ((y - x) ** 2).mean().item() * bs
        count += bs
        remaining -= bs
    return total / count
