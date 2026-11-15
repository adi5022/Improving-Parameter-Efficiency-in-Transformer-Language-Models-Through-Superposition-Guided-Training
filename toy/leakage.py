"""Stage 2 leakage penalty (decision D-S2-3, proposals/stage_2_design_proposal.md section 5.2).

For one sample with hidden activation h, unit direction h_hat = h / |h|, a dictionary of unit directions d_j
and the set S(h) of dictionary entries that are "on" for this sample:

    leak(h) = sum over j NOT in S(h) of  q_j * (d_j . h_hat)^2

The loss term is lambda times the batch mean of leak. Everything that comes from the SAE (d_j, S, q_j) is a
constant here: the gradient reaches the model only through h_hat. Samples with |h| below `floor` are skipped
(their direction is meaningless). Entries that almost duplicate an "on" entry (cosine above `twin_thr`) are
dropped from the sum for that sample, so the penalty does not fight the SAE's own feature splitting (failure
mode "twin latents", proposal section 5.3). Set twin_thr=None to disable.

Shapes (R runs stacked, as in toy/batched.py):  h [R,B,m],  D [R,N,m] (unit rows),  on [R,B,N] bool,  q [R,N] or None.

Hand-written gradient (same style as the Stage 1 overlap penalty, checked against autograd in the tests):
    d mean-leak / dh = (2 / (B |h|)) (I - h_hat h_hat^T) D_off diag(q) D_off^T h_hat
"""

from __future__ import annotations

import torch

HIDDEN_FLOOR = 1e-3   # samples with |h| below this are skipped


def off_mask(h, D, on, twin_thr=0.9, floor=HIDDEN_FLOOR):
    """Which dictionary entries count as "off" for each sample: not on, not a twin of an on entry, and the sample is valid."""
    valid = h.norm(dim=2, keepdim=True) > floor                       # [R,B,1]
    off = (~on) & valid
    twins = 0
    if twin_thr is not None:
        G = torch.bmm(D, D.transpose(1, 2))                           # [R,N,N] cosines (D has unit rows)
        T = G > twin_thr
        T.diagonal(dim1=1, dim2=2).fill_(False)
        if bool(T.any()):
            near_on = torch.bmm(on.float(), T.float()) > 0           # [R,B,N]: entry j is a twin of some on entry
            twins = int((near_on & ~on & valid).sum())
            off = off & ~near_on
    return off, twins


def leak_value_and_grad(h, D, on, q=None, twin_thr=0.9, want_value=True, floor=HIDDEN_FLOOR):
    """Mean leakage per run [R] and its gradient with respect to h [R,B,m] (all SAE inputs constant).

    Returns (value or None, dh, twin_count)."""
    R, B, m = h.shape
    off, twins = off_mask(h, D, on, twin_thr, floor)
    norm = h.norm(dim=2, keepdim=True).clamp_min(floor)
    hh = h / norm
    c = torch.bmm(hh, D.transpose(1, 2))                              # [R,B,N] cosines with every entry
    cq = c * off
    if q is not None:
        cq = cq * q.unsqueeze(1)
    value = (c * cq).sum(dim=(1, 2)) / B if want_value else None
    g_hh = torch.bmm(cq, D).mul_(2.0)                                 # [R,B,m]
    g_h = (g_hh - (g_hh * hh).sum(2, keepdim=True) * hh) / norm
    valid = (h.norm(dim=2, keepdim=True) > floor)
    g_h = g_h * valid / B
    return value, g_h, twins


def leak_value(h, D, on, q=None, twin_thr=0.9, floor=HIDDEN_FLOOR):
    """Differentiable reference (autograd): the same mean leakage per run, with `on`, `D` and `q` treated as constants."""
    D, on = D.detach(), on.detach()
    q = None if q is None else q.detach()
    off, _ = off_mask(h.detach(), D, on, twin_thr, floor)
    valid = h.norm(dim=2, keepdim=True) > floor
    hh = h / h.norm(dim=2, keepdim=True).clamp_min(floor)
    c = torch.bmm(hh, D.transpose(1, 2))
    cq2 = c * c * off * valid
    if q is not None:
        cq2 = cq2 * q.unsqueeze(1)
    return cq2.sum(dim=(1, 2)) / h.shape[1]


def oracle_readings(W, x):
    """Oracle dictionary for the control ORA-L: the model's own unit rows and the TRUE active set (x > 0)."""
    raw = W.norm(dim=2, keepdim=True)
    D = W / raw.clamp_min(1e-2)
    return D.detach(), (x > 0)
