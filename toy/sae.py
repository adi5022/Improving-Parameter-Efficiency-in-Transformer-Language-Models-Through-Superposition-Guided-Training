"""Batched sparse autoencoders for Stage 2 (decision D-S2-1: three types, chosen by the S2-A bake-off).

R independent SAEs are stacked (parameters have a leading run dimension), like the toy runs in toy/batched.py.
Each SAE reads the detached hidden activation h [R,B,m] and has N latents:

    pre = (h - b_dec) @ W_enc + b_enc          # [R,B,N]
    z   = activation(pre)                       # sparse codes
    h_hat = z @ W_dec + b_dec                   # W_dec rows are kept at unit length

Types:
  relu_l1    z = relu(pre); loss = reconstruction + l1 * sum |z|
  batchtopk  keep the B*k largest relu(pre) values of the batch (k = wanted mean number of active latents per
             sample); at read-out a learned threshold (moving average of the smallest kept value) is used, so
             the active set can be decided per sample; AuxK loss revives latents that have not fired for a while
  jumprelu   z = pre * (pre > theta) with a learned threshold per latent; sparsity = l0 coefficient * number of
             active latents; thresholds get straight-through gradients with a rectangular kernel

The SAE never receives a gradient from the penalty (stop-gradient, decision D-09): `step` takes a detached copy
of h, and `readings` returns detached tensors only.
"""

from __future__ import annotations

import torch


class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre, theta, bw):
        ctx.save_for_backward(pre, theta)
        ctx.bw = bw
        return pre * (pre > theta.unsqueeze(1))

    @staticmethod
    def backward(ctx, g):
        pre, theta = ctx.saved_tensors
        t3 = theta.unsqueeze(1)
        near = ((pre - t3).abs() < ctx.bw / 2).to(g.dtype)
        g_pre = g * (pre > t3)
        g_theta = (-(t3 / ctx.bw) * near * g).sum(dim=1)            # sum over the batch
        return g_pre, g_theta, None


class _Step(torch.autograd.Function):
    """Heaviside(pre - theta) with a straight-through gradient to theta (for the L0 term)."""

    @staticmethod
    def forward(ctx, pre, theta, bw):
        ctx.save_for_backward(pre, theta)
        ctx.bw = bw
        return (pre > theta.unsqueeze(1)).to(pre.dtype)

    @staticmethod
    def backward(ctx, g):
        pre, theta = ctx.saved_tensors
        near = ((pre - theta.unsqueeze(1)).abs() < ctx.bw / 2).to(g.dtype)
        return None, (-(1.0 / ctx.bw) * near * g).sum(dim=1), None


class SAEBank:
    def __init__(self, R, m, N, kind="batchtopk", k=1.0, l1=1e-3, l0=1e-3, lr=1e-3, bw=1e-3,
                 aux_alpha=1 / 32, aux_k=64, dead_after=200, device="cpu", seed=0):
        if kind not in ("relu_l1", "batchtopk", "jumprelu"):
            raise ValueError(f"unknown SAE type {kind}")
        self.R, self.m, self.N, self.kind = R, m, N, kind
        # sparsity settings may differ per run: a number or a list of R numbers
        per_run = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device).expand(R).clone()
        self.k, self.l1, self.l0 = per_run(k), per_run(l1), per_run(l0)
        self.bw, self.aux_alpha, self.aux_k, self.dead_after = bw, aux_alpha, aux_k, dead_after
        Wd = torch.stack([torch.randn(N, m, generator=torch.Generator(device="cpu").manual_seed(seed + r))
                          for r in range(R)])                           # run r's start depends only on seed + r
        Wd = (Wd / Wd.norm(dim=2, keepdim=True)).to(device)
        self.W_dec = Wd.clone().requires_grad_(True)
        self.W_enc = Wd.transpose(1, 2).clone().requires_grad_(True)       # tied start
        self.b_enc = torch.zeros(R, N, device=device, requires_grad=True)
        self.b_dec = torch.zeros(R, m, device=device, requires_grad=True)
        self.params = [self.W_dec, self.W_enc, self.b_enc, self.b_dec]
        if kind == "jumprelu":
            self.theta = torch.full((R, N), 0.2, device=device, requires_grad=True)       # start value 0.2 (0.05 stayed dense, 0.4 silenced latents in the easy-case test)
            self.params.append(self.theta)
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.threshold = torch.zeros(R, 1, device=device)                   # batchtopk read-out threshold (moving average)
        self.since_fired = torch.zeros(R, N, device=device)
        self.fvu_ema = torch.ones(R, device=device)
        self.last_fvu = torch.ones(R, device=device)

    # ----- encoding -----
    def _pre(self, h):
        return torch.baddbmm(self.b_enc.unsqueeze(1), h - self.b_dec.unsqueeze(1), self.W_enc)

    def _codes_train(self, h):
        pre = self._pre(h)
        if self.kind == "relu_l1":
            return torch.relu(pre), pre
        if self.kind == "batchtopk":
            a = torch.relu(pre)
            R, B, N = a.shape
            keep = (self.k * B).round().clamp_min(1).long()                # [R] values kept per run
            flat = a.reshape(R, B * N)
            top = flat.topk(int(keep.max()), dim=1).values
            thr = top.gather(1, (keep - 1).unsqueeze(1)).detach()            # smallest kept value per run
            self.threshold = 0.99 * self.threshold + 0.01 * thr if self.threshold.abs().sum() > 0 else thr.clone()
            return (a * (a >= thr.unsqueeze(2))), pre
        return _JumpReLU.apply(pre, self.theta, self.bw), pre

    @torch.no_grad()
    def encode(self, h):
        """Read-out codes z [R,B,N] for detached h, using the same activation rule per sample."""
        pre = self._pre(h)
        if self.kind == "relu_l1":
            return torch.relu(pre)
        if self.kind == "batchtopk":
            a = torch.relu(pre)
            return a * (a >= self.threshold.unsqueeze(2))
        return pre * (pre > self.theta.unsqueeze(1))

    @torch.no_grad()
    def encode_one(self, r, h):
        """Read-out codes for run r only: h [B,m] -> z [B,N] (for evaluation without stacking all runs)."""
        pre = (h - self.b_dec[r]) @ self.W_enc[r] + self.b_enc[r]
        if self.kind == "relu_l1":
            return torch.relu(pre)
        if self.kind == "batchtopk":
            a = torch.relu(pre)
            return a * (a >= self.threshold[r])
        return pre * (pre > self.theta[r])

    # ----- training -----
    def step(self, h):
        """One optimiser step on a detached copy of h [R,B,m]. Returns the reconstruction FVU per run [R]."""
        h = h.detach()
        z, pre = self._codes_train(h)
        rec = torch.baddbmm(self.b_dec.unsqueeze(1), z, self.W_dec)
        err = ((rec - h) ** 2).sum(2)                                         # [R,B]
        mse = err.mean(1)
        loss = mse
        if self.kind == "relu_l1":
            loss = loss + self.l1 * z.sum(2).mean(1)
        elif self.kind == "jumprelu":
            loss = loss + self.l0 * _Step.apply(pre, self.theta, self.bw).sum(2).mean(1)
        else:
            with torch.no_grad():
                fired = (z > 0).any(dim=1)
                self.since_fired = torch.where(fired, torch.zeros_like(self.since_fired), self.since_fired + 1)
                dead = self.since_fired > self.dead_after
            if bool(dead.any()):
                a = torch.relu(pre) * dead.unsqueeze(1)
                kk = min(self.aux_k, self.N)
                topv = a.topk(kk, dim=2)
                a_aux = torch.zeros_like(a).scatter(2, topv.indices, topv.values)
                aux_rec = torch.bmm(a_aux, self.W_dec)
                resid = (h - rec).detach()
                aux = ((aux_rec - resid) ** 2).sum(2).mean(1)
                loss = loss + self.aux_alpha * aux * dead.any(dim=1).float()
        self.opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        self.opt.step()
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=2, keepdim=True).clamp_min(1e-8))
            var = ((h - h.mean(dim=1, keepdim=True)) ** 2).sum(2).mean(1).clamp_min(1e-12)
            fvu = (mse.detach() / var)
            self.last_fvu = fvu
            self.fvu_ema = 0.98 * self.fvu_ema + 0.02 * fvu
        return fvu

    @torch.no_grad()
    def readings(self, h):
        """What the penalty may use (all detached): unit decoder rows D [R,N,m], active set on [R,B,N], codes z."""
        z = self.encode(h.detach())
        return self.W_dec.detach().clone(), z > 0, z

    @torch.no_grad()
    def dead_fraction(self, z):
        return (~(z > 0).any(dim=1)).float().mean(dim=1)                      # [R]
