"""Train many toy-model runs at once on one GPU.

One toy run is tiny (a 2048 x 1000 batch through two small matmuls), so the GPU is mostly
idle between kernel launches. Here R runs that share a width and training settings are
stacked into one batch: W has shape [R, n, m], and every step is a few batched kernels
instead of R times as many small ones.

The runs stay independent and give the same results as `train_run` (tested):
  * each run's loss is its own mean over batch and features; the total loss is their sum,
    so every run's gradient is exactly its own;
  * Adam is element-wise, so stacking does not couple runs;
  * weight decay uses each run's own gamma (same formulas as toy/optim.py);
  * runs with the same (alpha, seed) share one data stream, as in `train_run`, and
    runs with the same seed share the same initial weights.
Amortised `train_seconds` = group time / number of runs.
"""

from __future__ import annotations

import math
import time

import torch

from . import metrics
from .schedule import lr_at
from .train import TEST_SEED, RunConfig

SHARED = ("n_features", "width", "steps", "batch_size", "peak_lr", "warmup_steps",
          "final_lr_frac", "test_batches", "log_every")


def group_key(cfg: RunConfig) -> tuple:
    """Runs can be batched together only if this key is equal (alpha, gamma, seed may differ)."""
    return tuple(getattr(cfg, f) for f in SHARED)


def _adam_(p, g, m1, v1, lr, t, beta1, beta2, eps):
    m1.mul_(beta1).add_(g, alpha=1 - beta1)
    v1.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    denom = (v1.sqrt() / math.sqrt(1 - beta2 ** t)).add_(eps)
    p.addcdiv_(m1, denom, value=-lr / (1 - beta1 ** t))


def fused_loss_and_grads(x, W, b, want_loss=True):
    """Forward and hand-written backward for  y = relu(x W W^T + b),  loss_r = mean (y - x)^2.

    Same maths as autograd gives for the sum of the per-run losses, but with far fewer passes
    over the big [R, B, n] tensors (in-place relu and subtraction, no saved temporaries, the
    constant factor applied once to the small gradients). x [R,B,n], W [R,n,m], b [R,n].
    Returns (loss_r or None, dW, db).
    """
    R, B, n = x.shape
    scale = 2.0 / (B * n)
    h = torch.bmm(x, W)                                              # [R,B,m]
    z = torch.baddbmm(b.unsqueeze(1), h, W.transpose(1, 2))          # [R,B,n]
    y = z.relu_()
    active = y > 0                                                   # relu'(z), 1 byte per element
    d = y.sub_(x)                                                    # y - x, in place
    loss_r = torch.einsum("rbn,rbn->r", d, d) / (B * n) if want_loss else None
    d.mul_(active)                                                   # dL/dz without the factor `scale`
    dh = torch.bmm(d, W)                                             # [R,B,m]
    dW = torch.baddbmm(torch.bmm(x.transpose(1, 2), dh), d.transpose(1, 2), h).mul_(scale)
    db = d.sum(dim=1).mul_(scale)
    return loss_r, dW, db


def train_group(cfgs: list[RunConfig], device: str = "cuda", *, model_cls=None, probs_fn=None,
                sampler=None, betas=(0.9, 0.999), eps=1e-8, fused: bool = True) -> list[dict]:
    if not cfgs:
        return []
    key = group_key(cfgs[0])
    if any(group_key(c) != key for c in cfgs):
        raise ValueError("all runs in a group must share width and training settings")
    if model_cls is None:
        from .model import ToyModel as model_cls
    if probs_fn is None:
        from .data import feature_probs as probs_fn
    if sampler is None:
        from .data import sample_batch as sampler

    c0 = cfgs[0]
    R, n, m, B = len(cfgs), c0.n_features, c0.width, c0.batch_size
    beta1, beta2 = betas

    # data streams: one per unique (alpha, seed), exactly as train_run seeds its generator
    streams = sorted({(c.alpha, c.seed) for c in cfgs})
    gens = []
    for _, seed in streams:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        gens.append(g)
    probs = {a: probs_fn(n, a, device=device) for a in {c.alpha for c in cfgs}}
    idx = torch.tensor([streams.index((c.alpha, c.seed)) for c in cfgs], device=device)
    identity = len(streams) == R and bool((idx == torch.arange(R, device=device)).all())

    # initial weights: built exactly as train_run builds them (seed, then the model class)
    Ws, bs = [], []
    for c in cfgs:
        torch.manual_seed(c.seed)
        model = model_cls(n, m)
        Ws.append(model.W.detach().clone())
        bs.append(model.b.detach().clone())
    W = torch.stack(Ws).to(device).requires_grad_(not fused)
    b = torch.stack(bs).to(device).requires_grad_(not fused)

    gamma = torch.tensor([c.gamma for c in cfgs], dtype=torch.float32, device=device)
    g_pos = torch.where(gamma >= 0, gamma, torch.zeros_like(gamma))
    g_neg = torch.where(gamma < 0, gamma, torch.zeros_like(gamma))
    any_neg = bool((gamma < 0).any())
    mW, vW = torch.zeros_like(W), torch.zeros_like(W)
    mb, vb = torch.zeros_like(b), torch.zeros_like(b)

    log_steps: list[int] = []
    log_loss: list[torch.Tensor] = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(c0.steps):
        lr = lr_at(step, c0.peak_lr, c0.steps, c0.warmup_steps, c0.final_lr_frac)
        xs = torch.stack([sampler(B, probs[a], g) for (a, _), g in zip(streams, gens)])
        x = xs if identity else xs[idx]

        t = step + 1
        log_now = t % c0.log_every == 0 or step == c0.steps - 1
        if fused:
            with torch.no_grad():
                loss_r, gW, gb = fused_loss_and_grads(x, W, b, want_loss=log_now)
        else:                                   # reference path: autograd
            W.grad = None
            b.grad = None
            h = torch.bmm(x, W)
            y = torch.relu(torch.bmm(h, W.transpose(1, 2)) + b.unsqueeze(1))
            loss_r = ((y - x) ** 2).mean(dim=(1, 2))
            loss_r.sum().backward()
            gW, gb = W.grad, b.grad

        with torch.no_grad():
            # 1. weight decay on W (per-run gamma), same formulas as toy/optim.py
            if any_neg:
                row_norm = W.norm(dim=2, keepdim=True).add_(eps)
                delta = (lr * g_neg).view(R, 1, 1) * W * (1 - 1 / row_norm)
                W.mul_((1 - lr * g_pos).view(R, 1, 1)).add_(delta)
            else:
                W.mul_((1 - lr * g_pos).view(R, 1, 1))
            # 2. Adam update (no decay on b)
            _adam_(W, gW, mW, vW, lr, t, beta1, beta2, eps)
            _adam_(b, gb, mb, vb, lr, t, beta1, beta2, eps)

        if log_now:
            log_steps.append(t)
            log_loss.append(loss_r.detach().cpu())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    logs = torch.stack(log_loss)                      # [num_logs, R]
    results = []
    for r, c in enumerate(cfgs):
        col = logs[:, r]
        bad = (~torch.isfinite(col)).nonzero()
        if len(bad):
            stop = int(bad[0]) + 1
            status = f"non-finite loss at step {log_steps[stop - 1]}"
        else:
            stop, status = len(col), "ok"
        Wr, br = W[r].detach(), b[r].detach()
        res = {
            "config": {**c.__dict__}, "run_id": c.run_id(), "status": status,
            "train_seconds": seconds / R, "group_size": R, "batched": True,
            "train_log_steps": log_steps[:stop], "train_log_loss": col[:stop].tolist(),
            "final_train_loss": float(col[stop - 1]), "test_loss": float("nan"),
            "phi_half": metrics.represented_fraction(Wr, 0.5),
            "phi_one": metrics.represented_fraction(Wr, 1.0),
            "mso_all": metrics.mean_squared_overlap(Wr),
            "mso_represented": metrics.mean_squared_overlap(Wr, 0.5),
            "row_norms": metrics.row_norms(Wr).cpu(), "W": Wr.cpu().clone(), "b": br.cpu().clone(),
            "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
            "torch_version": torch.__version__,
        }
        if status == "ok":
            model = model_cls(n, m).to(device)
            with torch.no_grad():
                model.W.copy_(Wr)
                model.b.copy_(br)
            model.eval()
            res["test_loss"] = metrics.test_loss(model, probs[c.alpha], sampler,
                                                 c.test_batches * c.batch_size, c.batch_size, TEST_SEED)
        results.append(res)
    return results
