"""Train one toy-model run and return everything needed for analysis.

Seeding (decision S0-3): `seed` seeds both the weight init (torch.manual_seed) and the
training-data generator. Runs that differ only in gamma therefore share the same
initial weights and the same data stream (common random numbers), which makes
gamma-vs-gamma comparisons less noisy. The test set uses a separate fixed seed, so all
runs at the same alpha are scored on the same test samples. CUDA matmuls are not
guaranteed bitwise deterministic, so reruns can differ in the last digits.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch

from . import metrics
from .schedule import lr_at

TEST_SEED = 10_000  # fixed across all runs; never equal to a training seed


@dataclass(frozen=True)
class RunConfig:
    alpha: float
    width: int
    gamma: float
    seed: int
    n_features: int = 1000
    steps: int = 20_000
    batch_size: int = 2048
    peak_lr: float = 1e-2
    warmup_steps: int = 2000
    final_lr_frac: float = 0.05
    test_batches: int = 100          # test set = test_batches * batch_size samples
    log_every: int = 100
    lam: float = 0.0                 # Stage 1: strength of the overlap penalty (0 = none)
    penalty: str = "none"            # "none" | "plain" | "weighted" (batched trainer only, toy/batched.py)

    def run_id(self) -> str:
        base = f"a{self.alpha:g}_m{self.width}_g{self.gamma:+.2f}_s{self.seed}"
        return base if self.penalty == "none" else f"{base}_p{self.penalty}{self.lam:g}"


def _defaults(model_cls, optimizer_cls, probs_fn, sampler):
    """Import the user's implementations unless stand-ins are passed (tests do that)."""
    if model_cls is None:
        from .model import ToyModel as model_cls
    if optimizer_cls is None:
        from .optim import AdamWNegDecay as optimizer_cls
    if probs_fn is None:
        from .data import feature_probs as probs_fn
    if sampler is None:
        from .data import sample_batch as sampler
    return model_cls, optimizer_cls, probs_fn, sampler


def train_run(cfg: RunConfig, device: str = "cuda", *, model_cls=None, optimizer_cls=None,
              probs_fn=None, sampler=None) -> dict:
    if cfg.penalty != "none":
        raise NotImplementedError("the overlap penalty is only in the batched trainer (toy/batched.py)")
    model_cls, optimizer_cls, probs_fn, sampler = _defaults(model_cls, optimizer_cls, probs_fn, sampler)

    torch.manual_seed(cfg.seed)
    model = model_cls(cfg.n_features, cfg.width).to(device)
    probs = probs_fn(cfg.n_features, cfg.alpha, device=device)

    optimizer = optimizer_cls(
        [{"params": [model.W], "weight_decay": cfg.gamma},
         {"params": [model.b], "weight_decay": 0.0}],
        lr=cfg.peak_lr,
    )
    data_gen = torch.Generator(device=device)
    data_gen.manual_seed(cfg.seed)

    log_steps, log_loss = [], []
    status = "ok"
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        lr = lr_at(step, cfg.peak_lr, cfg.steps, cfg.warmup_steps, cfg.final_lr_frac)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x = sampler(cfg.batch_size, probs, data_gen)
        loss = ((model(x) - x) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % cfg.log_every == 0 or step == cfg.steps - 1:
            value = loss.item()
            log_steps.append(step + 1)
            log_loss.append(value)
            if not math.isfinite(value):
                status = f"non-finite loss at step {step + 1}"
                break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t0

    model.eval()
    W = model.W.detach()
    result = {
        "config": asdict(cfg),
        "run_id": cfg.run_id(),
        "status": status,
        "train_seconds": train_seconds,
        "train_log_steps": log_steps,
        "train_log_loss": log_loss,
        "final_train_loss": log_loss[-1] if log_loss else float("nan"),
        "test_loss": float("nan"),
        "phi_half": metrics.represented_fraction(W, 0.5),
        "phi_one": metrics.represented_fraction(W, 1.0),
        "mso_all": metrics.mean_squared_overlap(W),
        "mso_represented": metrics.mean_squared_overlap(W, 0.5),
        "row_norms": metrics.row_norms(W).cpu(),
        "W": W.cpu().clone(),
        "b": model.b.detach().cpu().clone(),
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
        "torch_version": torch.__version__,
    }
    if status == "ok":
        result["test_loss"] = metrics.test_loss(
            model, probs, sampler, cfg.test_batches * cfg.batch_size, cfg.batch_size, TEST_SEED)
    return result
