"""Tests for Claude's half (S0-7): schedule, metrics, runner plumbing, grids, analysis.

The runner is exercised with deliberately *different* stand-in components (a tanh
model, plain torch AdamW, a uniform sparse sampler) so these tests check the plumbing
without containing the user's implementations.
"""

import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from toy import analysis, metrics
from toy.schedule import lr_at
from toy.sweep import grid_configs, run_grid
from toy.train import RunConfig, train_run


# ---------- schedule (values computed from the reference get_lr) ----------

def reference_get_lr(step, lr, n_steps, warmup_steps=2000):
    step = step + 1
    min_lr = 0.05 * lr
    if warmup_steps < n_steps:
        if step < warmup_steps:
            return lr * step / warmup_steps
        return (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (n_steps - warmup_steps))) + min_lr
    return (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * step / n_steps)) + min_lr


@pytest.mark.parametrize("n_steps,warmup", [(20_000, 2000), (300, 50), (100, 2000)])
def test_schedule_matches_reference(n_steps, warmup):
    for s in list(range(0, n_steps, max(1, n_steps // 97))) + [n_steps - 1]:
        assert lr_at(s, 1e-2, n_steps, warmup) == pytest.approx(reference_get_lr(s, 1e-2, n_steps, warmup))
    assert lr_at(n_steps - 1, 1e-2, n_steps, warmup) == pytest.approx(0.05 * 1e-2)


# ---------- metrics ----------

def test_represented_fraction():
    W = torch.zeros(4, 3)
    W[0, 0], W[1, 1], W[2, 2] = 0.4, 0.6, 1.5
    assert metrics.represented_fraction(W, 0.5) == 0.5
    assert metrics.represented_fraction(W, 1.0) == 0.25


def test_mean_squared_overlap_orthogonal_and_random():
    assert metrics.mean_squared_overlap(torch.eye(5) * 3) == pytest.approx(0.0)
    torch.manual_seed(0)
    m = 40
    assert metrics.mean_squared_overlap(torch.randn(2000, m)) == pytest.approx(1 / m, rel=0.05)
    W = torch.cat([torch.eye(3), 0.1 * torch.ones(2, 3)])
    assert metrics.mean_squared_overlap(W, norm_threshold=0.5) == pytest.approx(0.0)
    assert math.isnan(metrics.mean_squared_overlap(torch.eye(3) * 0.1, norm_threshold=0.5))


def test_test_loss_is_mse_over_chunks():
    class Zero(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)
    probs = torch.full((10,), 0.1)
    ones = lambda bs, p, g: torch.ones(bs, p.numel())
    assert metrics.test_loss(Zero(), probs, ones, n_samples=1000, chunk=300, seed=0) == pytest.approx(1.0)


# ---------- runner plumbing with stand-ins ----------

class StandInModel(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, m) * 0.1)
        self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return torch.tanh(x @ self.W) @ self.W.T + self.b


def stand_in_probs(n, alpha, device=None):
    return torch.full((n,), 1.0 / n, device=device)


def stand_in_sampler(bs, probs, g):
    return (torch.rand(bs, probs.numel(), generator=g, device=probs.device) < 0.05).float()


STAND_INS = dict(model_cls=StandInModel, optimizer_cls=torch.optim.AdamW,
                 probs_fn=stand_in_probs, sampler=stand_in_sampler)


def small_cfg(**kw):
    base = dict(alpha=1.0, width=4, gamma=0.1, seed=0, n_features=30, steps=60, batch_size=32,
                warmup_steps=10, test_batches=2, log_every=20)
    base.update(kw)
    return RunConfig(**base)


def test_train_run_outputs_and_learning():
    r = train_run(small_cfg(), device="cpu", **STAND_INS)
    assert r["status"] == "ok"
    assert r["train_log_steps"] == [20, 40, 60]
    assert r["W"].shape == (30, 4) and r["row_norms"].shape == (30,)
    for k in ("test_loss", "phi_half", "phi_one", "mso_all"):
        assert math.isfinite(r[k])
    assert r["run_id"] == "a1_m4_g+0.10_s0"


def test_train_run_reproducible_and_seed_dependent():
    a = train_run(small_cfg(seed=3), device="cpu", **STAND_INS)
    b = train_run(small_cfg(seed=3), device="cpu", **STAND_INS)
    c = train_run(small_cfg(seed=4), device="cpu", **STAND_INS)
    assert torch.equal(a["W"], b["W"]) and a["test_loss"] == b["test_loss"]
    assert not torch.equal(a["W"], c["W"])


def test_run_grid_resumes(tmp_path):
    cfgs = [small_cfg(seed=0), small_cfg(seed=1)]
    git = {"commit": "test", "dirty_paths": []}
    run_grid(cfgs, tmp_path, "cpu", git, **STAND_INS)
    run_grid(cfgs, tmp_path, "cpu", git, **STAND_INS)     # second call must skip both
    lines = (tmp_path / "summary.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert {json.loads(l)["seed"] for l in lines} == {0, 1}


def test_grid_sizes_match_approved_plan():
    assert len(grid_configs("G-A")) == 147
    assert len(grid_configs("G-B")) == 72
    assert len({c.run_id() for c in grid_configs("G-A")}) == 147


# ---------- analysis on synthetic numbers ----------

def test_fit_exponent_recovers_power_law():
    ms = np.array([10, 15, 25, 39, 63, 100])
    e = analysis.fit_exponent(ms, 3.0 * ms ** -0.9)
    assert e["exponent"] == pytest.approx(0.9) and e["se"] == pytest.approx(0.0, abs=1e-9)


def _fake_rows():
    rows = []
    for m in (10, 20, 40, 80):
        for g, scale in ((-1.0, 1.0), (0.0, 1.3), (1.0, 2.0)):
            for s in (0, 1):
                loss = scale * m ** -1.0 * (1 + 0.01 * s)
                rows.append({"run_id": f"{m}{g}{s}", "status": "ok", "alpha": 1.0, "width": m,
                             "gamma": g, "seed": s, "test_loss": loss, "phi_half": 0.95 if g < 0 else 0.1})
    return rows


def test_yardstick_and_equivalent_width():
    agg = analysis.aggregate(_fake_rows())
    ys = analysis.yardstick(agg, 1.0)
    assert all(v["gamma"] == -1.0 for v in ys.values())
    # L*(m) = 1.005/m, so a loss of 1.005/30 corresponds to m_eq = 30
    assert analysis.equivalent_width(1.005 / 30, ys) == pytest.approx(30, rel=1e-6)
    assert math.isnan(analysis.equivalent_width(10.0, ys))
    assert analysis.interp_loglog(30, list(ys), [v["loss"] for v in ys.values()]) == pytest.approx(1.005 / 30)


def test_gates_on_synthetic_rows():
    gates = {"G0.1": {"alpha": 1.0, "width": 80, "strong_gamma": -1.0, "strong_min_phi_half": 0.9,
                      "weak_gamma": 1.0, "weak_max_phi_half": 0.3},
             "G0.2": {"alpha": 1.0, "gamma": -1.0, "exponent_min": 0.8, "exponent_max": 1.2},
             "G0.3": {"alphas": [1.5], "gamma": 0.55, "tolerance": 0.2}}
    res = {g["gate"]: g for g in analysis.evaluate_gates(_fake_rows(), [], gates)}
    assert res["G0.1"]["pass"] is True
    assert res["G0.2"]["pass"] is True
    assert res["G0.3"]["pass"] is None      # no G-B rows


def test_figures_render(tmp_path):
    written = analysis.make_figures(_fake_rows(), [], out_dir=tmp_path)
    assert len(written) == 3 and all(p.exists() for p in written)
