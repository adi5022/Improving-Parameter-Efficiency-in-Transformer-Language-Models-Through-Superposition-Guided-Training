"""The batched trainer must give the same results as training each run on its own."""

import json
import math

import pytest
import torch

from toy.batched import group_key, train_group
from toy.sweep import make_groups, run_grid
from toy.train import RunConfig, train_run


def cfg(**kw):
    base = dict(alpha=1.0, width=5, gamma=0.0, seed=0, n_features=40, steps=80, batch_size=64,
                warmup_steps=10, test_batches=3, log_every=20)
    base.update(kw)
    return RunConfig(**base)


# mixes signs of gamma, a shared seed (shared init and data), a different seed, and a different alpha
MIXED = [cfg(gamma=-1.0, seed=0), cfg(gamma=0.5, seed=0), cfg(gamma=0.0, seed=1),
         cfg(gamma=-0.3, seed=1, alpha=1.5), cfg(gamma=1.0, seed=2, alpha=1.5)]


def test_batched_matches_single_runs():
    batched = train_group(MIXED, device="cpu")
    for c, b in zip(MIXED, batched):
        s = train_run(c, device="cpu")
        assert b["run_id"] == s["run_id"] and b["status"] == s["status"] == "ok"
        assert torch.allclose(b["W"], s["W"], atol=1e-4), (c.run_id(), (b["W"] - s["W"]).abs().max())
        assert torch.allclose(b["b"], s["b"], atol=1e-4)
        assert b["test_loss"] == pytest.approx(s["test_loss"], rel=1e-3)
        assert b["final_train_loss"] == pytest.approx(s["final_train_loss"], rel=1e-3)
        assert b["train_log_steps"] == s["train_log_steps"]
        assert b["phi_half"] == pytest.approx(s["phi_half"], abs=0.03)


def test_batch_order_and_size_do_not_change_a_run():
    alone = train_group([MIXED[1]], device="cpu")[0]
    in_group = train_group(MIXED, device="cpu")[1]
    assert torch.allclose(alone["W"], in_group["W"], atol=1e-4)


def test_group_rejects_mismatched_settings():
    with pytest.raises(ValueError):
        train_group([cfg(width=5), cfg(width=6)], device="cpu")


def test_make_groups_splits_by_width_and_size():
    runs = [cfg(width=w, seed=s) for w in (5, 6) for s in range(5)]
    groups = make_groups(runs, max_group=3)
    assert sorted(len(g) for g in groups) == [2, 2, 3, 3]
    assert all(len({group_key(c) for c in g}) == 1 for g in groups)


def test_run_grid_batched_writes_and_resumes(tmp_path):
    runs = [cfg(gamma=g, seed=s) for g in (-1.0, 0.5) for s in (0, 1)]
    git = {"commit": "test", "dirty_paths": []}
    run_grid(runs, tmp_path, "cpu", git)
    run_grid(runs, tmp_path, "cpu", git)               # second call skips everything
    lines = (tmp_path / "summary.jsonl").read_text().splitlines()
    assert len(lines) == 4
    row = json.loads(lines[0])
    assert math.isfinite(row["test_loss"]) and row["git_commit"] == "test"
    saved = torch.load(tmp_path / f"{runs[0].run_id()}.pt", weights_only=False)
    assert saved["batched"] and saved["git"]["commit"] == "test"


def test_fused_gradients_match_autograd():
    torch.manual_seed(0)
    from toy.batched import fused_loss_and_grads
    R, B, n, m = 3, 32, 20, 4
    x = (torch.rand(R, B, n) < 0.3).float() * torch.rand(R, B, n) * 2
    W = (torch.randn(R, n, m) / 2).requires_grad_()
    b = torch.randn(R, n).requires_grad_()
    y = torch.relu(torch.bmm(torch.bmm(x, W), W.transpose(1, 2)) + b.unsqueeze(1))
    loss_r = ((y - x) ** 2).mean(dim=(1, 2))
    loss_r.sum().backward()
    with torch.no_grad():
        fl, gW, gb = fused_loss_and_grads(x, W.detach().clone(), b.detach().clone())
    assert torch.allclose(fl, loss_r.detach(), rtol=1e-5)
    assert torch.allclose(gW, W.grad, rtol=1e-4, atol=1e-6)
    assert torch.allclose(gb, b.grad, rtol=1e-4, atol=1e-6)


def test_fused_and_autograd_training_agree():
    fused = train_group(MIXED, device="cpu", fused=True)
    ref = train_group(MIXED, device="cpu", fused=False)
    for f, r in zip(fused, ref):
        assert torch.allclose(f["W"], r["W"], atol=1e-4)


def test_split_covers_every_run_once_and_follows_weights():
    from toy.sweep import split_configs
    runs = [cfg(width=w, seed=s, gamma=g) for w in (5, 6, 7, 8, 9, 10) for s in range(3) for g in (-1.0, 0.0, 1.0)]
    parts = [split_configs(runs, [7, 3], i, max_group=24) for i in (0, 1)]
    ids = [c.run_id() for p in parts for c in p]
    assert sorted(ids) == sorted(c.run_id() for c in runs)          # nothing lost, nothing twice
    cost = [sum(c.width for c in p) for p in parts]
    assert cost[0] > cost[1]                                         # the heavier weight gets more work
    assert split_configs(runs, [7, 3], 0) == parts[0]                # deterministic


def test_split_rejects_bad_arguments():
    from toy.sweep import split_configs
    with pytest.raises(ValueError):
        split_configs([cfg()], [1, 1], 2)
    with pytest.raises(ValueError):
        split_configs([cfg()], [1, 0], 0)
