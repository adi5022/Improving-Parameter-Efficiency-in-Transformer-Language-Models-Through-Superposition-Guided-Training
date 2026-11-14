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
    assert len(parts[0]) > len(parts[1])                              # the heavier weight gets more work
    assert split_configs(runs, [7, 3], 0) == parts[0]                # deterministic


def test_split_rejects_bad_arguments():
    from toy.sweep import split_configs
    with pytest.raises(ValueError):
        split_configs([cfg()], [1, 1], 2)
    with pytest.raises(ValueError):
        split_configs([cfg()], [1, 0], 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="half-precision products need a GPU")
def test_half_precision_gradients_close_to_float32():
    from toy.batched import fused_loss_and_grads
    torch.manual_seed(0)
    R, B, n, m = 3, 512, 200, 12
    x = (torch.rand(R, B, n, device="cuda") < 0.02).float() * torch.rand(R, B, n, device="cuda") * 2
    W = torch.randn(R, n, m, device="cuda") / m ** 0.5
    b = torch.randn(R, n, device="cuda")
    l32, gW32, gb32 = fused_loss_and_grads(x, W, b)
    l16, gW16, gb16 = fused_loss_and_grads(x, W, b, half=True)
    rel = lambda a, c: ((a - c).norm() / c.norm()).item()
    assert rel(l16, l32) < 5e-3 and rel(gW16, gW32) < 2e-2 and rel(gb16, gb32) < 2e-2


def test_subsplit_divides_a_part_without_changing_the_other_part():
    from toy.sweep import split_configs
    runs = [cfg(width=w, seed=s, gamma=g) for w in (5, 6, 7, 8, 9, 10) for s in range(4) for g in (-1.0, 0.0, 1.0)]
    first = split_configs(runs, [1, 1], 0)
    second = split_configs(runs, [1, 1], 1)
    halves = [split_configs(second, [1, 1], k) for k in (0, 1)]
    ids = sorted(c.run_id() for h in halves for c in h)
    assert ids == sorted(c.run_id() for c in second)                 # the second half is covered once
    assert not {c.run_id() for c in first} & set(ids)                # and nothing leaks into the first half


def test_grid_with_parts_is_the_union_of_its_rectangles(tmp_path):
    import json as _json
    from toy.sweep import grid_configs
    spec = {"defaults": {"n_features": 10, "steps": 5, "batch_size": 4, "warmup_steps": 1, "test_batches": 1, "log_every": 1},
            "grids": {"X": {"parts": [{"alpha": [1.0], "width": [3], "gamma": [0.0, 1.0], "seed": [0, 1]},
                                       {"alpha": [1.0], "width": [4], "gamma": [0.0], "seed": [0, 1]},
                                       {"alpha": [1.0], "width": [3], "gamma": [0.0], "seed": [0]}]}}}   # last one repeats a run
    p = tmp_path / "c.json"
    p.write_text(_json.dumps(spec))
    runs = grid_configs("X", p)
    assert len(runs) == 6 and len({c.run_id() for c in runs}) == 6


# ---- Stage 1: overlap penalty -------------------------------------------------------------

def _pen_inputs(weighted):
    torch.manual_seed(1)
    R, n, m = 3, 30, 5
    W = torch.randn(R, n, m)
    W[:, :3] *= 1e-4                                   # rows under the norm floor must not break the gradient
    W[:, 3:6] *= 0.3
    w = torch.rand(R, n) + 0.2 if weighted else torch.ones(R, n)
    return W, w


@pytest.mark.parametrize("weighted", [False, True])
def test_penalty_gradient_matches_autograd(weighted):
    from toy.batched import penalty_and_grad, penalty_value
    W, w = _pen_inputs(weighted)
    Wg = W.clone().double().requires_grad_()
    P = penalty_value(Wg, w.double())
    P.sum().backward()
    value, dW = penalty_and_grad(W.double(), w.double())
    assert torch.allclose(value, P.detach(), rtol=1e-10)
    assert torch.allclose(dW, Wg.grad, rtol=1e-8, atol=1e-10)


def test_plain_penalty_equals_mean_squared_overlap_metric():
    from toy import metrics
    from toy.batched import penalty_value
    torch.manual_seed(2)
    W = torch.randn(2, 25, 4)
    P = penalty_value(W, torch.ones(2, 25))
    for r in range(2):
        assert P[r].item() == pytest.approx(metrics.mean_squared_overlap(W[r]), rel=1e-4)


def test_penalty_training_fused_matches_autograd_and_changes_result():
    pen = [cfg(gamma=-0.5, seed=0, penalty="plain", lam=1e-2), cfg(gamma=0.0, seed=1, penalty="weighted", lam=1e-1),
           cfg(gamma=-0.5, seed=0)]
    fused = train_group(pen, device="cpu", fused=True)
    ref = train_group(pen, device="cpu", fused=False)
    for f, r in zip(fused, ref):
        assert torch.allclose(f["W"], r["W"], atol=1e-4)
    assert not torch.allclose(fused[0]["W"], fused[2]["W"], atol=1e-3)        # the penalty does something
    assert fused[0]["mso_all"] < fused[2]["mso_all"]                          # and it lowers the overlap
    base = train_run(pen[2], device="cpu")
    assert torch.allclose(fused[2]["W"], base["W"], atol=1e-4)               # a run without penalty is unchanged by sharing a group


def test_run_id_unchanged_without_penalty_and_unique_with():
    assert cfg().run_id() == "a1_m5_g+0.00_s0"
    ids = {cfg(penalty=p, lam=l).run_id() for p in ("plain", "weighted") for l in (1e-4, 1e-3)}
    assert len(ids) == 4
    with pytest.raises(NotImplementedError):
        train_run(cfg(penalty="plain", lam=1e-3), device="cpu")


def test_unknown_penalty_rejected():
    with pytest.raises(ValueError):
        train_group([cfg(penalty="bogus", lam=1.0)], device="cpu")
