"""Stage 2 building blocks: the leakage penalty (hand-written gradient) and the batched SAEs."""

import pytest
import torch

from toy.batched import penalty_value
from toy.leakage import leak_value, leak_value_and_grad, oracle_readings
from toy.sae import SAEBank


def unit_rows(R, N, m, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(R, N, m, generator=g, dtype=dtype)
    return D / D.norm(dim=2, keepdim=True)


# ---------------------------------------------------------------- leakage penalty

def test_leak_gradient_matches_autograd_float64():
    R, B, N, m = 3, 40, 12, 5
    g = torch.Generator().manual_seed(1)
    h = torch.randn(R, B, m, generator=g, dtype=torch.float64)
    D = unit_rows(R, N, m, seed=2)
    D[:, 3] = D[:, 2] + 0.01 * torch.randn(R, m, generator=g, dtype=torch.float64)      # a twin pair
    D[:, 3] /= D[:, 3].norm(dim=1, keepdim=True)
    on = torch.rand(R, B, N, generator=g) < 0.2
    q = torch.rand(R, N, generator=g, dtype=torch.float64) + 0.5
    for qq in (None, q):
        for thr in (None, 0.9):
            hh = h.clone().requires_grad_(True)
            val_ref = leak_value(hh, D, on, qq, twin_thr=thr)
            val_ref.sum().backward()
            val, dh, _ = leak_value_and_grad(h, D, on, qq, twin_thr=thr)
            assert torch.allclose(val, val_ref.detach(), atol=1e-12)
            assert torch.allclose(dh, hh.grad, atol=1e-9), (qq is None, thr, (dh - hh.grad).abs().max())


def test_twin_filter_removes_the_near_duplicate_of_an_on_entry():
    m = 4
    D = torch.zeros(1, 3, m, dtype=torch.float64)
    D[0, 0, 0] = 1.0
    D[0, 1] = torch.tensor([0.99, 0.141, 0, 0]).double()
    D[0, 1] /= D[0, 1].norm()
    D[0, 2, 2] = 1.0
    on = torch.tensor([[[True, False, False]]])
    h = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    with_filter, _, twins = leak_value_and_grad(h, D, on, twin_thr=0.9)
    without, _, _ = leak_value_and_grad(h, D, on, twin_thr=None)
    assert twins == 1
    assert with_filter.item() == pytest.approx(0.0, abs=1e-12)          # entry 1 is a twin of the on entry 0: not counted
    assert without.item() == pytest.approx(0.99 ** 2 / (0.99 ** 2 + 0.141 ** 2), rel=1e-6)


def test_leak_is_scale_free_and_skips_tiny_samples():
    R, B, N, m = 2, 10, 6, 4
    g = torch.Generator().manual_seed(3)
    h = torch.randn(R, B, m, generator=g, dtype=torch.float64)
    D = unit_rows(R, N, m)
    on = torch.rand(R, B, N, generator=g) < 0.3
    v1, g1, _ = leak_value_and_grad(h, D, on, twin_thr=None)
    v2, g2, _ = leak_value_and_grad(3.0 * h, D, on, twin_thr=None)
    assert torch.allclose(v1, v2, atol=1e-12)
    assert torch.allclose(g1, 3.0 * g2, atol=1e-9)                      # gradient shrinks as 1/|h|
    h[:, 0] = 0.0
    v3, g3, _ = leak_value_and_grad(h, D, on, twin_thr=None)
    assert torch.isfinite(g3).all() and (g3[:, 0] == 0).all()


def test_oracle_leakage_with_one_feature_per_sample_equals_the_stage1_overlap():
    """One sample per feature, true rows and true active set: mean leak = (n - 1) x the Stage 1 plain penalty."""
    n, m = 30, 6
    g = torch.Generator().manual_seed(4)
    W = torch.randn(1, n, m, generator=g, dtype=torch.float64)
    x = torch.eye(n, dtype=torch.float64).unsqueeze(0) * 1.3            # sample i has only feature i on, value 1.3
    h = torch.bmm(x, W)
    D, on = oracle_readings(W, x)
    val, _, _ = leak_value_and_grad(h, D, on, twin_thr=None)
    P = penalty_value(W, torch.ones(1, n, dtype=torch.float64))
    assert val.item() == pytest.approx((n - 1) * P.item(), rel=1e-9)


# ---------------------------------------------------------------- SAEs

KINDS = ["relu_l1", "batchtopk", "jumprelu"]


def sparse_data(R, B, m, K, seed):
    """h = values on a few of K fixed unit directions (about one on per sample)."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(R, K, m, generator=g)
    dirs = dirs / dirs.norm(dim=2, keepdim=True)
    on = torch.rand(R, B, K, generator=g) < (1.0 / K)
    vals = torch.rand(R, B, K, generator=g) * 2
    return torch.bmm(on * vals, dirs), dirs


@pytest.mark.parametrize("kind", KINDS)
def test_sae_trains_and_keeps_unit_decoder_rows(kind):
    R, m, N, K = 2, 8, 16, 12
    sae = SAEBank(R, m, N, kind=kind, k=1.0, l1=3e-3, l0=3e-3, lr=3e-3, seed=0)
    first = None
    for step in range(400):
        h, _ = sparse_data(R, 256, m, K, seed=step)
        fvu = sae.step(h)
        if first is None:
            first = fvu.clone()
    assert torch.isfinite(sae.W_dec).all() and torch.isfinite(fvu).all()
    assert (fvu < first).all(), (kind, first, fvu)
    assert torch.allclose(sae.W_dec.norm(dim=2), torch.ones(R, N), atol=1e-5)
    D, on, z = sae.readings(h)
    assert not D.requires_grad and not z.requires_grad and on.dtype == torch.bool
    assert D.shape == (R, N, m) and on.shape == (R, 256, N)


def test_batchtopk_keeps_exactly_the_requested_number_of_codes():
    R, m, N, B = 2, 6, 20, 100
    sae = SAEBank(R, m, N, kind="batchtopk", k=1.5)
    h = torch.randn(R, B, m)
    z, _ = sae._codes_train(h)
    assert ((z > 0).sum(dim=(1, 2)) <= int(1.5 * B)).all()
    assert ((z > 0).sum(dim=(1, 2)) >= int(1.5 * B) - 2).all()           # ties / non-positive values may drop a few


def test_stacked_saes_are_independent():
    m, N = 6, 10
    data = [sparse_data(1, 64, m, 8, seed=s)[0] for s in range(15)]
    data2 = [sparse_data(1, 64, m, 8, seed=100 + s)[0] for s in range(15)]
    one = SAEBank(1, m, N, kind="relu_l1", seed=5)
    two = SAEBank(2, m, N, kind="relu_l1", seed=5)
    for a, b in zip(data, data2):
        one.step(a)
        two.step(torch.cat([a, b]))
    assert torch.allclose(one.W_dec[0], two.W_dec[0], atol=1e-5)


# ---------------------------------------------------------------- trainer integration

from toy.batched import train_group          # noqa: E402
from toy.train import RunConfig              # noqa: E402


def tiny(**kw):
    base = dict(alpha=1.0, width=5, gamma=-0.5, seed=0, n_features=40, steps=300, batch_size=128,
                warmup_steps=30, test_batches=3, log_every=50, sae_n=40, sae_gate=2.0, sae_ramp=50)
    base.update(kw)
    return RunConfig(**base)


def test_run_ids_distinguish_the_stage2_arms():
    ids = {tiny().run_id(), tiny(penalty="leak_oracle", lam=1.0).run_id(),
           tiny(penalty="leak_sae", lam=1.0).run_id(), tiny(penalty="leak_sae", lam=1.0, sae_type="relu_l1").run_id(),
           tiny(penalty="leak_sae", lam=3.0).run_id()}
    assert len(ids) == 5


def test_oracle_leakage_trains_and_lowers_overlap():
    base, pen = train_group([tiny(), tiny(penalty="leak_oracle", lam=3.0)], device="cpu")
    assert base["status"] == pen["status"] == "ok"
    assert pen["mso_all"] < base["mso_all"], (pen["mso_all"], base["mso_all"])
    assert pen["leak_last"] >= 0 and pen["test_loss"] == pen["test_loss"]


@pytest.mark.parametrize("kind", KINDS)
def test_sae_leakage_runs_and_leaves_plain_runs_unchanged(kind):
    plain, leak = train_group([tiny(sae_type=kind), tiny(penalty="leak_sae", lam=1.0, sae_type=kind)], device="cpu")
    alone = train_group([tiny(sae_type=kind)], device="cpu")[0]
    assert leak["status"] == "ok" and leak["sae_gate_step"] >= 0           # gate (placeholder 2.0) is met at once
    assert leak["mso_dec"] == leak["mso_dec"]
    assert torch.allclose(plain["W"], alone["W"], atol=1e-5)                # a leakage run in the group does not touch the others
