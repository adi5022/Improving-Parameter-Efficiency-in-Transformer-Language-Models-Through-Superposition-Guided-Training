"""S2-A SAE bake-off (decision D-S2-1; proposals/stage_2_design_proposal.md sections 2 and 7).

    python scripts/s2a_bakeoff.py --tag a1 --seeds 0                 # A1 screen: one seed, all 18 configs
    python scripts/s2a_bakeoff.py --tag a2 --seeds 0 1 2 --types batchtopk --ns 1000   # A2 confirm (chosen configs only)

Question: on FROZEN toy models whose true feature arrows we know, how well does an SAE trained on the hidden
activation h = xW recover those arrows, and does the overlap it reads off (M_dec) track the true overlap (M_true),
including the DIRECTION of the difference between a baseline model and a penalty-trained model?

Frozen models, per width m in {15, 25, 63} and seed: the baseline at its best weight decay gamma* (15: -0.55,
25: -1.0, 63: -2.0) and the Stage 1 selected penalty arm (read from results/stage0/G-S1F/verdict.json). Weights are
read from results/stage0/*; a missing baseline is retrained with the same code and saved to results/stage0/S2-A/frozen.
SAEs: type in {relu_l1, batchtopk, jumprelu} x 3 sparsity values x N in {1000, 2000}. All SAE hyperparameters here
(sparsity grids, learning rate 3e-3, steps, bandwidth) are PLACEHOLDERS for the screen, not tuned.

Metrics per trained SAE (true arrows are the rows of W; evaluated on fresh data, seed 30000):
  MCC        mean |cos| over a one-to-one (Hungarian) matching of the true arrows to decoder columns
  uniqueness distinct best-matching true arrows / number of alive latents
  F1         each alive latent as a detector for its best-matching feature (on when latent > 0 vs feature on)
  M_dec      mean squared cosine over pairs of alive decoder columns;  ratio = M_dec / M_true
  Spearman   rank correlation of each feature's true total overlap with its matched latent's total overlap
  FVU, L0, dead fraction
  direction  d_meas = M_dec(baseline) - M_dec(penalised), d_true = M_true(baseline) - M_true(penalised), same seed
Selection score (my concrete reading of the D-S2-1 rule; to be confirmed by the user): per (width, seed) mean of
  MCC, D = clip(d_meas / d_true, 0, 1) if d_meas > 0 else 0, and R = max(0, 1 - |ratio - 1|) (ratio averaged over the two
  model types), then averaged over widths and seeds. No configuration is chosen automatically; the table is for review.
"""

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from toy import metrics                                    # noqa: E402
from toy.data import feature_probs, sample_batch           # noqa: E402
from toy.sae import SAEBank                                # noqa: E402
from toy.train import RunConfig                            # noqa: E402

GSTAR = {15: -0.55, 25: -1.0, 63: -2.0}
SPARSITY = {"relu_l1": [0.01, 0.03, 0.1], "batchtopk": [0.5, 1.0, 2.0], "jumprelu": [0.03, 0.1, 0.3]}
EVAL_SEED, TRAIN_SEED = 30_000, 20_000
N_FEATURES = 1000
FROZEN = ROOT / "results/stage0/S2-A/frozen"


# ------------------------------------------------------------------ frozen models
def find(run_id):
    for d in ("G-S1F", "G-S1P", "G-D", "G-A"):
        p = ROOT / "results/stage0" / d / f"{run_id}.pt"
        if p.exists():
            return torch.load(p, weights_only=False)
    p = FROZEN / f"{run_id}.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    for p in (ROOT / "results").glob(f"**/{run_id}.pt"):
        return torch.load(p, weights_only=False)
    return None


def frozen_model(width, seed, kind, selected, device):
    g = GSTAR[width]
    rid = f"a1_m{width}_g{g:+.2f}_s{seed}"
    if kind == "penalised":
        arm, lam = selected[width]
        rid += f"_p{arm}{lam:g}"
    res = find(rid)
    if res is None:
        if kind == "penalised":
            raise FileNotFoundError(f"penalised weights {rid} not found")
        from toy.batched import train_group
        print(f"  retraining missing baseline {rid}", flush=True)
        res = train_group([RunConfig(alpha=1.0, width=width, gamma=g, seed=seed)], device=device)[0]
        FROZEN.mkdir(parents=True, exist_ok=True)
        torch.save(res, FROZEN / f"{rid}.pt")
    return res["W"].to(device).float(), rid


# ------------------------------------------------------------------ metrics for one SAE
@torch.no_grad()
def evaluate(sae, r, W, x_eval):
    h = x_eval @ W                                           # [S,m]
    z = sae.encode_one(r, h)
    D = sae.W_dec[r].detach()                                # [N,m] unit rows
    N = D.shape[0]
    fires = z > 0
    alive = fires.any(0)
    n_alive = int(alive.sum())
    U = W / W.norm(dim=1, keepdim=True).clamp_min(1e-2)
    C = (D @ U.T).abs()                                      # [N,n]
    rows, cols = linear_sum_assignment((-C).cpu().numpy())
    mcc = float(C[rows, cols].mean())
    best = C.argmax(1)
    uniq = len(set(best[alive].tolist())) / max(1, n_alive)
    x_on = x_eval > 0
    xs = x_on[:, best]
    tp = (fires & xs).sum(0).float()
    prec, rec = tp / fires.sum(0).clamp_min(1), tp / xs.sum(0).clamp_min(1)
    f1 = (2 * prec * rec / (prec + rec).clamp_min(1e-12))[alive].mean().item() if n_alive else 0.0
    Dn = D[alive]
    Gd = (Dn @ Dn.T) ** 2
    m_dec = float((Gd.sum() - Gd.diagonal().sum()) / max(1, n_alive * (n_alive - 1)))
    m_true = metrics.mean_squared_overlap(W)
    # total overlap per feature (true) vs per matched latent (SAE): rank correlation
    Gt = (U @ U.T) ** 2
    T_true = Gt.sum(1) - Gt.diagonal()
    Gf = (D @ Dn.T) ** 2                                      # [N, n_alive]
    S_sae = Gf.sum(1) - (D * D).sum(1) ** 2 * alive.float()   # remove the self term for alive latents
    sp = spearmanr(T_true[cols].cpu().numpy(), S_sae[rows].cpu().numpy()).statistic
    rec_h = z @ D + sae.b_dec[r]
    fvu = float(((rec_h - h) ** 2).sum() / ((h - h.mean(0)) ** 2).sum().clamp_min(1e-12))
    return {"mcc": mcc, "uniqueness": uniq, "f1": f1, "m_dec": m_dec, "m_true": m_true, "ratio": m_dec / m_true,
            "spearman": float(sp), "fvu": fvu, "l0": float(fires.float().sum(1).mean()),
            "dead_fraction": 1 - n_alive / N}


# ------------------------------------------------------------------ one group of SAEs (same type, N, width)
def run_group(width, seeds, kind, N, steps, lr, models, eval_samples, device):
    """models[(seed, 'baseline'|'penalised')] = W. One SAE per (seed, model kind, sparsity value)."""
    spars = SPARSITY[kind]
    keys = [(s, mk, sp) for s in seeds for mk in ("baseline", "penalised") for sp in spars]
    W_stack = torch.stack([models[(s, mk)] for s, mk, _ in keys])            # [R,n,m]
    R = len(keys)
    sp_t = [sp for _, _, sp in keys]
    kw = {"k": sp_t} if kind == "batchtopk" else ({"l1": sp_t} if kind == "relu_l1" else {"l0": sp_t})
    sae = SAEBank(R, width, N, kind=kind, lr=lr, device=device, seed=7000 + width, **kw)
    probs = feature_probs(N_FEATURES, 1.0, device=device)
    g = torch.Generator(device=device).manual_seed(TRAIN_SEED + width)
    t0 = time.perf_counter()
    for step in range(steps):
        x = sample_batch(2048, probs, g)
        with torch.enable_grad():
            sae.step(torch.matmul(x.unsqueeze(0), W_stack))
    secs = time.perf_counter() - t0
    ge = torch.Generator(device=device).manual_seed(EVAL_SEED)
    x_eval = sample_batch(eval_samples, probs, ge)
    rows = []
    for r, (s, mk, sp) in enumerate(keys):
        ev = evaluate(sae, r, W_stack[r], x_eval)
        rows.append({"width": width, "seed": s, "model": mk, "type": kind, "N": N, "sparsity": sp, "steps": steps,
                     "sae_lr": lr, "train_seconds_group": secs, "group_size": R, **ev})
    return rows


# ------------------------------------------------------------------ analysis
def add_direction(rows):
    """Attach the direction test to every penalised row (same width, seed, type, N, sparsity)."""
    base = {(r["width"], r["seed"], r["type"], r["N"], r["sparsity"]): r for r in rows if r["model"] == "baseline"}
    for r in rows:
        if r["model"] != "penalised":
            continue
        b = base[(r["width"], r["seed"], r["type"], r["N"], r["sparsity"])]
        d_meas, d_true = b["m_dec"] - r["m_dec"], b["m_true"] - r["m_true"]
        r["d_meas"], r["d_true"] = d_meas, d_true
        r["direction_ok"] = bool(d_meas > 0 and d_true > 0)
        r["direction_score"] = min(1.0, max(0.0, d_meas / d_true)) if (d_meas > 0 and d_true > 0) else 0.0
    return rows


def summarise(rows):
    cfg = defaultdict(list)
    for r in rows:
        cfg[(r["type"], r["sparsity"], r["N"])].append(r)
    out = []
    for (t, sp, N), rs in sorted(cfg.items()):
        pair = defaultdict(dict)                                 # (width, seed) -> model -> row
        for r in rs:
            pair[(r["width"], r["seed"])][r["model"]] = r
        scores, dirs, ratios = [], [], []
        for (w, s), d in pair.items():
            if len(d) < 2:
                continue
            ratio = st.mean([d["baseline"]["ratio"], d["penalised"]["ratio"]])
            R_ = max(0.0, 1 - abs(ratio - 1))
            scores.append((st.mean([d["baseline"]["mcc"], d["penalised"]["mcc"]]) + d["penalised"]["direction_score"] + R_) / 3)
            dirs.append(d["penalised"]["direction_ok"])
            ratios.append(ratio)
        out.append({"type": t, "sparsity": sp, "N": N, "score": st.mean(scores) if scores else float("nan"),
                    "mcc": st.mean(r["mcc"] for r in rs), "f1": st.mean(r["f1"] for r in rs),
                    "uniq": st.mean(r["uniqueness"] for r in rs), "ratio": st.mean(ratios) if ratios else float("nan"),
                    "spearman": st.mean(r["spearman"] for r in rs), "fvu": st.mean(r["fvu"] for r in rs),
                    "l0": st.mean(r["l0"] for r in rs), "dead": st.mean(r["dead_fraction"] for r in rs),
                    "direction_ok": f"{sum(dirs)}/{len(dirs)}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--widths", type=int, nargs="+", default=[15, 25, 63])
    ap.add_argument("--types", nargs="+", default=["relu_l1", "batchtopk", "jumprelu"])
    ap.add_argument("--ns", type=int, nargs="+", default=[1000, 2000])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-samples", type=int, default=10_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    verdict = json.loads((ROOT / "results/stage0/G-S1F/verdict.json").read_text())
    selected = {int(w): (i["selected"]["arm"], i["selected"]["lam"]) for w, i in verdict["widths"].items()}
    out_dir = ROOT / "results/stage0/S2-A" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    for w in a.widths:
        models = {}
        for s in a.seeds:
            for mk in ("baseline", "penalised"):
                models[(s, mk)], _ = frozen_model(w, s, mk, selected, a.device)
        for kind in a.types:
            for N in a.ns:
                tg = time.perf_counter()
                rs = run_group(w, a.seeds, kind, N, a.steps, a.lr, models, a.eval_samples, a.device)
                rows += rs
                print(f"width {w:3d} {kind:10s} N={N}: {len(rs)} SAEs, {time.perf_counter() - tg:5.0f} s "
                      f"(total {time.perf_counter() - t0:5.0f} s)", flush=True)
    rows = add_direction(rows)
    (out_dir / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summ = summarise(rows)
    (out_dir / "summary.json").write_text(json.dumps({"args": vars(a), "selected_penalty_arms": selected, "configs": summ}, indent=2))
    print(f"\nS2-A {a.tag}: seeds {a.seeds}, {a.steps} SAE steps, SAE lr {a.lr}  (placeholders; no config is chosen here)")
    print(f"{'type':10s} {'sparsity':>8s} {'N':>5s} | {'score':>5s} {'MCC':>5s} {'F1':>5s} {'uniq':>5s} {'M ratio':>7s} {'Spear':>5s} {'FVU':>6s} {'L0':>5s} {'dead':>5s} dir")
    for c in sorted(summ, key=lambda c: -c["score"]):
        print(f"{c['type']:10s} {c['sparsity']:8g} {c['N']:5d} | {c['score']:5.2f} {c['mcc']:5.2f} {c['f1']:5.2f} {c['uniq']:5.2f} "
              f"{c['ratio']:7.2f} {c['spearman']:5.2f} {c['fvu']:6.3f} {c['l0']:5.2f} {c['dead']:5.2f} {c['direction_ok']}")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
