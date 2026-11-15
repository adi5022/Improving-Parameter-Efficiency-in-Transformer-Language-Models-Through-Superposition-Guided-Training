"""Stage 2 pilot (grid G-S2P) analysis: every model is re-scored on ONE test set on this machine, so that the
GPU-dependent test draw of the Kaggle runs cannot create or hide a difference.

    python scripts/s2_pilot_analysis.py

Compared on seeds 0-2, per width: the baseline at the best gamma (G-A / G-C), the Stage 1 selected weighted arm
(G-S1F), the oracle-label leakage penalty (ORA-L) and the SAE-label leakage penalty (SAE-L, width 63).
Reported: mean test error on the common test set, the paired change against the baseline of the same seed
(mean +- standard error over the 3 seeds), the true mean squared overlap of the weight rows.
"""

import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from toy import metrics                       # noqa: E402
from toy.data import feature_probs, sample_batch   # noqa: E402
from toy.model import ToyModel                # noqa: E402
from toy.train import TEST_SEED               # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
STAGE0 = ROOT / "results" / "stage0"
GSTAR = {15: -0.55, 25: -1.0, 63: -2.0}
S1_LAM = {15: 0.3, 25: 3.0, 63: 1.0}
probs = feature_probs(1000, 1.0, device=dev)


def score(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = ck["config"]
    model = ToyModel(cfg["n_features"], cfg["width"]).to(dev)
    with torch.no_grad():
        model.W.copy_(ck["W"].to(dev))
        model.b.copy_(ck["b"].to(dev))
    model.eval()
    loss = metrics.test_loss(model, probs, sample_batch, 100 * 2048, 2048, TEST_SEED)
    return loss, metrics.mean_squared_overlap(ck["W"].to(dev)), ck


def folder_for_baseline(w):
    return "G-C" if w == 63 else "G-A"


rows = {}          # (width, arm) -> {seed: (loss, overlap, extra)}
for w in (15, 25, 63):
    g = GSTAR[w]
    for s in (0, 1, 2):
        l, o, _ = score(STAGE0 / folder_for_baseline(w) / f"a1_m{w}_g{g:+.2f}_s{s}.pt")
        rows.setdefault((w, "baseline"), {})[s] = (l, o, {})
        l, o, _ = score(STAGE0 / "G-S1F" / f"a1_m{w}_g{g:+.2f}_s{s}_pweighted{S1_LAM[w]:g}.pt")
        rows.setdefault((w, f"Stage 1 weighted, lam {S1_LAM[w]:g}"), {})[s] = (l, o, {})
for p in sorted((STAGE0 / "G-S2P").glob("*.pt")):
    l, o, ck = score(p)
    c = ck["config"]
    if c["penalty"] == "leak_oracle":
        arm = f"ORA-L lam {c['lam']:g}"
    else:
        arm = f"SAE-L {c['sae_type']} lam {c['lam']:g}"
    extra = {k: ck[k] for k in ("leak_last", "sae_gate_step", "sae_fvu_ema", "mso_dec") if k in ck}
    rows.setdefault((c["width"], arm), {})[c["seed"]] = (l, o, extra)

out = []
for w in (15, 25, 63):
    base = rows[(w, "baseline")]
    bl = st.mean(v[0] for v in base.values())
    print(f"\n=== width {w}   (baseline at gamma {GSTAR[w]}: test error {bl:.3e}, true overlap {st.mean(v[1] for v in base.values()):.4f})")
    print(f"{'arm':40s} {'test error':>11s} {'vs baseline (paired)':>26s} {'true overlap':>13s}  extra")
    for (ww, arm), d in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if ww != w or arm == "baseline" or len(d) < 3:
            continue
        diffs = [100 * (d[s][0] / base[s][0] - 1) for s in (0, 1, 2)]
        m, se = st.mean(diffs), (st.pstdev(diffs) * (3 / 2) ** 0.5) / 3 ** 0.5
        ex = ""
        if "sae_gate_step" in d[0][2]:
            ex = (f"gate at step {[d[s][2]['sae_gate_step'] for s in (0, 1, 2)]}, SAE FVU {st.mean(d[s][2]['sae_fvu_ema'] for s in (0, 1, 2)):.2f}, "
                  f"SAE overlap {st.mean(d[s][2]['mso_dec'] for s in (0, 1, 2)):.4f}")
        print(f"{arm:40s} {st.mean(v[0] for v in d.values()):11.3e} {m:+10.1f}% +- {se:4.1f}% {'':6s} {st.mean(v[1] for v in d.values()):13.4f}  {ex}")
        out.append({"width": w, "arm": arm, "test_error": st.mean(v[0] for v in d.values()), "paired_change_pct": m, "paired_se_pct": se,
                    "true_overlap": st.mean(v[1] for v in d.values()), "baseline_test_error": bl})
(STAGE0 / "G-S2P" / "analysis.json").write_text(json.dumps(out, indent=1))
print("\nnegative % = lower error than the baseline. Seeds 0-2 only (3 pairs): a pilot, not a verdict.")
