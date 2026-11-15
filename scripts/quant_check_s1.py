"""Quantisation fragility check for the toy model (recommended in prior_art_search.md section 8; not part of gate G1).

    python scripts/quant_check_s1.py            # after scripts/verdict_s1.py has written verdict.json

Question: does a penalty-trained solution (arrows spread to the minimum overlap) lose more or less accuracy than the
baseline one when W is rounded to few bits? Per width, compares the baseline at gamma* with the penalty candidate that
the verdict script selected, on seeds 0-2 (the only baseline weights on disk; a baseline run that is missing is retrained here with the same code and settings). W is rounded to b bits with a symmetric uniform quantiser, either
one scale for the whole matrix ("tensor") or one scale per row ("row"); the bias stays in float32 (it is tiny).
Reports the test error after rounding relative to the float32 test error. Runs on the CPU so it never competes with a
GPU sweep. Writes results/stage0/G-S1F/quant_check.json.
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

GSTAR = {15: -0.55, 25: -1.0, 63: -2.0}
BITS = [8, 6, 4, 3]
SEEDS = [0, 1, 2]   # baseline weights for seeds 3-9 were not downloaded from the Colab/Kaggle runs; seeds 0-2 are on disk (G-A)


def quantise(W, bits, mode):
    qmax = 2 ** (bits - 1) - 1
    scale = (W.abs().amax(dim=1, keepdim=True) if mode == "row" else W.abs().max()) / qmax
    return (W / scale).round().clamp(-qmax, qmax) * scale


def find(run_id):
    for d in ("G-S1F", "G-S1P", "G-D", "G-A"):
        p = ROOT / "results/stage0" / d / f"{run_id}.pt"
        if p.exists():
            return torch.load(p, weights_only=False)
    for p in (ROOT / "results").glob(f"**/{run_id}.pt"):
        return torch.load(p, weights_only=False)
    raise FileNotFoundError(run_id)


def find_or_train(run_id, w, g):
    try:
        return find(run_id)
    except FileNotFoundError:
        if "_p" in run_id:
            raise
        from dataclasses import replace
        from toy.batched import train_group
        from toy.train import RunConfig
        seed = int(run_id.split("_s")[-1])
        print(f"  (retraining missing baseline {run_id})")
        return train_group([RunConfig(alpha=1.0, width=w, gamma=g, seed=seed)], device="cuda" if torch.cuda.is_available() else "cpu")[0]


def test_error(W, b, m):
    model = ToyModel(1000, m)
    with torch.no_grad():
        model.W.copy_(W)
        model.b.copy_(b)
    model.eval()
    return metrics.test_loss(model, feature_probs(1000, 1.0), sample_batch, 100 * 2048, 2048, TEST_SEED)


def main():
    verdict = json.loads((ROOT / "results/stage0/G-S1F/verdict.json").read_text())
    out = {}
    for w_s, info in verdict["widths"].items():
        w = int(w_s)
        if info.get("status") != "ok":
            continue
        arm, lam = info["selected"]["arm"], info["selected"]["lam"]
        g = GSTAR[w]
        groups = {"baseline": [f"a1_m{w}_g{g:+.2f}_s{s}" for s in SEEDS],
                  f"{arm} lambda {lam:g}": [f"a1_m{w}_g{g:+.2f}_s{s}_p{arm}{lam:g}" for s in SEEDS]}
        out[w] = {}
        print(f"\nwidth {w}: baseline vs selected penalty arm ({arm}, lambda {lam:g}); test error after rounding W, relative to float32")
        for name, ids in groups.items():
            runs = [find_or_train(i, w, g) for i in ids]
            ref = [test_error(r["W"], r["b"], w) for r in runs]
            row = {"fp32": st.mean(ref)}
            line = f"  {name:22s} fp32 {st.mean(ref):.3e}"
            for mode in ("tensor", "row"):
                for bits in BITS:
                    rel = [test_error(quantise(r["W"], bits, mode), r["b"], w) / e for r, e in zip(runs, ref)]
                    row[f"{mode}_{bits}bit"] = st.mean(rel)
                    line += f" | {mode[0]}{bits}: x{st.mean(rel):.2f}"
            out[w][name] = row
            print(line)
    dest = ROOT / "results/stage0/G-S1F/quant_check.json"
    dest.write_text(json.dumps(out, indent=2))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
