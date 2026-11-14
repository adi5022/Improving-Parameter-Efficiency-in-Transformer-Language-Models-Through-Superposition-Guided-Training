"""Step B: the 10-seed weight decay baseline (the yardstick a new penalty has to beat).

    python scripts/yardstick_10seed.py

Pools seeds 0-2 from G-A (gamma -1, -0.55, 0) and G-C (gamma -2, -0.3) with seeds 3-9 from G-D
(results/stage0/G-D/summary.jsonl), all at alpha = 1, and prints for every width and gamma the mean
test error over seeds, the spread, the best gamma, and how much better the best gamma is than
plain weight decay (gamma = 0). It also prints the error a model of 1.5x the width reaches, the
number the Stage 1 test has to match. Writes results/stage0/yardstick_10seed.json.
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAMMAS = [-2.0, -1.0, -0.55, -0.3, 0.0]


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    rows = {}
    sources = [("G-A", ROOT / "results/stage0/G-A/summary.jsonl"),
               ("G-C", ROOT / "results/colab/fp32/summary.jsonl"),
               ("G-D", ROOT / "results/stage0/G-D/summary.jsonl")]
    for tag, path in sources:
        for r in load(path):
            if r["alpha"] == 1.0 and r["gamma"] in GAMMAS and r["status"] == "ok":
                rows[r["run_id"]] = r
    by = defaultdict(list)
    for r in rows.values():
        by[(r["width"], r["gamma"])].append(r["test_loss"])
    widths = sorted({k[0] for k in by})
    short = [(w, g) for (w, g), v in by.items() if len(v) != 10]
    print(f"{len(rows)} runs pooled; cells with other than 10 seeds: {short or 'none'}\n")

    print("mean test error over 10 seeds (relative s.d. in brackets); * = lowest mean at that width")
    print("width | " + " | ".join(f"g={g:+.2f}      " for g in GAMMAS))
    best = {}
    for w in widths:
        mean = {g: statistics.mean(by[(w, g)]) for g in GAMMAS}
        b = min(mean, key=mean.get)
        sd = {g: statistics.stdev(by[(w, g)]) / mean[g] for g in GAMMAS}
        best[w] = {"gamma": b, "mean": mean[b], "sem": statistics.stdev(by[(w, b)]) / math.sqrt(len(by[(w, b)])),
                   "mean_gamma0": mean[0.0], "all_means": {str(g): mean[g] for g in GAMMAS}}
        print(f"{w:>5} | " + " | ".join(f"{mean[g]:.2e}({100 * sd[g]:2.0f}%)" + ("*" if g == b else " ") for g in GAMMAS))

    print("\nyardstick L*(m): best gamma per width")
    print("width | best gamma | L*(m)     | +/- s.e.m. | gamma 0 / L*(m) | L*(1.5m) by log-log interpolation | L*(m) / L*(1.5m)")
    xs = [math.log(w) for w in widths]
    ys = [math.log(best[w]["mean"]) for w in widths]
    out = {}
    for i, w in enumerate(widths):
        target = 1.5 * w
        if target <= widths[-1]:
            lo = max(j for j in range(len(widths)) if widths[j] <= target)
            hi = min(j for j in range(len(widths)) if widths[j] >= target)
            t = 0 if lo == hi else (math.log(target) - xs[lo]) / (xs[hi] - xs[lo])
            l15 = math.exp(ys[lo] + t * (ys[hi] - ys[lo]))
        else:
            l15 = float("nan")
        b = best[w]
        out[w] = {**b, "L_1.5m": l15}
        print(f"{w:>5} | {b['gamma']:+8.2f}   | {b['mean']:.3e} | {100 * b['sem'] / b['mean']:5.1f}%     | {b['mean_gamma0'] / b['mean']:6.2f}x          | "
              f"{l15:.3e}                           | {b['mean'] / l15:.2f}x")
    Path(ROOT / "results/stage0/yardstick_10seed.json").write_text(json.dumps({str(k): v for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
