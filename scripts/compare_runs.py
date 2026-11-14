"""Compare two sweeps of the same grid run-by-run (for example float32 vs fp16).

    python scripts/compare_runs.py results/stage0/G-C/summary.jsonl results/stage0/G-C-fp16/summary.jsonl

Matches runs by run_id and prints, for each, the relative difference in test error and the
difference in the fraction of features kept. It also prints the seed-to-seed spread of the
first file (the normal noise between runs), so you can see whether the difference between
the two sweeps is small next to it.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return {r["run_id"]: r for r in rows}


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("no run_id in common: are these the same grid?")
    print(f"{len(common)} runs in common ({len(a)} in first file, {len(b)} in second)\n")
    print(f"{'run':24s} {'first':>10s} {'second':>10s} {'rel diff':>9s}  kept first/second")
    diffs = []
    for rid in common:
        x, y = a[rid], b[rid]
        d = (y["test_loss"] - x["test_loss"]) / x["test_loss"]
        diffs.append(abs(d))
        print(f"{rid:24s} {x['test_loss']:10.3e} {y['test_loss']:10.3e} {100 * d:+8.2f}%  {x['phi_half']:.3f}/{y['phi_half']:.3f}")
    by = defaultdict(list)
    for r in a.values():
        by[(r["alpha"], r["width"], r["gamma"])].append(r["test_loss"])
    spread = [statistics.pstdev(v) / statistics.mean(v) for v in by.values() if len(v) > 1]
    print(f"\ndifference between the two sweeps: median {100 * statistics.median(diffs):.2f}%, worst {100 * max(diffs):.2f}%")
    if spread:
        print(f"seed-to-seed spread in the first file: median {100 * statistics.median(spread):.2f}%, worst {100 * max(spread):.2f}%")
        ratio = statistics.median(diffs) / statistics.median(spread)
        print(f"median difference is {ratio:.2f}x the median seed spread (well under 1x means the two sweeps agree within normal noise)")
    prec = {r.get("precision", "fp32") for r in a.values()} | {r.get("precision", "fp32") for r in b.values()}
    print("precision labels found:", ", ".join(sorted(prec)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
