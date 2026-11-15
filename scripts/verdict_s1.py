"""Stage 1 verdict (gate G1) for the overlap penalty, with the rule fixed before the final data (decision D-S1-4).

    python scripts/verdict_s1.py

Data: the penalty runs in results/stage0/G-S1P (pilot; only its lambda = 0.1 runs on seeds 0-2 are used) and
results/stage0/G-S1F (final grid), and the no-penalty baseline at the best weight decay gamma* per width
(15: -0.55, 25: -1.0, 63: -2.0) from G-A, G-C and G-D (results/stage0/yardstick_10seed.json gives L*(m)).

Procedure (fixed in advance):
  1. Candidates per width: arm in {plain, weighted} x lambda in {0.1, 0.3, 1, 3}, both at gamma*.
  2. Select the candidate with the lowest mean test error on seeds 0-2 (selection seeds).
  3. Judge it on seeds 3-9 only (judging seeds): improvement = 1 - mean(penalty) / L*(m), L*(m) = the 10-seed baseline
     mean at gamma*; the standard error is the combined s.e.m. of the 7 penalty runs and the 10 baseline runs.
  4. PASS if, at 2 of 3 widths (15, 25, 63), improvement > 10% and > 3 combined standard errors.
     INCONCLUSIVE if exactly one width passes, or any width improves by 3-10%.  FAIL otherwise.
     Stronger success mark: the judged mean is within one s.e.m. of L*(1.5 m) at any width.
Extras (not part of the verdict): the same improvement measured pair-wise against the baseline on the same seeds
(same data stream and initial weights), every candidate on the judging seeds, whether the selected lambda sits at the
edge of the grid, and the verdict if the baseline were the G-E gamma = -3 / -4 runs where those exist.
"""

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GSTAR = {15: -0.55, 25: -1.0, 63: -2.0}
LAMS = [0.1, 0.3, 1.0, 3.0]
SELECT_SEEDS = [0, 1, 2]
JUDGE_SEEDS = [3, 4, 5, 6, 7, 8, 9]
PASS_MARGIN, INCONCLUSIVE_FLOOR, K_SE = 0.10, 0.03, 3.0


def load(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def sem(xs):
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")


def main():
    base = {}                                   # (width, seed) -> test loss at gamma*
    for src in ("results/stage0/G-A", "results/colab/fp32", "results/stage0/G-D"):
        for r in load(ROOT / src / "summary.jsonl"):
            if r["alpha"] == 1.0 and r["status"] == "ok" and GSTAR.get(r["width"]) == r["gamma"] and r.get("penalty", "none") == "none":
                base[(r["width"], r["seed"])] = r["test_loss"]
    yard = json.loads((ROOT / "results/stage0/yardstick_10seed.json").read_text())

    pen = {}
    for src in ("G-S1P", "G-S1F"):
        for r in load(ROOT / "results/stage0" / src / "summary.jsonl"):
            if r.get("penalty", "none") in ("plain", "weighted") and r["status"] == "ok" and r["lam"] in LAMS \
                    and r["gamma"] == GSTAR.get(r["width"]):
                pen[r["run_id"]] = r
    by = defaultdict(dict)                      # (width, arm, lam) -> seed -> loss
    for r in pen.values():
        by[(r["width"], r["penalty"], r["lam"])][r["seed"]] = r["test_loss"]

    out = {"rule": {"pass_margin": PASS_MARGIN, "k_se": K_SE, "select_seeds": SELECT_SEEDS, "judge_seeds": JUDGE_SEEDS, "lams": LAMS},
           "widths": {}}
    n_pass, any_mid = 0, False
    for w in GSTAR:
        bl = [base[(w, s)] for s in range(10) if (w, s) in base]
        Lstar, Lse = st.mean(bl), sem(bl)
        L15 = yard[str(w)]["L_1.5m"]
        cands = {k: v for k, v in by.items() if k[0] == w}
        sel = {k: st.mean(v[s] for s in SELECT_SEEDS) for k, v in cands.items() if all(s in v for s in SELECT_SEEDS)}
        info = {"baseline_seeds": len(bl), "L_star": Lstar, "L_star_se": Lse, "L_1.5m": L15}
        if not sel or any(len([s for s in JUDGE_SEEDS if s in cands[k]]) < len(JUDGE_SEEDS) for k in sel):
            info["status"] = "incomplete: missing runs"
            out["widths"][w] = info
            print(f"\nwidth {w}: incomplete data ({len(cands)} candidates with runs)")
            continue
        best = min(sel, key=sel.get)
        j = [cands[best][s] for s in JUDGE_SEEDS]
        jm, jse = st.mean(j), sem(j)
        imp = 1 - jm / Lstar
        cse = math.sqrt(jse ** 2 + Lse ** 2)
        z = (Lstar - jm) / cse
        paired = [1 - cands[best][s] / base[(w, s)] for s in JUDGE_SEEDS if (w, s) in base]
        passed = imp > PASS_MARGIN and z > K_SE
        mid = (not passed) and INCONCLUSIVE_FLOOR <= imp <= PASS_MARGIN
        n_pass += passed
        any_mid = any_mid or mid
        info.update({
            "selected": {"arm": best[1], "lam": best[2], "mean_selection_seeds": sel[best]},
            "judged_mean": jm, "judged_se": jse, "improvement": imp, "combined_se": cse, "z": z,
            "paired_improvement_mean": st.mean(paired), "paired_improvement_se": sem(paired),
            "pass": passed, "inconclusive_band": mid,
            "within_1se_of_L_1.5m": abs(jm - L15) <= jse or jm <= L15,
            "needed_improvement_to_match_1.5m": 1 - L15 / Lstar,
            "selected_lambda_at_grid_top": best[2] == max(LAMS),
            "all_candidates_judging_seeds": {f"{k[1]}@{k[2]:g}": st.mean(cands[k][s] for s in JUDGE_SEEDS) / Lstar - 1 for k in sorted(cands)},
            "status": "ok",
        })
        out["widths"][w] = info
        print(f"\nwidth {w}: baseline L*(m) = {Lstar:.3e} (+/- {100 * Lse / Lstar:.1f}%, {len(bl)} seeds); L*(1.5m) = {L15:.3e} "
              f"(needs {100 * info['needed_improvement_to_match_1.5m']:.0f}% lower error)")
        print(f"  selected on seeds 0-2: {best[1]} lambda {best[2]:g}")
        print(f"  judged on seeds 3-9: {jm:.3e} (+/- {100 * jse / jm:.1f}%) -> improvement {100 * imp:+.1f}%, {z:.1f} combined s.e.  "
              f"{'PASS' if passed else ('inconclusive band' if mid else 'no pass')}")
        print(f"  pair-wise vs the same-seed baseline: {100 * st.mean(paired):+.1f}% (+/- {100 * sem(paired):.1f}%)"
              + ("   [selected lambda is the TOP of the grid]" if info["selected_lambda_at_grid_top"] else ""))
        print("  every candidate on judging seeds, change vs L*(m): " +
              ", ".join(f"{k}: {100 * v:+.0f}%" for k, v in info["all_candidates_judging_seeds"].items()))
    done = all(v.get("status") == "ok" for v in out["widths"].values())
    verdict = "INCOMPLETE" if not done else ("PASS" if n_pass >= 2 else ("INCONCLUSIVE" if (n_pass == 1 or any_mid) else "FAIL"))
    out["verdict"], out["widths_passed"] = verdict, int(n_pass)
    print(f"\nGATE G1: {verdict}  ({int(n_pass)} of 3 widths pass)")
    dest = ROOT / "results/stage0/G-S1F/verdict.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float))
    print("wrote", dest)


if __name__ == "__main__":
    main()
