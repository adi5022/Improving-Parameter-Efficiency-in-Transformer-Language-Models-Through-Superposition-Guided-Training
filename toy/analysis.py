"""Stage 0 analysis: yardstick L*(m), width-scaling exponents, gates G0.1-G0.4, figures.

    python -m toy.analysis            # reads results/stage0/<grid>/, writes report + figures

Outputs: results/stage0/report.md, results/stage0/yardstick.json, figures/stage0/*.png.
Figures are drawn only from stored results.

Caveat on the yardstick: L*(m) is the minimum over gamma of seed-mean losses, so it is
biased slightly low (picking the best of 7 noisy means). With 3 seeds the bias is of the
order of the seed s.d., which the report prints next to every L*.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "stage0"
FIGURES = ROOT / "figures" / "stage0"
CONFIG = ROOT / "configs" / "stage0.json"


# ---------- loading and aggregation ----------

def load_summary(grid: str, results_dir: Path = RESULTS) -> list[dict]:
    path = results_dir / grid / "summary.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    latest = {r["run_id"]: r for r in rows}           # a rerun overrides an earlier line
    return list(latest.values())


def aggregate(rows: list[dict], key: str = "test_loss") -> dict:
    """{(alpha, width, gamma): {"mean", "sd", "n", "values"}} over seeds, ok runs only."""
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and math.isfinite(r[key]):
            groups[(r["alpha"], r["width"], r["gamma"])].append(r[key])
    out = {}
    for k, v in groups.items():
        a = np.asarray(v, dtype=float)
        out[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
                  "n": len(a), "values": v}
    return out


# ---------- yardstick and equivalent width ----------

def yardstick(agg: dict, alpha: float) -> dict:
    """{width: {"gamma": best gamma, "loss": L*(m), "sd": seed s.d. at that gamma}}."""
    best = {}
    for (a, m, g), s in agg.items():
        if a != alpha:
            continue
        if m not in best or s["mean"] < best[m]["loss"]:
            best[m] = {"gamma": g, "loss": s["mean"], "sd": s["sd"]}
    return dict(sorted(best.items()))


def interp_loglog(x: float, xs, ys) -> float:
    """Piecewise-linear interpolation in log-log space; nan outside [min(xs), max(xs)]."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if not (xs[0] <= x <= xs[-1]):
        return float("nan")
    return float(np.exp(np.interp(np.log(x), np.log(xs), np.log(ys))))


def equivalent_width(loss: float, ystick: dict) -> float:
    """Width m_eq at which the tuned baseline reaches `loss`: solves L*(m_eq) = loss.

    This is the Stage 1+ success measure: a width-m model with the penalty "wins" if
    m_eq / m >= 1.5 beyond seed noise. Requires L* to decrease with m (checked);
    nan if `loss` lies outside the measured range.
    """
    ms = np.array(list(ystick.keys()), float)
    ls = np.array([ystick[m]["loss"] for m in ystick], float)
    order = np.argsort(ms)
    ms, ls = ms[order], ls[order]
    if np.any(np.diff(ls) >= 0):
        raise ValueError("L*(m) is not strictly decreasing in m; equivalent width is ill-defined")
    if not (ls[-1] <= loss <= ls[0]):
        return float("nan")
    # np.interp needs increasing x: interpolate log m as a function of -log L
    return float(np.exp(np.interp(-np.log(loss), -np.log(ls), np.log(ms))))


# ---------- scaling exponents ----------

def fit_exponent(widths, losses) -> dict:
    """Least-squares fit log L = c - alpha_m log m. Returns exponent alpha_m and its s.e."""
    x, y = np.log(np.asarray(widths, float)), np.log(np.asarray(losses, float))
    if len(x) < 3:
        return {"exponent": float("nan"), "se": float("nan"), "n": len(x)}
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = float(resid @ resid) / (len(x) - 2)
    se = math.sqrt(s2 / float(((x - x.mean()) ** 2).sum()))
    return {"exponent": float(-coef[0]), "se": se, "n": len(x)}


def exponent_for(agg: dict, alpha: float, gamma: float) -> dict:
    pts = sorted((m, s["mean"]) for (a, m, g), s in agg.items() if a == alpha and g == gamma)
    if not pts:
        return {"exponent": float("nan"), "se": float("nan"), "n": 0}
    return fit_exponent([p[0] for p in pts], [p[1] for p in pts])


def per_seed_exponents(rows: list[dict], alpha: float, gamma: float) -> list[float]:
    by_seed = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and r["alpha"] == alpha and r["gamma"] == gamma:
            by_seed[r["seed"]].append((r["width"], r["test_loss"]))
    out = []
    for pts in by_seed.values():
        pts.sort()
        out.append(fit_exponent([p[0] for p in pts], [p[1] for p in pts])["exponent"])
    return out


# ---------- gates ----------

def evaluate_gates(rows_a: list[dict], rows_b: list[dict], gates: dict) -> list[dict]:
    out = []
    phi_a = aggregate(rows_a, "phi_half")
    loss_a = aggregate(rows_a, "test_loss")
    loss_b = aggregate(rows_b, "test_loss")

    g = gates["G0.1"]
    s = phi_a.get((g["alpha"], g["width"], g["strong_gamma"]))
    w = phi_a.get((g["alpha"], g["width"], g["weak_gamma"]))
    if s and w:
        ok = s["mean"] >= g["strong_min_phi_half"] and w["mean"] <= g["weak_max_phi_half"]
        detail = f"phi1/2 = {s['mean']:.3f} at gamma={g['strong_gamma']}, {w['mean']:.3f} at gamma={g['weak_gamma']}"
        out.append({"gate": "G0.1", "pass": ok, "detail": detail})
    else:
        out.append({"gate": "G0.1", "pass": None, "detail": "runs missing"})

    g = gates["G0.2"]
    e = exponent_for(loss_a, g["alpha"], g["gamma"])
    if e["n"]:
        ok = g["exponent_min"] <= e["exponent"] <= g["exponent_max"]
        out.append({"gate": "G0.2", "pass": ok,
                    "detail": f"alpha_m = {e['exponent']:.3f} +/- {e['se']:.3f} over {e['n']} widths"})
    else:
        out.append({"gate": "G0.2", "pass": None, "detail": "runs missing"})

    g = gates["G0.3"]
    parts, oks = [], []
    for alpha in g["alphas"]:
        e = exponent_for(loss_b, alpha, g["gamma"])
        if e["n"]:
            oks.append(abs(e["exponent"] - (alpha - 1)) <= g["tolerance"])
            parts.append(f"alpha={alpha}: alpha_m = {e['exponent']:.3f} (expected {alpha - 1:.2f})")
        else:
            oks.append(None)
            parts.append(f"alpha={alpha}: runs missing")
    out.append({"gate": "G0.3", "pass": None if None in oks else all(oks), "detail": "; ".join(parts)})
    return out


# ---------- figures ----------
# Colours: gamma is signed (weight decay pulls one way or the other), so it uses a
# diverging scale - blue for negative (strong superposition), grey at 0, red for
# positive (weak). Width is a magnitude, so it uses a single-hue blue ramp.

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GAMMA_COLORS = {-1.0: "#184f95", -0.55: "#2a78d6", -0.1: "#86b6ef", 0.0: "#6f6e6a",
                0.1: "#ee8a89", 0.55: "#e34948", 1.0: "#9e2a2a"}
WIDTH_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#1c5cab", "#104281"]


def _gamma_color(g: float) -> str:
    return GAMMA_COLORS.get(round(g, 2), MUTED)


def _ramp(k: int) -> list[str]:
    """k colours spread evenly over the blue ramp (light = small, dark = large)."""
    idx = np.linspace(0, len(WIDTH_RAMP) - 1, k).round().astype(int) if k > 1 else [len(WIDTH_RAMP) - 1]
    return [WIDTH_RAMP[i] for i in idx]


def _width_ticks(ax, widths):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator(sorted(widths)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")


def make_figures(rows_a: list[dict], rows_b: list[dict], out_dir: Path = FIGURES) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    if rows_a:
        loss = aggregate(rows_a, "test_loss")
        alpha = rows_a[0]["alpha"]
        gammas = sorted({k[2] for k in loss})

        # 1. loss vs width per gamma, with the yardstick
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        for g in gammas:
            pts = sorted((m, s["mean"], s["sd"]) for (a, m, gg), s in loss.items() if gg == g)
            ms, mu, sd = zip(*pts)
            ax.errorbar(ms, mu, yerr=[0 if math.isnan(v) else v for v in sd], color=_gamma_color(g),
                        linewidth=2, marker="o", markersize=5, capsize=2, label=f"γ = {g:+.2f}")
            ax.annotate(f"{g:+.2f}", (ms[-1], mu[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=MUTED)
        ys = yardstick(loss, alpha)
        ax.plot(list(ys), [v["loss"] for v in ys.values()], color=TEXT, linewidth=1, linestyle="--",
                label="L*(m): best γ per width", zorder=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        _width_ticks(ax, ys.keys())
        _style(ax, "width m", "test loss (MSE)", f"Test loss vs width, α = {alpha:g} (mean ± s.d. over seeds)")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        fig.tight_layout()
        p = out_dir / "loss_vs_width_GA.png"; fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

        # 2. represented fraction vs gamma, one line per width
        phi = aggregate(rows_a, "phi_half")
        widths = sorted({k[1] for k in phi})
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        for m, color in zip(widths, _ramp(len(widths))):
            pts = sorted((g, s["mean"]) for (a, mm, g), s in phi.items() if mm == m)
            gs, mu = zip(*pts)
            ax.plot(gs, mu, color=color, linewidth=2, marker="o", markersize=5, label=f"m = {m}")
        _style(ax, "weight decay γ", "represented fraction φ½", "Fraction of features stored (‖W_i‖ > ½) vs γ")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, frameon=False, ncol=2)
        fig.tight_layout()
        p = out_dir / "phi_vs_gamma_GA.png"; fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

        # 3. row-norm histograms, strongest vs weakest decay at the largest paper width
        import torch
        m_hist = 100 if 100 in widths else widths[-1]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), facecolor=SURFACE, sharey=True)
        for ax, g in zip(axes, (gammas[0], gammas[-1])):
            norms = []
            for r in rows_a:
                if r["width"] == m_hist and r["gamma"] == g and r["status"] == "ok":
                    path = RESULTS / "G-A" / f"{r['run_id']}.pt"
                    if path.exists():
                        norms.append(torch.load(path, weights_only=False)["row_norms"].numpy())
            if norms:
                ax.hist(np.concatenate(norms), bins=40, color=_gamma_color(g), edgecolor=SURFACE, linewidth=0.5)
            _style(ax, "row norm ‖W_i‖", "features" if ax is axes[0] else "", f"γ = {g:+.2f}, m = {m_hist}")
        fig.tight_layout()
        p = out_dir / "row_norms_GA.png"; fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    if rows_b:
        loss = aggregate(rows_b, "test_loss")
        alphas = sorted({k[0] for k in loss})
        gammas = sorted({k[2] for k in loss})
        fig, axes = plt.subplots(1, len(gammas), figsize=(4 * len(gammas), 4), facecolor=SURFACE, sharey=True)
        axes = np.atleast_1d(axes)
        ramp = _ramp(len(alphas))
        for ax, g in zip(axes, gammas):
            for i, a in enumerate(alphas):
                pts = sorted((m, s["mean"]) for (aa, m, gg), s in loss.items() if aa == a and gg == g)
                if pts:
                    ms, mu = zip(*pts)
                    ax.plot(ms, mu, color=ramp[i % len(ramp)], linewidth=2, marker="o", markersize=5,
                            label=f"α = {a:g}")
            ax.set_xscale("log"); ax.set_yscale("log")
            _width_ticks(ax, {k[1] for k in loss})
            _style(ax, "width m", "test loss (MSE)" if ax is axes[0] else "", f"γ = {g:+.2f}")
            ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        p = out_dir / "loss_vs_width_GB.png"; fig.savefig(p, dpi=150); plt.close(fig); written.append(p)
    return written


# ---------- report ----------

def main() -> int:
    spec = json.loads(CONFIG.read_text())
    rows_a, rows_b = load_summary("G-A"), load_summary("G-B")
    if not rows_a and not rows_b:
        print("No results in results/stage0/. Run python -m toy.sweep --grid G-A first.")
        return 1
    lines = ["# Stage 0 report (generated by toy/analysis.py)", "",
             f"Runs: G-A {len(rows_a)}, G-B {len(rows_b)}. Failed runs: "
             f"{sum(r['status'] != 'ok' for r in rows_a + rows_b)}.",
             f"Git commits: {sorted({r.get('git_commit') for r in rows_a + rows_b} - {None}) or 'none recorded'}", ""]

    if rows_a:
        loss = aggregate(rows_a, "test_loss")
        alpha = rows_a[0]["alpha"]
        ys = yardstick(loss, alpha)
        (RESULTS / "yardstick.json").write_text(json.dumps({str(k): v for k, v in ys.items()}, indent=2))
        lines += ["## Yardstick L*(m) (G-A)", "", "| m | best γ | L*(m) | seed s.d. | loss at γ=0 | L*(1.5m) interpolated |",
                  "|---|---|---|---|---|---|"]
        ms = list(ys)
        for m, v in ys.items():
            g0 = loss.get((alpha, m, 0.0))
            l15 = interp_loglog(1.5 * m, ms, [ys[k]["loss"] for k in ms])
            lines.append(f"| {m} | {v['gamma']:+.2f} | {v['loss']:.4e} | {v['sd']:.2e} | "
                         f"{g0['mean']:.4e} | {l15:.4e} |" if g0 else
                         f"| {m} | {v['gamma']:+.2f} | {v['loss']:.4e} | {v['sd']:.2e} | n/a | {l15:.4e} |")
        lines += ["", "## Seed s.d. of test loss (G0.4)", "", "| m \\ γ | " +
                  " | ".join(f"{g:+.2f}" for g in sorted({k[2] for k in loss})) + " |",
                  "|---" * (1 + len({k[2] for k in loss})) + "|"]
        for m in sorted({k[1] for k in loss}):
            cells = []
            for g in sorted({k[2] for k in loss}):
                s = loss.get((alpha, m, g))
                cells.append(f"{s['sd'] / s['mean'] * 100:.1f}%" if s and not math.isnan(s["sd"]) else "n/a")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines += ["", "(relative s.d. = s.d. / mean)", ""]
        lines += ["## Width-scaling exponents (G-A)", "", "| γ | α_m | s.e. | per-seed α_m |", "|---|---|---|---|"]
        for g in sorted({k[2] for k in loss}):
            e = exponent_for(loss, alpha, g)
            ps = ", ".join(f"{x:.3f}" for x in per_seed_exponents(rows_a, alpha, g))
            lines.append(f"| {g:+.2f} | {e['exponent']:.3f} | {e['se']:.3f} | {ps} |")
        lines.append("")

    lines += ["## Gates (thresholds are the DRAFT in configs/stage0.json until pre-registered)", "",
              "| Gate | Result | Detail |", "|---|---|---|"]
    for g in evaluate_gates(rows_a, rows_b, spec["gates_draft"]):
        res = "PASS" if g["pass"] else ("not evaluated" if g["pass"] is None else "FAIL")
        lines.append(f"| {g['gate']} | {res} | {g['detail']} |")

    figs = make_figures(rows_a, rows_b)
    lines += ["", "## Figures", ""] + [f"- {p.relative_to(ROOT).as_posix()}" for p in figs]
    report = RESULTS / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
