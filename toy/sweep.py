"""Run a Stage 0 grid.

    python -m toy.sweep --grid G-A                  # full grid, resumes where it stopped
    python -m toy.sweep --grid G-A --dry-run        # list runs, train nothing
    python -m toy.sweep --grid G-A --smoke          # 2 runs x 300 steps into results/smoke/
    python -m toy.sweep --grid G-A --split 7,3 --part 0   # this machine's 70% of the grid (other: --part 1)
    python -m toy.sweep --grid G-D --split 1,1 --part 1 --subsplit 1,1 --subpart 0   # half of the second half

Runs with the same width are trained together in one batched loop on the GPU
(toy/batched.py); --no-batch runs them one at a time, --max-group caps the batch.
Each run is saved to results/stage0/<grid>/<run_id>.pt (full result incl. W) and one
line of scalars is appended to results/stage0/<grid>/summary.jsonl. Finished runs are
skipped, so an interrupted sweep can be restarted with the same command.

Decision S0-6: a real (non-smoke) sweep refuses to start if toy/ or configs/ have
uncommitted changes, and every result records the git commit it was produced with.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from .batched import group_key, train_group
from .train import RunConfig, train_run

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "stage0.json"
WIDTH_COST_OFFSET = 170   # a step costs about 10 ms plus 0.06 ms per unit of width, so cost is not proportional to width
SCALARS = ["test_loss", "final_train_loss", "phi_half", "phi_one", "mso_all", "mso_represented", "train_seconds"]
EXTRA_SCALARS = ["leak_last", "sae_fvu_ema", "sae_gate_step", "mso_dec", "twin_entries_total_group"]


def git_state() -> dict:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    commit = run("rev-parse", "--verify", "--quiet", "HEAD") or None
    dirty = run("status", "--porcelain", "--", "toy", "configs")
    return {"commit": commit, "dirty_paths": dirty.splitlines() if dirty else []}


def grid_configs(grid_name: str, config_path: Path = CONFIG) -> list[RunConfig]:
    spec = json.loads(config_path.read_text())
    defaults, grid = spec["defaults"], spec["grids"][grid_name]
    # a grid is one rectangle of settings, or several rectangles under "parts" (for runs that are not a full product)
    rects = grid["parts"] if "parts" in grid else [grid]
    out, seen = [], set()
    for rect in rects:
        # "extra": other RunConfig fields fixed for this rectangle (Stage 2: the SAE type and its settings)
        # "gamma_by_width": {"15": -0.55, ...} instead of "gamma", when each width has its own weight decay
        # "lam_by_width":   {"15": [..], ...}  instead of "lam",   when each width has its own penalty range
        extra = rect.get("extra", {})
        gbw, lbw = rect.get("gamma_by_width"), rect.get("lam_by_width")
        for a, m, s, pen in itertools.product(rect["alpha"], rect["width"], rect["seed"], rect.get("penalty", ["none"])):
            gammas = [gbw[str(m)]] if gbw is not None else rect["gamma"]
            lams = lbw[str(m)] if lbw is not None else rect.get("lam", [0.0])
            for g, lam in itertools.product(gammas, lams):
                c = RunConfig(alpha=a, width=m, gamma=g, seed=s, penalty=pen, lam=lam, **{**defaults, **extra})
                if c.run_id() not in seen:
                    seen.add(c.run_id())
                    out.append(c)
    return out


def _save(result: dict, out_dir: Path, git: dict) -> None:
    result["git"] = git
    torch.save(result, out_dir / f"{result['run_id']}.pt")
    extras = {k: result[k] for k in EXTRA_SCALARS if k in result}          # Stage 2 runs only
    line = {"run_id": result["run_id"], "status": result["status"], **result["config"],
            **{k: result[k] for k in SCALARS}, **extras, "git_commit": git["commit"], "precision": result.get("precision", "fp32")}
    with (out_dir / "summary.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")


def make_groups(todo: list[RunConfig], max_group: int) -> list[list[RunConfig]]:
    """Runs that can share one batched training loop (same width and settings), in chunks."""
    by_key: dict = {}
    for c in todo:
        by_key.setdefault(group_key(c), []).append(c)
    return [runs[i:i + max_group] for runs in by_key.values() for i in range(0, len(runs), max_group)]


def split_configs(configs: list[RunConfig], weights: list[float], part: int, max_group: int = 24) -> list[RunConfig]:
    """The runs one machine should do when a grid is shared between machines.

    Whole batched groups are dealt out, biggest first, to the part furthest below its share
    (share = weight / sum of weights; a group's cost is runs x (170 + width) x steps). The result
    depends only on the grid, the weights and max_group, so every machine computes the same
    split, and every run lands on exactly one part.
    """
    if not 0 <= part < len(weights) or any(w <= 0 for w in weights):
        raise ValueError("part must index into weights, and weights must be positive")
    groups = make_groups(configs, max_group)
    # time per step is roughly 10 ms + 0.06 ms per width unit (measured on a T4); a group with an SAE costs about 4x (measured locally)
    cost = lambda g: len(g) * (WIDTH_COST_OFFSET + g[0].width) * g[0].steps * (4 if g[0].penalty == "leak_sae" else 1)
    load = [0.0] * len(weights)
    mine: list[list[RunConfig]] = []
    for g in sorted(groups, key=lambda g: (-cost(g), g[0].run_id())):
        k = min(range(len(weights)), key=lambda i: (load[i] / weights[i], i))
        load[k] += cost(g)
        if k == part:
            mine.append(g)
    keep = {c.run_id() for g in mine for c in g}
    return [c for c in configs if c.run_id() in keep]


def run_grid(configs: list[RunConfig], out_dir: Path, device: str, git: dict, *,
             batched: bool = True, max_group: int = 24, **impl) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in configs if not (out_dir / f"{c.run_id()}.pt").exists()]
    print(f"{len(configs)} runs in grid, {len(configs) - len(todo)} already done, {len(todo)} to run -> {out_dir}")
    groups = make_groups(todo, max_group) if batched else [[c] for c in todo]
    t_start, done = time.perf_counter(), 0
    for gi, runs in enumerate(groups, 1):
        results = train_group(runs, device=device, **impl) if batched else [train_run(runs[0], device=device, **impl)]
        for res in results:
            _save(res, out_dir, git)
        done += len(runs)
        eta_min = (time.perf_counter() - t_start) / done * (len(todo) - done) / 60
        c0 = runs[0]
        bad = [r["run_id"] for r in results if r["status"] != "ok"]
        print(f"[group {gi}/{len(groups)}] m={c0.width}, {len(runs)} runs, "
              f"{results[0]['train_seconds'] * len(runs):.0f}s | {done}/{len(todo)} done, ETA {eta_min:.0f} min"
              + (f" | NON-OK: {bad}" if bad else ""), flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True, choices=sorted(json.loads(CONFIG.read_text())["grids"]))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2 runs x 300 steps, written to results/smoke/")
    p.add_argument("--allow-dirty", action="store_true", help="run with uncommitted code (results are marked)")
    p.add_argument("--no-batch", action="store_true", help="one run at a time (slow; for cross-checks)")
    p.add_argument("--max-group", type=int, default=24, help="max runs trained together on the GPU")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                   help="fp16 = the three large matrix products in half precision (needs a GPU with tensor cores to pay off); results go to a separate folder <grid>-fp16 so they never mix with fp32 results")
    p.add_argument("--split", default=None,
                   help="share the grid between machines, e.g. 7,3 (relative speeds); use with --part")
    p.add_argument("--part", type=int, default=0, help="which share of --split this machine runs (0-based)")
    p.add_argument("--subsplit", default=None,
                   help="split this machine's share again between several machines, e.g. 1,1 (use with --subpart); "
                        "lets one half of a 1,1 split be shared without changing what the other half runs")
    p.add_argument("--subpart", type=int, default=0, help="which share of --subsplit to run (0-based)")
    args = p.parse_args(argv)

    configs = grid_configs(args.grid)
    if args.split:
        weights = [float(w) for w in args.split.split(",")]
        configs = split_configs(configs, weights, args.part, args.max_group)
        print(f"part {args.part} of split {args.split}: {len(configs)} of the grid's runs")
        if args.subsplit:
            configs = split_configs(configs, [float(w) for w in args.subsplit.split(",")], args.subpart, args.max_group)
            print(f"  sub-part {args.subpart} of sub-split {args.subsplit}: {len(configs)} runs")
    git = git_state()
    if args.dry_run:
        for c in configs:
            print(c.run_id())
        print(f"{len(configs)} runs; git commit {git['commit']}; uncommitted: {git['dirty_paths'] or 'none'}")
        return 0

    if args.smoke:
        configs = [replace(c, steps=300, warmup_steps=50, test_batches=2, log_every=50) for c in configs[:2]]
        out_dir = ROOT / "results" / "smoke" / args.grid
    else:
        out_dir = ROOT / "results" / "stage0" / (args.grid + ("-fp16" if args.precision == "fp16" else ""))
        if (git["commit"] is None or git["dirty_paths"]) and not args.allow_dirty:
            print("Refusing to run: commit toy/ and configs/ first (decision S0-6). "
                  f"Uncommitted: {git['dirty_paths'] or 'no commits yet'}", file=sys.stderr)
            return 1
    if args.precision == "fp16" and args.no_batch:
        sys.exit("--precision fp16 needs the batched trainer (drop --no-batch)")
    impl = {"half": True} if args.precision == "fp16" else {}
    run_grid(configs, out_dir, args.device, git, batched=not args.no_batch, max_group=args.max_group, **impl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
