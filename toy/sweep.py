"""Run a Stage 0 grid.

    python -m toy.sweep --grid G-A                  # full grid, resumes where it stopped
    python -m toy.sweep --grid G-A --dry-run        # list runs, train nothing
    python -m toy.sweep --grid G-A --smoke          # 2 runs x 300 steps into results/smoke/

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

from .train import RunConfig, train_run

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "stage0.json"
SCALARS = ["test_loss", "final_train_loss", "phi_half", "phi_one", "mso_all", "mso_represented", "train_seconds"]


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
    return [RunConfig(alpha=a, width=m, gamma=g, seed=s, **defaults)
            for a, m, g, s in itertools.product(grid["alpha"], grid["width"], grid["gamma"], grid["seed"])]


def run_grid(configs: list[RunConfig], out_dir: Path, device: str, git: dict, **impl) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.jsonl"
    todo = [c for c in configs if not (out_dir / f"{c.run_id()}.pt").exists()]
    print(f"{len(configs)} runs in grid, {len(configs) - len(todo)} already done, {len(todo)} to run -> {out_dir}")
    t_start = time.perf_counter()
    for i, cfg in enumerate(todo, 1):
        result = train_run(cfg, device=device, **impl)
        result["git"] = git
        torch.save(result, out_dir / f"{cfg.run_id()}.pt")
        line = {"run_id": result["run_id"], "status": result["status"], **result["config"],
                **{k: result[k] for k in SCALARS}, "git_commit": git["commit"]}
        with summary.open("a") as f:
            f.write(json.dumps(line) + "\n")
        elapsed = time.perf_counter() - t_start
        eta_min = elapsed / i * (len(todo) - i) / 60
        print(f"[{i}/{len(todo)}] {cfg.run_id()}  test={result['test_loss']:.3e}  "
              f"phi1/2={result['phi_half']:.3f}  {result['train_seconds']:.0f}s  ETA {eta_min:.0f} min"
              + ("" if result["status"] == "ok" else f"  STATUS: {result['status']}"), flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True, choices=["G-A", "G-B"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2 runs x 300 steps, written to results/smoke/")
    p.add_argument("--allow-dirty", action="store_true", help="run with uncommitted code (results are marked)")
    args = p.parse_args(argv)

    configs = grid_configs(args.grid)
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
        out_dir = ROOT / "results" / "stage0" / args.grid
        if (git["commit"] is None or git["dirty_paths"]) and not args.allow_dirty:
            print("Refusing to run: commit toy/ and configs/ first (decision S0-6). "
                  f"Uncommitted: {git['dirty_paths'] or 'no commits yet'}", file=sys.stderr)
            return 1
    run_grid(configs, out_dir, args.device, git)
    return 0


if __name__ == "__main__":
    sys.exit(main())
