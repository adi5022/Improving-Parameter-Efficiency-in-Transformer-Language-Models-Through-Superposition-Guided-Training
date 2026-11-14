"""Merge sweep results produced on another machine (for example Colab) into this project.

    python scripts/merge_results.py <folder with the other machine's G-A or G-B results> results/stage0/G-A

Copies every run file (<run_id>.pt) that is not already present and appends the matching
summary.jsonl lines. A run already present is never overwritten, so merging twice is safe.
"""

import json
import shutil
import sys
from pathlib import Path


def merge(src: Path, dst: Path) -> tuple[int, int]:
    dst.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in dst.glob("*.pt")}
    copied = 0
    for pt in sorted(src.glob("*.pt")):
        if pt.stem not in have:
            shutil.copy2(pt, dst / pt.name)
            have.add(pt.stem)
            copied += 1
    seen = set()
    out = dst / "summary.jsonl"
    if out.exists():
        seen = {json.loads(l)["run_id"] for l in out.read_text().splitlines() if l.strip()}
    added = 0
    src_summary = src / "summary.jsonl"
    if src_summary.exists():
        with out.open("a") as f:
            for line in src_summary.read_text().splitlines():
                if line.strip() and json.loads(line)["run_id"] not in seen:
                    f.write(line + "\n")
                    seen.add(json.loads(line)["run_id"])
                    added += 1
    return copied, added


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    c, a = merge(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"copied {c} run files, added {a} summary lines")
