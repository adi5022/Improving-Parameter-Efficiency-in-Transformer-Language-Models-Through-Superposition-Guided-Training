"""Write notebooks/kaggle_lm.ipynb (plain language-model training on Kaggle). Run: python scripts/make_lm_notebook.py"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


CELLS = []

CELLS.append(md('''
# Language-model training on Kaggle (TinyStories, plain models, no penalty)

Trains the small Llama-style models from `lm/configs/` on Kaggle's GPUs. It prepares the data (our own 16,384-entry tokenizer on TinyStories), then runs the configs you list. **Nothing here uses the penalty.**

**One-time setup:** import this file (File > Import notebook); Session options: **Accelerator = GPU T4 x2**, **Internet = On**; fill in `COMMIT` below (a commit that contains `lm/`, `scripts/` and the configs).

**Use it in two steps.**
1. `MODE = "benchmark"`: prepares the data and times 20 training steps of each model size, then prints how many hours one full run would take. Nothing is trained for real. Look at the numbers before the next step (the hour estimates so far are unmeasured, and the local GTX 1660 Ti turned out much slower than assumed).
2. `MODE = "train"`: runs the configs in `CONFIGS`, one per GPU at a time. Run folders are `repo/results/lm/<run name>_s<seed>`. A Kaggle session lasts 12 hours at most, so keep single runs well under that; a run whose folder still exists continues from its last checkpoint.

Save Version > Save & Run All (Commit) runs it in the background; the Output tab then has the zips.
'''))

CELLS.append(code('''
# ---- settings: edit these ----
REPO_URL = "https://github.com/adi5022/Improving-Parameter-Efficiency-in-Transformer-Language-Models-Through-Superposition-Guided-Training.git"
COMMIT   = ""            # full commit hash (after you commit and push lm/, scripts/ and the configs)
MODE     = "benchmark"   # "benchmark" first, then "train"
CONFIGS  = ["tinystories_r1_10m_lr0.0006", "tinystories_r1_10m_lr0.0015", "tinystories_r1_10m_lr0.003"]   # the learning-rate sweep (3 runs, seed 0)
MAX_DOCS = 0             # 0 = the whole TinyStories train file; a small number (e.g. 100000) only to test the data step quickly
assert COMMIT, "Fill in COMMIT first"
assert MODE in ("benchmark", "train")
'''))

CELLS.append(code('''
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No GPU: Session options > Accelerator"
print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count())
'''))

CELLS.append(code('''
import os
url = REPO_URL
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GH_TOKEN")
    if token:
        url = REPO_URL.replace("https://", f"https://{token}@")
except Exception:
    pass                      # no token: fine for a public repository
os.chdir("/kaggle/working")
if not os.path.exists("repo"):
    subprocess.run(["git", "clone", url, "repo"], check=True)
os.chdir("/kaggle/working/repo")
subprocess.run(["git", "fetch", "--all", "-q"], check=True)
subprocess.run(["git", "checkout", "-q", COMMIT], check=True)
print(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout)
subprocess.run(["pip", "install", "-q", "tokenizers", "huggingface_hub"], check=True)
'''))

CELLS.append(code('''
# Data: download TinyStories (V2, GPT-4 written; 2.2 GB train + 22 MB held-out), train our tokenizer on the train file, write token shards.
from pathlib import Path
DATA = Path("/kaggle/working/repo/results/lm_data/tinystories_v0")
if (DATA / "meta.json").exists():
    print("data already prepared:", (DATA / "prep_summary.json").read_text())
else:
    from huggingface_hub import hf_hub_download
    files = {n: hf_hub_download("roneneldan/TinyStories", n, repo_type="dataset", local_dir="/kaggle/working/data")
             for n in ("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt")}
    cmd = ["python", "-u", "-m", "lm.prep_text", "--train", files["TinyStoriesV2-GPT4-train.txt"],
           "--valid", files["TinyStoriesV2-GPT4-valid.txt"], "--out", str(DATA), "--vocab-size", "16384"]
    if MAX_DOCS:
        cmd += ["--max-train-docs", str(MAX_DOCS)]
    subprocess.run(cmd, check=True)
'''))

CELLS.append(code('''
# MODE "benchmark": 20 real training steps per model size (written to a throw-away folder), then the projected hours of a full run.
import json, glob
def load_cfg(name):
    return json.load(open(f"/kaggle/working/repo/lm/configs/{name}.json"))
if MODE == "benchmark":
    sizes = ["tinystories_r1_10m", "tinystories_r1b_16m", "tinystories_r2_24m", "tinystories_r2b_36m"]
    print(f"{'config':24s} {'tokens/s':>9s} {'steps':>6s} {'hours for a full run':>22s}")
    for name in sizes:
        out = f"/kaggle/working/bench_{name}"
        r = subprocess.run(["python", "-u", "-m", "lm.train", "--config", f"lm/configs/{name}.json", "--smoke", "--device", "cuda",
                            "--out-root", out], capture_output=True, text=True)
        logs = glob.glob(f"{out}/smoke/*/log.jsonl")
        rows = [json.loads(l) for l in open(logs[0])] if logs else []
        tr = [x for x in rows if x["type"] == "train"]
        if len(tr) < 2:
            print(name, "FAILED:", (r.stderr or r.stdout)[-600:]); continue
        tps = (tr[-1]["tokens_seen"] - tr[0]["tokens_seen"]) / (tr[-1]["elapsed_s"] - tr[0]["elapsed_s"])
        steps = load_cfg(name)["steps"]
        print(f"{name:24s} {tps:9.0f} {steps:6d} {steps * 65536 / tps / 3600:22.1f}")
    print("(tokens/s is measured over steps 5-20 including a few evaluation pauses, so it is slightly pessimistic.)")
'''))

CELLS.append(code('''
# MODE "train": one job per GPU at a time. A run whose folder already exists is continued from its last checkpoint.
import threading, time, shutil
RUNS = Path("/kaggle/working/repo/results/lm")
def run_id(name):
    c = load_cfg(name); return f"{c['run_name']}_s{c['seed']}"
def status(name):
    m = RUNS / run_id(name) / "manifest.json"
    return json.load(open(m)).get("status") if m.exists() else None
if MODE == "train":
    queue = [n for n in CONFIGS if status(n) != "ok"]
    print("to run:", queue, "| already finished:", [n for n in CONFIGS if status(n) == "ok"])
    lock, ngpu = threading.Lock(), max(1, torch.cuda.device_count())
    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                name = queue.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = ["python", "-u", "-m", "lm.train", "--config", f"lm/configs/{name}.json", "--device", "cuda"]
            if (RUNS / run_id(name)).exists():
                cmd.append("--resume")
            with open(f"/kaggle/working/{name}.log", "w") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            print(f"[gpu {gpu}] {name} finished with exit code {rc}", flush=True)
    threads = [threading.Thread(target=worker, args=(g,)) for g in range(ngpu)]
    for t in threads: t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(120)
        for n in CONFIGS:
            lg = RUNS / run_id(n) / "log.jsonl"
            if lg.exists() and status(n) != "ok":
                rows = [json.loads(l) for l in open(lg) if l.strip()]
                last = rows[-1] if rows else {}
                print(n, last.get("step"), {k: last[k] for k in ("train_loss", "val_bpb") if k in last}, flush=True)
    for t in threads: t.join()
'''))

CELLS.append(code('''
# Results: a table and two zips for the Output tab (small = manifests and logs; full = also the checkpoints).
if MODE == "train":
    print(f"{'run':34s} {'status':12s} {'step':>6s} {'val bits/byte':>13s} {'hours':>6s}")
    for n in CONFIGS:
        d = RUNS / run_id(n)
        if not (d / "manifest.json").exists():
            print(f"{run_id(n):34s} not started"); continue
        m = json.load(open(d / "manifest.json"))
        vals = [json.loads(l) for l in open(d / "log.jsonl") if '"type": "val"' in l]
        tr = [json.loads(l) for l in open(d / "log.jsonl") if '"type": "train"' in l]
        print(f"{run_id(n):34s} {str(m.get('status')):12s} {str(m.get('final_step')):>6s} {vals[-1]['val_bpb'] if vals else float('nan'):13.4f} {(tr[-1]['elapsed_s'] / 3600 if tr else 0):6.2f}")
    os.makedirs("/kaggle/working/small", exist_ok=True)
    for n in CONFIGS:
        d = RUNS / run_id(n)
        if d.exists():
            os.makedirs(f"/kaggle/working/small/{d.name}", exist_ok=True)
            for f in ("manifest.json", "log.jsonl"):
                if (d / f).exists():
                    shutil.copy(d / f, f"/kaggle/working/small/{d.name}/{f}")
    tag = f"lm_{COMMIT[:7]}"
    shutil.make_archive(f"/kaggle/working/{tag}_small", "zip", "/kaggle/working/small")
    shutil.make_archive(f"/kaggle/working/{tag}_full", "zip", str(RUNS))
    print("wrote", f"/kaggle/working/{tag}_small.zip", "and", f"{tag}_full.zip", round(os.path.getsize(f"/kaggle/working/{tag}_full.zip") / 1e6), "MB")
'''))


def main():
    nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "notebooks" / "kaggle_lm.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    for i, c in enumerate(CELLS):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    print("notebooks/kaggle_lm.ipynb written; all code cells compile")


if __name__ == "__main__":
    main()
