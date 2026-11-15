"""Train a decoder-only transformer language model from a JSON config.

    python -m lm.train --config lm/configs/dryrun.json --device cpu --allow-dirty
    python -m lm.train --config lm/configs/dryrun.json --device cpu --smoke     # 20 steps -> results/lm/smoke/
    python -m lm.train --config lm/configs/dryrun.json --resume                 # continue from the latest checkpoint

Everything is configuration; all default values are PLACEHOLDER, NOT APPROVED (CLAUDE.md
section 9). A run writes results/lm/<run_id>/ with manifest.json, log.jsonl and checkpoints.

Reproducibility (same conventions as toy/sweep.py): a real (non-smoke) run refuses to start if
lm/ or the config file has uncommitted changes (unless --allow-dirty, which is recorded in the
manifest), refuses to overwrite an existing run directory, and records the git commit.

Seeding: `seed` seeds the weight init and dropout (torch's global RNG) and, separately, the
stream of random training windows (a torch.Generator owned by the data sampler). A checkpoint
stores model, optimiser, GradScaler (fp16 only), the step counter (the LR schedule is a pure
function of it), the global RNG states and the sampler state, so a resumed run reproduces an
uninterrupted one exactly (tested on CPU; CUDA kernels are not guaranteed bitwise deterministic).

Auxiliary-loss hook: pass `aux_loss_fn(model, (x, y)) -> (scalar_tensor, dict_of_floats)` to
`train()`. The scalar is added to the language-model loss for every micro-batch (it should
already include any weight) and its dict is logged as `aux/<key>`. It is called under the same
autocast context as the forward pass. The default is None; no such loss is implemented here.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Callable

import torch

from .data import BatchSampler, TokenShards
from .eval import evaluate
from .model import GPT, ModelConfig, check_param_count, lm_loss

ROOT = Path(__file__).resolve().parent.parent
PRECISIONS = ("fp32", "bf16", "fp16")
EMBEDDING_PARAMS = ("tok_emb.weight", "pos_emb.weight", "lm_head.weight")


@dataclass(frozen=True)
class TrainConfig:
    # PLACEHOLDER, NOT APPROVED: all values in this class, and in the nested ModelConfig.
    run_name: str = "dryrun"
    seed: int = 0
    data_dir: str = "results/lm_data/synthetic_v0"   # shard directory; relative paths are from the repo root
    model: ModelConfig = field(default_factory=ModelConfig)
    steps: int = 300                    # optimiser steps
    batch_size: int = 16                # sequences per micro-batch
    grad_accum: int = 1                 # micro-batches per optimiser step
    peak_lr: float = 3e-3
    warmup_steps: int = 20              # linear warm-up, then cosine to final_lr_frac * peak_lr
    final_lr_frac: float = 0.1
    weight_decay: float = 0.1           # decoupled (AdamW), applied only to 2-D weights, see decay_param_names
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0              # global norm; 0 disables clipping
    decay_embeddings: bool = False      # if true the embedding tables / lm_head also get weight decay
    precision: str = "fp32"             # "fp32" | "bf16" (autocast) | "fp16" (autocast + GradScaler, CUDA only)
    compile: bool = False               # torch.compile the model
    eval_every: int = 50                # validation every N steps (and at step 0 and the last step)
    eval_windows: int = 64              # fixed validation windows of context_len tokens
    eval_batch_size: int = 16
    log_every: int = 10
    ckpt_every: int = 100
    keep_last_ckpts: int = 2            # older periodic checkpoints are deleted; 0 keeps all

    def __post_init__(self):
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if min(self.steps, self.batch_size, self.grad_accum, self.log_every, self.eval_every, self.ckpt_every) < 1:
            raise ValueError("steps, batch_size, grad_accum, log_every, eval_every, ckpt_every must be >= 1")

    @property
    def run_id(self) -> str:
        return f"{self.run_name}_s{self.seed}"

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = {k: v for k, v in d.items() if not k.startswith("_")}   # "_note" keys are comments
        model = ModelConfig(**{k: v for k, v in d.pop("model", {}).items() if not k.startswith("_")})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(model=model, **d)

    @classmethod
    def from_json(cls, path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- small helpers

def lr_at(step: int, peak_lr: float, total_steps: int, warmup_steps: int, final_frac: float) -> float:
    """LR for the 0-based optimiser step `step`: linear warm-up to peak_lr over warmup_steps, then
    cosine down to final_frac * peak_lr. A pure function of `step`, so resuming needs no scheduler state."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    min_lr = final_frac * peak_lr
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def decay_param_names(model: GPT, decay_embeddings: bool = False) -> set[str]:
    """Names of the parameters that get weight decay: 2-D weight matrices inside the blocks
    (attention and MLP). Not decayed: norm weights and all biases (1-D), and, unless
    decay_embeddings, the token/position embeddings and an untied lm_head."""
    names = set()
    for name, p in model.named_parameters():
        if p.ndim == 2 and (decay_embeddings or name not in EMBEDDING_PARAMS):
            names.add(name)
    return names


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    decay_names = decay_param_names(model, cfg.decay_embeddings)
    named = dict(model.named_parameters())
    decay = [p for n, p in named.items() if n in decay_names]
    no_decay = [p for n, p in named.items() if n not in decay_names]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def git_state(paths=("lm",)) -> dict:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    commit = run("rev-parse", "--verify", "--quiet", "HEAD") or None
    dirty = run("status", "--porcelain", "--", *paths)
    return {"commit": commit, "dirty_paths": dirty.splitlines() if dirty else []}


def checkpoint_files(run_dir: Path) -> list[Path]:
    return sorted(Path(run_dir).glob("ckpt_*.pt"))


def latest_checkpoint(run_dir) -> Path:
    files = checkpoint_files(Path(run_dir))
    if not files:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    return files[-1]


def _autocast(precision: str, device: str):
    if precision == "fp32":
        return None
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    dev = device.split(":")[0]
    if precision == "fp16" and dev != "cuda":
        raise ValueError("fp16 autocast needs a CUDA device (use bf16 or fp32 on CPU)")
    if precision == "bf16" and dev == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bf16; use fp16 or fp32")
    return lambda: torch.autocast(device_type=dev, dtype=dtype)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _save_checkpoint(run_dir: Path, step: int, model, optimizer, scaler, sampler, cfg: TrainConfig, device: str) -> Path:
    state = {
        "step": step,
        "config": asdict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
        "data": sampler.state_dict(),
    }
    path = run_dir / f"ckpt_{step:07d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    if cfg.keep_last_ckpts > 0:
        for old in checkpoint_files(run_dir)[:-cfg.keep_last_ckpts]:
            old.unlink()
    return path


def _load_log_prefix(log_path: Path, step: int) -> list[dict]:
    """Records of an earlier run up to and including `step` (later ones are re-done after a resume)."""
    if not log_path.exists():
        return []
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    return [r for r in records if r["step"] <= step]


# ---------------------------------------------------------------- training

def train(cfg: TrainConfig, run_dir, device: str = "cpu", *, git: dict | None = None, smoke: bool = False,
          resume: bool = False, stop_at: int | None = None, aux_loss_fn: Callable | None = None,
          command: list[str] | None = None) -> dict:
    """Run (or resume) one training run in `run_dir`. `stop_at` ends the call after that many total
    steps WITHOUT writing a checkpoint, to simulate a crash (tests and time-limited sessions)."""
    run_dir = Path(run_dir)
    git = git if git is not None else git_state()
    manifest_path, log_path = run_dir / "manifest.json", run_dir / "log.jsonl"
    autocast = _autocast(cfg.precision, device)

    # ---- fresh start or resume
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"cannot resume: no manifest in {run_dir}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != asdict(cfg):
            raise ValueError("cannot resume: the config differs from the one in the manifest")
        if manifest["git_commit"] != git["commit"] and not smoke:
            raise ValueError(f"cannot resume: code is at {git['commit']}, run was started at {manifest['git_commit']}")
        ckpt = torch.load(latest_checkpoint(run_dir), map_location=device, weights_only=True)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"{run_dir} already exists; use --resume, or change run_name or seed")
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = None

    # ---- model, optimiser, data
    torch.manual_seed(cfg.seed)
    model = GPT(cfg.model).to(device)
    counts = check_param_count(model)      # raises if the formula and the real count disagree
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
    data_dir = resolve(cfg.data_dir)
    train_shards, val_shards = TokenShards(data_dir, "train"), TokenShards(data_dir, "val")
    sampler = BatchSampler(train_shards, cfg.model.context_len, cfg.seed)
    val_x, val_y = val_shards.fixed_windows(cfg.model.context_len, cfg.eval_windows)

    step = 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["rng_cpu"].cpu())
        if ckpt["rng_cuda"] is not None and device.startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_cuda"]])
        sampler.load_state_dict(ckpt["data"])
        step = ckpt["step"]
        log = _load_log_prefix(log_path, step)
        log_path.write_text("".join(json.dumps(r) + "\n" for r in log))
        manifest.setdefault("resumes", []).append({"from_step": step, "git_commit": git["commit"],
                                                   "git_dirty_paths": git["dirty_paths"]})
    else:
        manifest = {
            "run_id": cfg.run_id, "config": asdict(cfg), "seed": cfg.seed, "smoke": smoke,
            "git_commit": git["commit"], "git_dirty": bool(git["dirty_paths"]), "git_dirty_paths": git["dirty_paths"],
            "torch_version": torch.__version__, "python_version": sys.version.split()[0],
            "device": device, "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
            "precision": cfg.precision, "compile": cfg.compile,
            "param_counts": counts,
            "data": {"dir": cfg.data_dir, "tokenizer": train_shards.meta["tokenizer"], "dtype": train_shards.meta["dtype"],
                     "train_tokens": train_shards.n_tokens, "val_tokens": val_shards.n_tokens,
                     "val_bytes_per_token": val_shards.bytes_per_token, "meta": train_shards.meta["shards"]},
            "command": command, "status": "running", "resumes": [],
        }
    manifest["started_utc"] = manifest.get("started_utc") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(manifest_path, manifest)

    fwd = torch.compile(model) if cfg.compile else model
    params = list(model.parameters())
    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.model.context_len
    print(f"[{cfg.run_id}] params total {counts['total']:,} (non-embedding {counts['non_embedding']:,}), "
          f"device {device}, precision {cfg.precision}, step {step}/{cfg.steps}", flush=True)

    log_file = log_path.open("a")

    def write(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        if record["type"] == "val":
            print(f"  step {record['step']:>6} val_loss {record['val_loss']:.4f} ppl {record['val_ppl']:.2f}", flush=True)

    def validate(at_step: int) -> dict:
        res = evaluate(model, val_x, val_y, cfg.eval_batch_size, device, autocast, val_shards.bytes_per_token)
        rec = {"type": "val", "step": at_step, "val_loss": res["loss"], "val_ppl": res["perplexity"],
               "val_bpb": res.get("bits_per_byte")}
        write(rec)
        return rec

    status, last_val, t0 = "ok", None, time.perf_counter()
    last_step = cfg.steps if stop_at is None else min(cfg.steps, stop_at)
    if step == 0:
        last_val = validate(0)
    model.train()
    while step < last_step:
        lr = lr_at(step, cfg.peak_lr, cfg.steps, cfg.warmup_steps, cfg.final_lr_frac)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        lm_sum, aux_sum, aux_log = torch.zeros(()), torch.zeros(()), {}
        for _ in range(cfg.grad_accum):
            x, y = sampler.next_batch(cfg.batch_size)
            x, y = x.to(device), y.to(device)
            with (autocast() if autocast else nullcontext()):
                loss = lm_loss(fwd(x), y)
                total = loss
                if aux_loss_fn is not None:
                    aux, info = aux_loss_fn(model, (x, y))
                    total = loss + aux
                    aux_sum = aux_sum + aux.detach().float().cpu() / cfg.grad_accum
                    for k, v in info.items():
                        aux_log[k] = aux_log.get(k, 0.0) + float(v) / cfg.grad_accum
            scaler.scale(total / cfg.grad_accum).backward()
            lm_sum = lm_sum + loss.detach().float().cpu() / cfg.grad_accum
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
        scaler.step(optimizer)
        scaler.update()
        step += 1

        train_loss = lm_sum.item()
        if not math.isfinite(train_loss):
            status = f"non-finite loss at step {step}"
            print(status, file=sys.stderr)
            break
        if step % cfg.log_every == 0 or step == cfg.steps:
            rec = {"type": "train", "step": step, "lr": lr, "train_loss": train_loss, "grad_norm": float(gnorm),
                   "tokens_seen": step * tokens_per_step, "elapsed_s": time.perf_counter() - t0}
            if aux_loss_fn is not None:
                rec.update({"aux_loss": aux_sum.item(), "total_loss": train_loss + aux_sum.item(),
                            **{f"aux/{k}": v for k, v in aux_log.items()}})
            write(rec)
            if step % (cfg.log_every * 5) == 0 or step == cfg.steps:
                print(f"  step {step:>6} loss {train_loss:.4f} lr {lr:.2e} gnorm {float(gnorm):.2f}", flush=True)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            last_val = validate(step)
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            if stop_at is None or step < stop_at:      # a simulated crash writes nothing at the stop step
                _save_checkpoint(run_dir, step, model, optimizer, scaler, sampler, cfg, device)
    log_file.close()

    manifest.update({"status": status if step >= cfg.steps or status != "ok" else "interrupted",
                     "final_step": step, "final_val": last_val,
                     "train_seconds_last_call": time.perf_counter() - t0})
    _write_json(manifest_path, manifest)
    return {"run_dir": run_dir, "status": manifest["status"], "final_step": step, "final_val": last_val,
            "param_counts": counts}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="20 steps into <out-root>/smoke/; skips the clean-tree check")
    p.add_argument("--allow-dirty", action="store_true", help="run with uncommitted code (recorded in the manifest)")
    p.add_argument("--resume", action="store_true", help="continue the run from its latest checkpoint")
    p.add_argument("--out-root", default=str(ROOT / "results" / "lm"))
    args = p.parse_args(argv)

    config_path = Path(args.config).resolve()
    cfg = TrainConfig.from_json(config_path)
    watched = ["lm"]
    if ROOT in config_path.parents:
        watched.append(config_path.relative_to(ROOT).as_posix())
    git = git_state(watched)
    out_root = Path(args.out_root)
    if args.smoke:
        cfg = replace(cfg, steps=min(cfg.steps, 20), warmup_steps=min(cfg.warmup_steps, 5), eval_every=10,
                      ckpt_every=10, eval_windows=min(cfg.eval_windows, 8), log_every=5)
        out_root = out_root / "smoke"
    elif (git["commit"] is None or git["dirty_paths"]) and not args.allow_dirty:
        print("Refusing to run: commit lm/ and the config file first (or pass --allow-dirty to mark the run). "
              f"Uncommitted: {git['dirty_paths'] or 'no commits yet'}", file=sys.stderr)
        return 1
    result = train(cfg, out_root / cfg.run_id, args.device, git=git, smoke=args.smoke, resume=args.resume,
                   command=[sys.executable, "-m", "lm.train", *(argv if argv is not None else sys.argv[1:])])
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
