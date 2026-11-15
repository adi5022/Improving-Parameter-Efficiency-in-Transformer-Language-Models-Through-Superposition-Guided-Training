"""Validation loss, perplexity and bits-per-byte on a fixed set of windows.

The loss is the sum of token negative log-likelihoods divided by the token count, so it does
not depend on how the windows are batched. Perplexity = exp(loss). Bits-per-byte =
loss / ln 2 / bytes_per_token, where bytes_per_token is (UTF-8 bytes of the split's text) /
(tokens in the split) from meta.json; it is only reported when that number is available, and is
exact for the whole split but an approximation for a subset of windows (exact for byte-level
tokenizers, where it is 1).

    python -m lm.eval --run-dir results/lm/dryrun_s0 [--ckpt path] [--windows 64]
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from .model import lm_loss


@torch.no_grad()
def evaluate(model, x, y, batch_size: int, device: str, autocast=None, bytes_per_token: float | None = None) -> dict:
    """x, y [N, T] token windows (y = x shifted). `autocast` is an optional context manager factory
    (a no-argument callable) used around the forward pass. Restores the model's train/eval mode."""
    was_training = model.training
    model.eval()
    total_nll, n_tokens = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        with (autocast() if autocast else nullcontext()):
            logits = model(xb)
        total_nll += lm_loss(logits, yb, reduction="sum").item()
        n_tokens += yb.numel()
    model.train(was_training)
    loss = total_nll / n_tokens
    out = {"loss": loss, "perplexity": math.exp(loss), "bits_per_token": loss / math.log(2),
           "n_tokens": n_tokens, "n_windows": len(x)}
    if bytes_per_token:
        out["bits_per_byte"] = loss / math.log(2) / bytes_per_token
    return out


def main(argv=None) -> int:
    from .data import TokenShards
    from .model import GPT
    from .train import ROOT, TrainConfig, latest_checkpoint

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default=None, help="checkpoint file (default: the latest in the run directory)")
    p.add_argument("--windows", type=int, default=None, help="default: eval_windows from the run's config")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg = TrainConfig.from_dict(json.loads((run_dir / "manifest.json").read_text())["config"])
    ckpt_path = Path(args.ckpt) if args.ckpt else latest_checkpoint(run_dir)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model = GPT(cfg.model).to(args.device)
    model.load_state_dict(ckpt["model"])
    val = TokenShards(Path(cfg.data_dir) if Path(cfg.data_dir).is_absolute() else ROOT / cfg.data_dir, "val")
    x, y = val.fixed_windows(cfg.model.context_len, args.windows or cfg.eval_windows)
    res = evaluate(model, x, y, cfg.eval_batch_size, args.device, bytes_per_token=val.bytes_per_token)
    print(json.dumps({"checkpoint": str(ckpt_path), "step": ckpt["step"], **res}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
