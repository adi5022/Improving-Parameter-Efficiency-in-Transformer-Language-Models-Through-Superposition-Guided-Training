"""Write the language-model run configs (lm/configs/tinystories_*.json) from the approved settings.

    python scripts/make_lm_configs.py

Approved (docs/research/architectural_decisions.md, 2026-11-15): Llama-style block (D-LM-2), tied embeddings (D-LM-3),
16,384-entry vocabulary (D-LM-4), context 1,024 (D-LM-6), about 20 tokens per parameter (D-LM-7), AdamW (0.9, 0.95),
clip 1.0, warmup max(100, 2%), cosine to 10% (D-LM-8), about 65,536 tokens per step (D-LM-9), fp16 (D-LM-10).
NOT YET APPROVED: the four model shapes below (the 16M and 36M shapes are new proposals), the provisional peak
learning rate of the ladder configs (to be replaced by the result of the 10.6M sweep), dropout 0, init std 0.02.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lm.model import ModelConfig, param_counts   # noqa: E402

VOCAB, CONTEXT, TOKENS_PER_STEP, TOKENS_PER_PARAM = 16384, 1024, 65536, 20
DATA_DIR = "results/lm_data/tinystories_v0"
# name: (d_model, layers, heads, d_ff, micro-batch, grad_accum)    micro-batch x accum x 1024 = 65,536 tokens
SHAPES = {
    "r1_10m": (256, 8, 4, 704, 32, 2),
    "r1b_16m": (320, 9, 5, 832, 32, 2),
    "r2_24m": (384, 10, 6, 1024, 16, 4),
    "r2b_36m": (448, 12, 7, 1216, 16, 4),
}
SWEEP_LRS = [6e-4, 1.5e-3, 3e-3]


def config(name: str, shape, peak_lr: float, seed: int = 0, note: str = "") -> dict:
    d, layers, heads, ff, mb, accum = shape
    assert mb * accum * CONTEXT == TOKENS_PER_STEP
    model = dict(n_layer=layers, d_model=d, n_head=heads, d_ff=ff, vocab_size=VOCAB, context_len=CONTEXT,
                 pos_encoding="rope", rope_base=10000.0, norm="rmsnorm", mlp="swiglu", tie_embeddings=True,
                 bias=False, dropout=0.0)
    total = param_counts(ModelConfig(**model))["total"]
    steps = round(TOKENS_PER_PARAM * total / TOKENS_PER_STEP)
    return {
        "_note": note, "run_name": name, "seed": seed, "data_dir": DATA_DIR, "model": model,
        "steps": steps, "batch_size": mb, "grad_accum": accum, "peak_lr": peak_lr,
        "warmup_steps": max(100, round(0.02 * steps)), "final_lr_frac": 0.1, "weight_decay": 0.1,
        "grad_clip": 1.0, "precision": "fp16", "compile": False,
        "eval_every": max(50, steps // 12), "eval_windows": 512, "eval_batch_size": 16,
        "log_every": 20, "ckpt_every": 250, "keep_last_ckpts": 2,
    }, total, steps


def main() -> None:
    out = ROOT / "lm" / "configs"
    rows = []
    for key, shape in SHAPES.items():
        cfg, total, steps = config(f"ts_{key}", shape, 1.5e-3,
                                   note="PROVISIONAL peak_lr until the 10.6M learning-rate sweep is done; model shape not yet approved")
        (out / f"tinystories_{key}.json").write_text(json.dumps(cfg, indent=2) + "\n")
        rows.append((key, total, steps, cfg["warmup_steps"], total * TOKENS_PER_PARAM / 1e6))
    for lr in SWEEP_LRS:
        cfg, total, steps = config(f"ts_r1_10m_lr{lr:g}", SHAPES["r1_10m"], lr,
                                   note="learning-rate sweep at 10.6M (D-LM-8, modified): 1 seed, 3 values; model shape not yet approved")
        (out / f"tinystories_r1_10m_lr{lr:g}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"{'rung':10s} {'params':>11s} {'steps':>6s} {'warmup':>6s} {'tokens (M)':>10s}")
    for r in rows:
        print(f"{r[0]:10s} {r[1]:11,d} {r[2]:6d} {r[3]:6d} {r[4]:10.0f}")


if __name__ == "__main__":
    main()
