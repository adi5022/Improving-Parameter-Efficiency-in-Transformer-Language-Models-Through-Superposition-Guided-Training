# lm/ - decoder-only transformer training pipeline

Machinery only. **No architecture or training choice here is approved** (CLAUDE.md section 9, decision
register D-08 and following). Every default value in the code and in `lm/configs/dryrun.json` is
**PLACEHOLDER, NOT APPROVED**. Nothing in this package depends on a value being final: it is all
configuration. Nothing here is a research result.

## Files

| File | What it does |
|---|---|
| `model.py` | GPT-style model (`GPT`, `ModelConfig`), exact parameter counts (`param_counts`, `check_param_count`) |
| `data.py` | shard writer and reader (memmap), seeded random-window sampler, byte tokenizer, synthetic corpus |
| `eval.py` | validation loss / perplexity / bits-per-byte on a fixed set of windows; CLI to evaluate a checkpoint |
| `train.py` | CLI trainer, `TrainConfig`, manifest, checkpoints, exact resume, auxiliary-loss hook |
| `configs/dryrun.json` | tiny CPU dry run (about 357k parameters, 300 steps) |
| `../tests/lm/` | tests (see below) |

## How to run (CPU dry run, nothing is downloaded)

From the repo root, with `.venv/Scripts/python.exe` (use `set CUDA_VISIBLE_DEVICES=` if a GPU job is running):

```
python -m lm.data  --out results/lm_data/synthetic_v0 --docs 4000 --seed 0     # procedural text, byte-level, 5 train + 1 val shard
python -m lm.train --config lm/configs/dryrun.json --device cpu --allow-dirty  # about 1.5 minutes on a laptop CPU
python -m lm.train --config lm/configs/dryrun.json --device cpu --resume        # continue an interrupted run
python -m lm.train --config lm/configs/dryrun.json --device cpu --smoke         # 20 steps into results/lm/smoke/
python -m lm.eval  --run-dir results/lm/dryrun_s0 --device cpu                  # re-evaluate the latest checkpoint
```

Output goes to `results/lm/<run_name>_s<seed>/` (git-ignored): `manifest.json`, `log.jsonl`, `ckpt_<step>.pt`.
`log.jsonl` has one JSON object per line, `"type": "train"` (step, lr, train_loss, grad_norm, tokens_seen,
elapsed_s) or `"type": "val"` (step, val_loss, val_ppl, val_bpb). `train_loss` is the language-model loss of
that optimiser step (mean over its micro-batches); validation also runs at step 0 and the last step.

### Reproducibility rules (same as `toy/sweep.py`)

- A real run refuses to start if `lm/` or the config file has uncommitted changes (including untracked files),
  or if there is no commit. `--allow-dirty` overrides it and the manifest records `git_dirty: true` and the
  dirty paths. `--smoke` skips the check and writes to `results/lm/smoke/`.
- A run directory is never overwritten: starting a run whose directory exists is an error; use `--resume` or
  change `run_name` / `seed`. `--resume` refuses if the config differs from the manifest's or the git commit
  differs (smoke runs excepted).
- `manifest.json`: run id, full config, seed, git commit, dirty flag and paths, torch and python versions,
  device, precision, parameter counts by component, data description (tokenizer, shard list, token counts,
  bytes per token), command line, status, final step and final validation result, and one entry per resume.
- Exact resume: a checkpoint holds model, optimiser, GradScaler (fp16 only), step (the LR schedule is a pure
  function of the step), torch global RNG state (weight init and dropout), CUDA RNG states (only for CUDA
  runs), and the data sampler state (generator state and batches drawn = the data position). On resume
  the log is cut back to the checkpoint step so no step appears twice. Tested on CPU only; CUDA kernels
  are not guaranteed to be bitwise deterministic, and `torch.compile` and thread-count changes are untested.
- Python and NumPy RNGs are not used inside the loop, so they are not stored.

## Config fields

JSON keys starting with `_` are ignored (use `"_note"`). Unknown keys are an error. All defaults are
PLACEHOLDER, NOT APPROVED.

`TrainConfig` (top level): `run_name`, `seed` (weight init, dropout and the training-window stream),
`data_dir` (shard directory; relative to the repo root), `model` (below), `steps` (optimiser steps),
`batch_size` (sequences per micro-batch), `grad_accum`, `peak_lr`, `warmup_steps`, `final_lr_frac`
(linear warm-up, then cosine to `final_lr_frac * peak_lr`), `weight_decay`, `beta1`, `beta2`, `eps`,
`grad_clip` (global norm, 0 = off), `decay_embeddings`, `precision` (`fp32`, `bf16` autocast, `fp16`
autocast with GradScaler on CUDA only), `compile` (`torch.compile`, off by default), `eval_every`,
`eval_windows` (number of fixed validation windows of `context_len` tokens), `eval_batch_size`,
`log_every`, `ckpt_every`, `keep_last_ckpts` (0 keeps all).

`ModelConfig` (`model`): `n_layer`, `d_model`, `n_head`, `d_ff` (0 means `round(d_ff_ratio * d_model)`),
`d_ff_ratio`, `vocab_size`, `context_len`, `pos_encoding` (`learned` or `rope`), `rope_base`, `norm`
(`layernorm` or `rmsnorm`), `mlp` (`gelu` with 2 matrices or `swiglu` with 3), `tie_embeddings`, `bias`
(linear layers and layernorm; rmsnorm and the lm_head never have one), `dropout` (one rate: embedding output,
attention weights, sub-layer outputs), `init_std`, `scale_residual_init` (GPT-2 style 1/sqrt(2 n_layer)
scaling of the two output projections of each block).

### Weight decay

AdamW with decoupled decay. Decayed: the 2-D weight matrices inside the blocks (attention and MLP). Not
decayed: norm weights, all biases, and (unless `decay_embeddings` is true) the token embedding, the learned
position embedding and an untied `lm_head`. Documented in `decay_param_names`; tested.

### Parameter counts

`param_counts(cfg)` gives the exact count from the config alone, by component: `token_embedding`,
`position_embedding`, `attention`, `mlp`, `norms`, `lm_head` (zero when tied), `total`, `embeddings`,
`non_embedding`. `check_param_count(model)` raises unless the formula equals the real count component by
component. `train()` calls it at the start of every run and stores the result in the manifest.

### Data

Shards: `meta.json` plus flat `shard_00000.bin`, ... (uint16 if `vocab_size <= 65536`, else uint32, no header).
Train and val are separated by shard (the last `val_shards` shards are val). `write_shards(texts, encode, out_dir,
vocab_size=..., tokens_per_shard=...)` writes them from any text iterator and tokenizer callable; documents never
straddle shards. Training windows are `context_len + 1` tokens drawn uniformly over all valid start positions
with a seeded `torch.Generator`. Validation uses the first `eval_windows` non-overlapping windows of the val
shards, so it needs no RNG. Bits-per-byte uses `bytes_per_token` of the whole val split (exact for the byte
tokenizer; an approximation for a subset of windows with a subword tokenizer).

### Auxiliary-loss hook

`train(cfg, run_dir, aux_loss_fn=fn)` where `fn(model, (x, y)) -> (scalar_tensor, dict_of_floats)`. The scalar
(already including any weight) is added to the language-model loss for each micro-batch under the same autocast
context; the dict is logged as `aux/<key>`, with `aux_loss` and `total_loss` (language-model loss plus aux). The
log's `train_loss` and `val_loss` stay pure language-model loss. The default is `None`. **No auxiliary loss is
implemented.** The hook is not reachable from the JSON config or the CLI: it is a Python argument, so using it
needs a small driver script. The function receives the uncompiled model; reading activations needs a second
forward pass or forward hooks, which is left to whoever writes the loss.

## What is tested (`tests/lm/`, CPU only, CUDA hidden)

Shapes; causality (a changed token never changes earlier logits) for all 32 combinations of position encoding,
norm, MLP, tying and bias; parameter formula equals the real count for those 32 variants plus others, and a
count worked by hand; RoPE rotation properties; dropout off in eval; shard writing, uint16/uint32, split by
shard, memmap windows, seeded and restorable sampler, fixed evaluation windows; LR schedule; weight-decay
groups; both position encodings learn a repeating corpus; other variants train; bf16 autocast on CPU; gradient
accumulation equals the large-batch gradient; **exact resume** (a run "crashed" at step 12 and resumed from the
step-10 checkpoint reproduces every logged loss, learning rate, gradient norm and validation loss and the final
weights of an uninterrupted run, with and without dropout); resume refuses a changed config or commit; run
directories are not overwritten; checkpoint pruning; manifest contents; clean-tree refusal, `--smoke`,
`--allow-dirty`; auxiliary-loss hook (called once per micro-batch, adds to the loss, gradient reaches the
parameters); evaluation (perplexity, bits-per-byte, batching independence).

```
set CUDA_VISIBLE_DEVICES=
.venv/Scripts/python.exe -m pytest tests/lm -q       # this package only
.venv/Scripts/python.exe -m pytest tests -q          # whole repo
```

## Not tested / not done

- fp16 + GradScaler, `torch.compile`, and any CUDA run (CPU-only so far; one CUDA test exists and is skipped without a GPU).
- Resume across different hardware, thread counts or torch versions.
- Multi-GPU, flash-attention-specific paths, KV-cache generation, text sampling.
- Real datasets and real tokenizers (nothing is downloaded). Only the byte tokenizer and the procedural
  synthetic corpus exist; the synthetic corpus is easy and says nothing about real data.
- Throughput and memory were not profiled (the dry run takes about 1.5 minutes for 300 steps on this laptop CPU).

## Values left as placeholders (all of them)

Model: `n_layer`, `d_model`, `n_head`, `d_ff` / `d_ff_ratio`, `vocab_size`, `context_len`, `pos_encoding`,
`rope_base`, `norm`, `mlp`, `tie_embeddings`, `bias`, `dropout`, `init_std`, `scale_residual_init`.
Training: `seed`, `data_dir`, `steps`, `batch_size`, `grad_accum`, `peak_lr`, `warmup_steps`, `final_lr_frac`,
`weight_decay`, AdamW `beta1`/`beta2`/`eps`, `grad_clip`, `decay_embeddings` (and which parameters are
decayed), `precision`, `eval_every`, `eval_windows`, `ckpt_every`, `keep_last_ckpts`, the tokenizer and corpus.
The schedule shape (linear warm-up then cosine) and the choice of AdamW are also not yet approved.
