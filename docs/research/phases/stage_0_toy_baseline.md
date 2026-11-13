# Stage 0 — Reproduce the Toy-Model Baseline (Liu, Liu & Gore 2025)

Status: **S0-1…S0-7 approved by the user 2026-11-11.** Implementation started: Claude's scaffolding (runner, metrics, tests, analysis); the user's core (model, sampler, optimiser) is pending. Nothing run yet. The gates in §4 are still a draft and must be fixed before the first real run.

## 1. Objective

1. Build a trusted, seeded re-implementation of the toy model of SUP_2025_002 that reproduces weak vs strong superposition and their width-scaling behaviour.
2. Produce the **equivalent-width yardstick**: L*(m) = the best (over weight decay γ) mean test loss at width m. Every later stage is judged by whether a width-m model with our penalty reaches L*(≈ 1.5·m) — the toy version of "our 2 GB model matches their 3 GB model".
3. Measure seed-to-seed variance, which sets the margins for the Stage 1 go/no-go gate.

## 2. Reference setup (verified from the paper HTML and the official code, 2026-11-11)

Source code inspected: `github.com/liuyz0/SuperpositionScaling`, files `exp/exp-10.py`, `exp/exp-10-3.py`, `exp/adamw.py` (downloaded to the session scratchpad for reading only). **The repository has no licence**, so we re-implement rather than copy.

| Item | Value | Source |
|---|---|---|
| Features n | 1000 | paper §3 / App. B.2; code |
| Widths m | 10, 15, 25, 39, 63, 100 (log-spaced) | paper; code (`floor(exp(linspace(log10, log100, 6)))`) |
| Model | h = Wᵀx (W ∈ ℝ^{n×m}), y = ReLU(W h + b) | paper §2; code `ReLU(x @ W @ W.T + b)` |
| Init | W ~ N(0,1)/√m; b ~ N(0,1) (not zero) | code |
| Data | x_i = Bernoulli(p_i) · U(0,2), fresh batch each step | paper; code |
| Frequencies | p_i ∝ i^(−α), normalised so Σp_i = E = 1 (expected one active feature per sample) | paper; code uses exponent 1+alpha with alpha ∈ [−1,0] (exp-10-3) and [0,1] (exp-10), i.e. α ∈ [0,2], 17 values |
| Loss | MSE ‖y − x‖² | paper; code |
| Optimiser | AdamW, decay on W only (b: 0), peak lr 1e-2, 2,000-step linear warm-up, cosine decay to 0.05 × peak | paper App. B.2; code |
| Steps / batch | 20,000 / 2,048 | paper; code |
| Weight decay γ | linspace(−1, 1, 10) | paper; code |
| γ ≥ 0 | W ← W·(1 − lr·γ) (decoupled) | `adamw.py` |
| γ < 0 | W ← W + lr·γ·W·(1 − 1/‖W_i‖) row-wise, pulls row norms toward 1 | `adamw.py` |
| Test loss | MSE on 100 × batch size fresh samples | paper |
| Represented fraction | ϕ_{1/2} = #{i : ‖W_i‖ > 1/2}/n; strongly represented ϕ_1 uses threshold 1 | paper Eq. 3, App. A.1 |
| Strong regime signature | ϕ_{1/2} ≈ 1 ≫ m/n; row norms bimodal; mean squared overlap of unit rows ∝ 1/m | paper §3 |
| Seeds | **none set in the reference code** | code |

Discrepancies noted: the paper says the hyper-parameter scan used 96 CPU cores; the script expects 90 tasks. Weight initialisation and seed count are not stated in the paper text read. Neither affects our reproduction plan.

## 3. Proposed experiment grids

- **G-A (yardstick, the important one):** α = 1 (Zipf-like, the case the paper links to natural language), widths m ∈ {10, 15, 25, 39, 63, 100, **150**} (150 added so 1.5 × 100 is inside the measured range), γ ∈ {−1, −0.55, −0.1, 0, 0.1, 0.55, 1}, **3 seeds** → 147 runs.
- **G-B (exponent check):** α ∈ {0.5, 1.5, 2.0}, widths = the six paper widths, γ ∈ {−1 (strong), 0.55 (weak)}, 2 seeds → 72 runs.

**Compute estimate (not measured):** a step costs ≈ 2.5 GFLOP at m = 100; a 20k-step run should take ≈ 0.5–1 min on the GTX 1660 Ti (dominated by kernel-launch overhead), so ≈ 2–4 h for both grids. Small enough to run on CPU if needed (slower).

## 4. Proposed gates (pre-registration draft — to be fixed before running)

| Gate | Check | Pass rule (proposed) |
|---|---|---|
| G0.1 | Regimes appear | At α = 1, m = 100: γ = −1 gives ϕ_{1/2} ≥ 0.9; γ = +1 gives ϕ_{1/2} ≤ 3·m/n (= 0.3) |
| G0.2 | Strong-regime width exponent | At α = 1, γ = −1: fitted loss ∝ m^(−α_m) with α_m ∈ [0.8, 1.2] |
| G0.3 | Weak-regime exponent | At α ∈ {1.5, 2.0}, γ = 0.55: α_m within ±0.2 of α − 1 |
| G0.4 | Variance known | Report seed s.d. of final test loss at every (m, γ) in G-A (no pass/fail; feeds Stage 1 margins) |

If G0.1–G0.3 fail: compare against the reference implementation line by line before anything else; Stage 1 does not start on an unverified baseline.

## 5. Outputs

- Code (small, standalone): model, data sampler, optimiser with negative weight decay, sweep runner, analysis.
- Results: final test loss, ϕ_{1/2}, ϕ_1, row-norm histograms, mean squared overlaps, per (α, m, γ, seed); saved W matrices.
- Figures from stored data only: loss vs m (log-log) per γ; ϕ vs γ; row-norm histograms (strong vs weak); the yardstick L*(m) with seed error bars.
- An experiment record per run group (CLAUDE.md §14) and a journal entry.

## 6. Decisions (CLAUDE.md §9) — all approved by the user 2026-11-11

| ID | Decision | Approved choice | Why |
|---|---|---|---|
| S0-1 | Re-implement vs copy the reference code | Re-implement (reference used for reading/cross-checks only) | No licence on the repository; re-implementing is also the best way for the user to learn it |
| S0-2 | Grids | G-A + G-B as above (≈ 220 runs) instead of the full 1,020-run scan | Covers the yardstick and the exponent checks at ~20% of the cost |
| S0-3 | Seeds | Fixed seeds; 3 for G-A, 2 for G-B | Reference has none; variance is needed for Stage 1 margins |
| S0-4 | Baseline definition for later stages | Report both plain AdamW (γ = 0) and best-tuned γ per width; **success claims only against best-tuned** | Avoids beating an untuned baseline |
| S0-5 | Environment | New project venv (Python 3.12), PyTorch with CUDA for the GTX 1660 Ti (≈ 2–3 GB download), numpy, matplotlib | Global Python 3.13 has no PyTorch; keep the project isolated |
| S0-6 | Version control | `git init` this project now; commit code before each run (experiment records cite commits) | CLAUDE.md §14 requires a code version per experiment; the folder is not a git repository |
| S0-7 | Who writes the code | User writes the model, data sampler and negative-weight-decay optimiser (~80–120 lines) after Tier-1 reading; Claude writes the sweep runner, tests against the reference behaviour, and the analysis; both review each other's parts | User asked to be closely involved; the core is small and teaches the mechanics |

**Implementation note (2026-11-11, minor, not a methodology change):** S0-5 named Python 3.12, but the 3.12 entry on this machine points to a deleted executable. The venv uses Python 3.13, which already runs CUDA PyTorch on this GPU (FeatureScalpel venv: torch 2.6.0+cu124, sm_75 present).

## 7. Reference-code details that matter for our implementation (read 2026-11-11)

- **Loss reduction:** `nn.MSELoss()` = mean over batch *and* features (i.e. ‖y−x‖²/n per sample). We use the same reduction so numbers are comparable.
- **The reference code records only the per-step training loss; it computes no separate test loss.** Our test loss (100 × 2,048 fresh samples, fixed test seed) is our addition, from the paper's description.
- **Weight-decay order:** decay is applied to W *before* the Adam update, using the current step's lr — the same order as `torch.optim.AdamW`. So for γ ≥ 0 the custom optimiser must match `torch.optim.AdamW` exactly; this is one of our tests.
- **Negative decay:** row norms get `+ eps` (1e-8) before `1/‖W_i‖`.
- **LR schedule:** `step+1` indexing; linear warm-up for steps < 2,000, then cosine from peak to 0.05 × peak.
- Default Adam settings: betas (0.9, 0.999), eps 1e-8, no AMSGrad.

## 8. Code layout and how to run (scaffold written 2026-11-11)

| File | Author (S0-7) | Content |
|---|---|---|
| `toy/model.py` | **user** (stub) | `ToyModel(n, m)`: W, b, ReLU(x W Wᵀ + b) |
| `toy/data.py` | **user** (stub) | `feature_probs(n, alpha)`, `sample_batch(batch, probs, generator)` |
| `toy/optim.py` | **user** (stub) | `AdamWNegDecay`: AdamW with signed (row-wise negative) weight decay |
| `toy/schedule.py` | Claude | LR warm-up + cosine, identical to reference `get_lr` (tested) |
| `toy/metrics.py` | Claude | ϕ½, ϕ₁, mean squared overlap, chunked test loss |
| `toy/train.py` | Claude | one seeded run → result dict (W, b, row norms, losses, metrics, timing) |
| `toy/sweep.py` | Claude | grid runner; resumes; refuses uncommitted code (S0-6) |
| `toy/analysis.py` | Claude | yardstick L*(m), equivalent width, exponents, gates, figures, report |
| `configs/stage0.json` | Claude | grids G-A, G-B; draft gate thresholds |
| `tests/test_model.py`, `test_data.py`, `test_optim.py` | Claude | contracts for the user's files; **skip while stubs** |
| `tests/test_pipeline.py` | Claude | Claude's half, using deliberately different stand-in components |

Each user file's docstring states the exact contract and lists questions worth answering from R1–R2.

```
.venv\Scripts\python -m pytest -q                      # user tests skip until implemented
.venv\Scripts\python -m toy.sweep --grid G-A --smoke   # 2 short runs, results/smoke/
.venv\Scripts\python -m toy.sweep --grid G-A           # full grid (commit first)
.venv\Scripts\python -m toy.analysis                   # report + figures
```

Seeding choice inside S0-3 (implementation detail): the run seed seeds both init and the data stream, so runs differing only in γ share init and data (paired comparison); the test set has its own fixed seed (10,000).

Before the first real run: (1) user's three files pass their tests; (2) Claude reviews them and the user reviews Claude's files; (3) gate thresholds in `configs/stage0.json` are fixed (pre-registration) and committed; (4) confirm timing with the real model.

**Timing measured 2026-11-11 (pipeline with stand-in components of the same matmul cost, GPU, 1,000 steps):** ≈ 3.7–4.1 s per 1,000 steps at both m = 10 and m = 150, i.e. ≈ 1.2–1.4 min per 20k-step run, nearly independent of width (data sampling and kernel launches dominate). Both grids (219 runs) ≈ **4.5–5 h**, above the earlier 2–4 h estimate. To be re-measured with the user's model.

Environment (S0-5): `.venv` on Python 3.13.7, torch 2.14.1+cu126 (CUDA available, sm_75 in arch list), numpy, matplotlib, pytest; exact versions in `requirements-lock.txt`.

## 9. Change to S0-7 (2026-11-13)

The user asked Claude to write all the code ("just give me the code"), so Claude also wrote `toy/model.py`, `toy/data.py` and `toy/optim.py`, overriding the S0-7 split (user writes the core). Full suite: 33 passed, 0 skipped. The docstrings in those files explain the mechanics. The user still reviews them before the first real run.

Short sanity runs with the real model (alpha = 1, m = 100, 2,000 steps, seed 0; **not** a gate result): gamma = -1 gave phi_half 1.000, test loss 2.5e-4; gamma = +1 gave phi_half 0.012, test loss 9.4e-4. Real-model speed ≈ 3.4-3.8 s per 1,000 steps, so ≈ 1.2-1.3 min per 20k-step run; both grids ≈ 4.5 h.
