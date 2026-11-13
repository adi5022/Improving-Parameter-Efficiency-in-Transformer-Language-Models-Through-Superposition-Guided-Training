# Feasibility Assessment

> **Banner (2026-11-11, later in session 1):** written before the user's clarification (goal = more superposition; toy model first; quantisation A-vs-B kept). Compute estimates and the KD/SAE-reliability analysis still apply; the recommended scope (§5) is superseded by research_roadmap.md (v2). A full revision is pending.

Session 1 (2026-11-11). Covers CLAUDE.md §7.2, §7.3, §7.5, §7.6. Compute numbers are planning estimates with stated assumptions, not measurements.

## 1. Technical feasibility, mechanism by mechanism

| Mechanism (README) | Implementable with existing tools? | Main obstacle | Verdict |
|---|---|---|---|
| From-scratch decoder + pipeline | Yes (PyTorch; nanoGPT-style) | User has not done pretraining before; tuning a fair baseline is the hard part | **Feasible**, learning cost 2–4 weeks |
| Online SAE at 1–2 layers during training | Yes (custom loop; SAELens for offline SAEs) | Non-stationary target; refresh schedule; extra compute; dead latents | **Feasible at small scale**, needs a pilot |
| SAE-derived auxiliary loss into model gradients | Yes (differentiate through SAE encoder/decoder, SAE frozen per refresh window) | Goodhart risk (model learns to please the SAE); seed instability of features | **Feasible**, scientifically risky — that is the experiment |
| Attribution / causal-intervention-derived losses | Possible (ablation effects per feature are expensive: one forward per feature, or gradient approximations) | Cost scales with number of features; noisy | **Postpone**; use cheaper SAE statistics first |
| Activation steering during training | Yes (preventative-steering style) | Targets specific concepts, not capacity | Out of scope for the capability question |
| Cross-checkpoint feature stability analysis | Yes (SAEs/crosscoders on checkpoints) | Compute for many SAEs | **Feasible** as analysis |
| KD from gpt-oss-120b, logit level | No in practice (201k vocab; 80 GB GPU or API; storage) | Vocabulary, cost | **Not feasible** as proposed |
| KD sequence level (synthetic data) | Yes (API or rented GPU) | Confounds data with method; cost | Feasible, **low research value now** |
| Instruction tuning a ≤ 30M model | Yes | Little capability to show | **Remove** |
| PTQ (INT8/INT4) + feature survival | Yes (simple RTN/GPTQ-style; frozen-SAE protocol of QNT_2026_003) | None major | **Feasible and cheap** |
| Local deployment | Trivial at this size | Not research | Keep minimal |

## 2. Compute

**Hardware observed:** GTX 1660 Ti 6 GB (Turing: no tensor cores, no bf16, no FlashAttention-2). Assumed effective throughput ≈ 3 TFLOP/s FP16 mixed precision — **must be benchmarked in Phase 3**.

Training FLOPs C ≈ 6·N·D. Online SAE overhead estimated at +10–40% depending on layers instrumented and refresh frequency (to be measured).

**A controlled study is many runs.** Minimum credible screening grid for one regime:
- Baseline tuning: ~3 learning rates × 2 weight-decay values = 6 runs.
- Arms: baseline, 2 non-MI regularisers, 2 SAE-derived losses, 1 random-dictionary control = 6 arms × 3 seeds = 18 runs, plus ~2 strength values per regulariser → ~30 runs.
- Total ≈ 35–40 runs per model size per regime.

| Scale | Per-run time (1660 Ti, est.) | 40 runs | Practical? |
|---|---|---|---|
| ~5M params, ~100M tokens | ≈ 0.3–0.5 h | ≈ 15–20 h | **Yes, locally** |
| ~20–30M params, ~0.5B tokens | ≈ 6–10 h | ≈ 250–400 h | Only a reduced grid locally; better rented |
| ~124M params, ~2.5B tokens | ≈ 170 h | infeasible locally | Rented GPU: ≈ 1.5–4 h/run on one H100/A100 → ~100–150 GPU-h for a reduced confirmation grid |

Rented-GPU cost depends on provider and is not estimated here (no price check done; ask before committing money).

**Recommendation:** screen at ≈ 5–20M parameters locally; confirm only the surviving arms at one larger size on rented GPUs.

## 3. Research value (§7.3)

- Academic contribution: an empirical, controlled answer to whether interpretability-derived training signals beat non-interpretability regularisers — currently answered only by small, unseeded, opposite-sign claims. Realistic venues: workshop papers (e.g. mechanistic-interpretability or efficient-ML workshops at NeurIPS/ICLR/ICML), BabyLM workshop if G2 is pursued. A main-conference paper would need a clearly positive, scaled result or a very thorough negative one.
- Reproducibility: high if configs, seeds, logs and checkpoints are released; small models make independent reproduction cheap.
- Relevance: training-time interpretability and SAE reliability are active topics in 2025–2026.

## 4. Time investment (§7.5)

Hours per week are unknown (Q-SCOPE-1). Estimates assume ~15–20 h/week of focused work.

| Scope | Content | Duration | Difficulty |
|---|---|---|---|
| **Minimum viable** | Tuned baseline pipeline at ~5–20M; offline SAEs on checkpoints with random-init controls; 1 SAE-derived loss vs 1 non-MI regulariser vs random dictionary, 3 seeds, one regime; PTQ + feature-survival measurement | ~3–4 months | Moderate–high |
| **Realistic** | MVP + second regime (compute-limited vs data-limited) + second SAE-derived loss + representation-metric correlation analysis + one rented-GPU confirmation at larger scale + write-up | ~5–7 months | High |
| **Ambitious** | Realistic + KD factor (teacher with shared tokenizer) + attribution-based losses + scaling to ~124M with full grids | ~9–12 months | Very high; compute cost material |

**Major risks:** (1) null result for capability (likely — must be designed as acceptable); (2) unfair baseline tuning → spurious positive; (3) online-SAE instability or dead latents; (4) compute shortfall on a 6 GB Turing GPU; (5) scope creep back to KD/instruction-tuning/deployment; (6) prior art appears (Q-LIT-2).

**Dependencies:** hardware/budget answer; approved research question; dataset choice; baseline recipe.

**Likely bottlenecks:** baseline tuning; GPU hours; the SAE refresh schedule; statistical power with 3 seeds at small effect sizes.

**Learning outcomes:** pretraining engineering, SAE training, controlled-experiment design at training time, quantisation measurement.

## 5. Final recommendation (§7.6)

**Pursue with significant modifications, narrowing the scope to a more defensible question** (gap G1, optionally G2/G3; question in [research_gap_analysis.md](research_gap_analysis.md) §4).

Evidence summary:
- Against the original framing: LMs already use strong superposition as their scaling mechanism (SUP_2025_002); interpretable-by-design training has never shown capability gains at matched compute (SUP_2022_003, RAT_2025_004, RAT_2025_005); SAE signals are unstable and often beaten by simple baselines (SAE_2025_003/007/008/009); the proposed teacher is a poor fit (KD_2025_001, KD_2025_004).
- For a narrowed version: the specific combination (online SAE → auxiliary loss → from-scratch pretraining → compared with non-MI regularisers and random dictionaries) was not found; the sign of monosemanticity effects is contested (RAT_2024_001 vs RAT_2025_003); interpretability already influences fine-tuning successfully (RAT_2025_009), so the pretraining question is natural; quantisation links are grounded (QNT_2023_001, QNT_2025_002, QNT_2026_003).

**Change or remove from the README:**
- Remove the target "substantially better performance per parameter"; replace with a testable question and equivalence margins.
- Remove instruction tuning and deployment-as-research.
- Postpone KD (and drop gpt-oss-120b logit KD entirely).
- Postpone attribution/causal-intervention losses until the cheap SAE-statistics losses are tested.
- Add mandatory controls: tuned weight decay/dropout, non-MI regularisers, random/PCA dictionaries, random-init interpretability controls, ≥ 3 seeds.

**Worth several months?** Yes, *if* the user accepts that the most likely capability answer is "no meaningful gain" and values a rigorous answer, a strong from-scratch pipeline, and SAE-training experience. No, if the goal is a demonstrably better small LM — the evidence does not support expecting that.
