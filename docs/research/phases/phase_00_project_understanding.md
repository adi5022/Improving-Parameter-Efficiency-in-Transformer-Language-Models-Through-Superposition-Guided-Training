# Phase 0 — Project Understanding and Background Assessment

Status: **substantially complete** (session 1, 2026-11-11). Remaining: user confirmation of the background summary (§3) and answers to Q-HW-1, Q-SCOPE-1.

## 1. Sources read

- `docs/PreResearch/README.md` (the proposal) — read completely.
- `CLAUDE.md` (working rules).
- There is **no root `README.md`** in this repository; CLAUDE.md's "Immediate Instruction" refers to one. Treated `docs/PreResearch/README.md` as the proposal.
- Prior work on disk (level L): `D:\Work\PROJECTS\FeatureScalpel` — README, `docs/handoff_2026-10-10_cross_model.md`, `docs/literature_review_sae_editing.md`, `papers/technical_paper/technical_paper.tex` (abstract to §"Gradient descent against the sweep"), `research/hypothesis_log.md` (first rows), `requirements.txt`, `src/` listing, git log (144 commits).
- `D:\Work\RESEARCH\1` was checked: it is an unrelated project (speculative execution for agentic systems).
- Other projects in `D:\Work\PROJECTS` were scanned by name and by a code search for transformer implementations; none found outside library code.

## 2. The proposal, decomposed

| # | Component (README section) | Type |
|---|---|---|
| 1 | GPT-style decoder from random init, full pipeline (§2) | Engineering |
| 2 | Objectives encouraging capacity per parameter, reduced interference, recoverable features, good composition (§3) | Research hypothesis |
| 3 | MI in the training loop: SAEs, attribution, steering, stability → losses (§4) | Research method |
| 4 | KD from a gpt-oss-120b-class teacher via synthetic SFT data, combined with (3) (§5) | Research method + engineering |
| 5 | 2×2 comparison: baseline / KD / MI-guided / both, matched params, data, compute; single-objective ablations (§6) | Experimental design |
| 6 | Evaluation: performance per parameter, retention, generalisation, feature utilisation, interference, convergence, quantisation robustness, speed/memory; correlation of representation quality with capability (§7) | Evaluation |
| 7 | Quantisation and local deployment; robustness hypothesis (§8) | Research + engineering |
| 8 | Overall claim: "substantially better performance per parameter" (§1, §9) | Target outcome |

## 3. Assumptions not yet validated (and what the literature says)

| Assumption in README | Status after session 1 |
|---|---|
| Superposition is a representational inefficiency that can be optimised to raise capacity per parameter | **Contradicted in its simple form** by SUP_2025_002: LMs are already in strong superposition, which drives their width scaling. |
| Reducing interference improves capability | Unsupported. Interference is the cost of capacity; no capability gain found for interference-reducing objectives in pretraining. |
| More consistent/recoverable features ⇒ better model | Unsupported; interpretable-by-design models reach ≈ equal or lower capability (SUP_2022_003, RAT_2025_004, RAT_2025_005). README §7 itself flags this. |
| SAE features are reliable enough to drive gradient updates | Doubtful: seed instability (SAE_2025_003), absorption (SAE_2024_004), weak downstream performance (SAE_2025_007/008), metric validity (SAE_2025_009). |
| A very large teacher helps a small student | Contradicted in general form by the capacity-gap result (KD_2025_001). |
| Logit-level KD from gpt-oss-120b is possible | Practically blocked by its 201k vocabulary for a small student (KD_2025_004). |
| Constrained models are more quantisation-robust | **Plausible and grounded**: training-time activation changes are known to change outliers and quantisation error (QNT_2023_001, QNT_2025_002). |

## 4. Ambiguities and unrealistic expectations

1. **"Substantially better performance per parameter"** is the headline outcome, but the only mechanism that can lower loss at a fixed objective optimum is better optimisation or better generalisation (see papers/superposition.md §2.3). "Substantially" is not supported by any prior result.
2. **Compute matching with SAE overhead** is not addressed: an online SAE costs FLOPs; a fair comparison gives the baseline the same FLOPs (more steps or tokens).
3. **"Performance" is undefined**: held-out LM loss? A benchmark suite? Instruction following? These lead to different projects.
4. **Scope:** the README contains at least five research projects (superposition optimisation, MI-in-the-loop, KD, instruction tuning, quantisation) plus a deployment system.
5. **Comparison with external models of "comparable size"** (README §7) is confounded by data and tokens; only internal, tuned baselines support causal claims.
6. **Hardware:** this machine has a GTX 1660 Ti (6 GB); the proposal's budget is unstated.

## 5. The user's background (verified from files; please correct anything wrong)

FeatureScalpel is a **four-person student team project** (the paper's author block lists four authors; handoff docs refer to "the author"). The parts listed below exist in the repository; which of them the user personally built is not determinable from files and is not assumed.

**Implemented and evidenced in the repository**
- TransformerLens 3.x + SAELens 6.x on GPT-2 Small with the pretrained `gpt2-small-res-jb` SAEs (24,576 features/layer); residual-stream hooks at `blocks.L.hook_resid_pre`.
- Feature-level causal attribution by single-feature ablation, batched (186 → 6 model calls per prompt); safety filter; mute/boost sweeps; layer sweeps (layers 6–9 plateau, layer 8 default).
- Delta patching through SAE features; **gradient-based activation optimisation** (per-feature multipliers trained with Adam against a hinge-margin + restricted-KL + size loss, model frozen); additive edits on silent features using first-order scores.
- Learned strength networks (PromptNet, FeatureNet) and a headroom study on 1,450 cached prompts.
- CounterFact evaluation with paraphrase/neighbourhood scores; ROME baseline calibration; exact McNemar, Wilcoxon, bootstrap.
- **Cross-model feature transfer**: ridge and MLP translators between GPT-2 Small and Medium residual streams; Export/Import tests; output-matching map training; pre-registered gates (Gates 2–9), several honestly reported failures.
- Process: 37 research-journal entries, hypothesis log, plans committed before runs, figures only from stored data.

**Key results that matter here**
- Gradient-descent multipliers reached rank 1 on 271/300 held-out prompts vs 161 for an equal-range sweep.
- "Rank 1 is a weak test": an additive edit forced random words to rank 1 on 20/20 prompts.
- A free residual edit reached rank 1 in 21/21 failure cells vs 1 for the SAE edit; the SAE basis beat 200 random directions 95% vs 81% in the D6 ceiling test — **the SAE helps but is not required**.

**Not found in any repository (likely learning areas)**
- Training a Transformer from random initialisation (tokenizer training, data pipeline, LR schedules, mixed precision, stability).
- Training an SAE (all SAEs were pretrained; "train a GPT-2 Medium SAE" appears only as an open option).
- Knowledge distillation, quantisation, deployment runtimes.
- Training with modified objectives where the *model's weights* receive the interpretability-derived gradient.

**Strengths to build on:** experimental discipline (pre-set gates, controls, honest negatives) is unusually strong for a student project and is exactly what this project's central question needs.

## 6. Hardware observed

`nvidia-smi` on this machine: NVIDIA GeForce GTX 1660 Ti, 6,144 MiB. FeatureScalpel docs cite an RTX 4050 laptop GPU (6 GB) — Q-HW-1.

## 7. Completion conditions

- [x] Proposal read and decomposed.
- [x] Assumptions and ambiguities listed.
- [x] Prior work investigated and summarised.
- [ ] User confirms the background summary and answers the hardware/budget/time questions.
