# Research Question

Status: **original question recorded; narrowed question PROPOSED, NOT APPROVED** (decision D-01).

## 1. Original question (docs/PreResearch/README.md §1)

> Can we build a decoder-only Transformer language model from random initialization and develop a novel training methodology that combines conventional language-model training, knowledge distillation, and mechanistic interpretability-guided optimization to achieve substantially better performance per parameter than conventionally trained models of comparable size?

## 2. Problems with the original (summary; evidence in research_gap_analysis.md and feasibility_assessment.md)

1. "Substantially better performance per parameter" has no supporting precedent; theory and evidence suggest LMs already use superposition efficiently (SUP_2025_002).
2. Three interventions (KD, MI-guided optimisation, their combination) plus instruction tuning and quantisation in one question make attribution of any effect hard and the project too broad.
3. "Performance" is undefined.
4. A negative answer is framed as failure, which pressures toward over-claiming.

## 3. Proposed narrowed question (for discussion) — SUPERSEDED by §5 (it misread the direction of the proposal; kept for history)

> **Can auxiliary training objectives derived from a sparse autoencoder trained online on a small Transformer's own activations improve its held-out language-modelling performance, generalisation, or quantisation robustness, relative to compute-matched, hyper-parameter-tuned baselines and to equal-cost regularisers that do not use interpretability tools?**

Secondary question:

> Do representation metrics measured during training (interference/superposition statistics, feature stability, SAE fidelity) predict those outcomes across runs?

## 4. Alternatives considered

| Option | Description | Pros | Cons |
|---|---|---|---|
| A (recommended) | Narrowed question above, compute-limited and/or data-limited regime | Tests the README's core idea; null result publishable; feasible on small GPUs | Likely null for capability |
| B | Diagnostic-only: track features/superposition across training of small LMs under different standard hyper-parameters (weight decay, width, data size) | Cheap, low risk, clean | Drops "interpretability influences training" — the README's central aim |
| C | Quantisation-focused: feature-level vs outlier-suppression training for PTQ robustness and feature survival | Best-grounded positive link | Narrower; strong existing baselines (OSP) |
| D | Original README scope | Ambitious | Not feasible to do rigorously; attribution impossible |

Option A can absorb C as its quantisation outcome and B as its secondary question.

## 5. Clarified proposal from the user (2026-11-11, same session)

The user corrected the first assessment, which had partly read the proposal as being about interpretability and *reducing* interference. The user's statements, recorded as given:

1. **Goal: more superposition, not less.** The model should be pushed to superimpose *more* concepts into its activation space than standard training does, so a smaller model stores as much as a larger one. Interpretability of the final model is explicitly **not** a goal ("sacrificing interpretability for storage"). Monosemanticity is not wanted.
2. **Mechanism: a reward/penalty term inside the training loop** (not a separate reward model). Interpretability (SAEs) is used only to *measure* the amount of superposition; that measurement drives the penalty.
3. **SAE use:** the SAE reconstruction loss is taken into account; if an SAE's output is ever inserted into the forward pass, only **deltas via delta patching** (as in FeatureScalpel) are added, never the raw reconstruction.
4. **SAE kept current by joint training:** the user proposes updating the SAE weights in the same loop as the transformer, so the SAE tracks the model (and fits the model's final output).
5. **Success criterion (illustrative sizes):** model A (our method, ≈ 2 GB) vs model B (standard training, ≈ 3 GB); quantise both (≈ 1 GB vs ≈ 1.5 GB). If A ≈ B in performance, the method wins. Quantisation stays in scope.
6. **Toy model first:** prove the mechanism on a toy model before any small language model.
7. **Teacher:** use a smaller teacher (solves the vocabulary/capacity-gap issues); distillation remains a later factor.
8. **Scope:** reuse standard components (tokenizer, attention, model code); new work is the training path only.

### Restated research question (draft — wording needs user approval, D-01)

> **Can an auxiliary penalty, computed from an SAE trained jointly with the model and measuring how much superposition the model uses, push a Transformer to store more features per dimension than standard training — so that a smaller model matches the performance of a larger, conventionally trained one, including after quantisation?**

Evaluated first in the toy model of superposition (ground-truth features known), then in small language models.

### Points of agreement with the literature
- SUP_2025_002 explicitly names "encouraging superposition might let smaller models match larger ones" as an implication (via the coefficient of the ~1/width loss law).
- RAT_2025_003 (L2E) pushes against monosemantic neurons during pretraining in the same direction (weak evidence).

### Open design problems raised by the clarification (see open_questions.md Q-DES-1…5)
- Superposition is a property of directions in the space, not of individual neurons; the measure must count features per dimension, otherwise a basis rotation "wins" the penalty without storing anything.
- Standard training already balances packing against interference for its own loss; pushing further must win through optimisation speed, generalisation, or capacity the model has not yet used.
- Denser packing may make features *more* fragile under quantisation (QNT_2026_003), which directly threatens the A-vs-B success criterion.
- The measuring SAE must not be allowed to learn to report more superposition (Goodhart); the independent check is ground truth in the toy model and fresh, independently trained SAEs on checkpoints.
- Weight decay (including negative) already controls superposition in the toy model (SUP_2025_002) and is the mandatory cheap baseline.
