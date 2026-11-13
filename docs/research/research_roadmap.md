# Research Roadmap

Status: **staged plan accepted by the user as the working plan (2026-11-11); individual stage designs and gates still need approval before each stage runs.**

Revision history:
- 2026-11-11 (v1): phases 0–9 built around a narrowed "SAE signal vs non-MI regulariser" study. **Superseded** — it was based on a misreading of the goal (interference reduction).
- 2026-11-11 (v2, this version): rebuilt around the user's clarified goal — push a model to store **more features per dimension** with an in-loop penalty measured by a jointly trained SAE, toy model first (research_question.md §5; hypotheses HS1–HS5).

Durations assume ~15–20 h/week (unconfirmed, Q-SCOPE-1). The user wants to be closely involved: each stage starts with a short reading block and ends with a review of results together.

## Overview

| Stage | Name | Hardware | Est. duration | Status | Gate to pass before next stage |
|---|---|---|---|---|---|
| — | Phase 0: project understanding | — | done | Substantially complete | — |
| — | Phase 1: literature review | — | ongoing | First pass done; reading list issued to user | ★ sources verified before they justify approved decisions |
| **0** | **Reproduce the toy-model baseline (Liu et al.)** | Laptop GPU/CPU | 1–2 weeks | **Started (planning)** | G0.1–G0.3 reproduction checks; seed variance measured |
| 1 | Oracle penalty (ground-truth features) — go/no-go | Laptop | 1–2 weeks | Not started | **G1: oracle beats tuned weight decay at width m by a pre-set margin, ≥ 5 seeds; else stop main line** |
| 2 | Real method: jointly trained SAE + penalty (D-09) | Laptop | 2–4 weeks | Not started | G2: recovers a pre-set share of the oracle's gain; beats non-SAE penalty; SAE measurement tracks ground truth (HS5) |
| 3 | Toy model that computes (tiny transformer, synthetic task with known features) | Laptop | 3–4 weeks | Not started | G3: gain survives when features must be used, not just stored |
| 4 | Small LM (≈ 5–20M, TinyStories-scale) + synthetic fact-storage test | Laptop / rented | 6–8 weeks | Not started | G4 (HS3): penalised model ≈ standard model ≈ 1.5× larger, ≥ 3 seeds |
| 5 | Quantise both, compare (A vs B) | Laptop | ~2 weeks | Not started | G5 (HS4): gap persists after identical quantisation |
| Side | A1 superposition ↔ quantisation fragility; A2 Muon vs AdamW packing | Laptop | parallel to 0–2 | Proposed | — |
| Late | A3 feature-level distillation via translators (replaces plain KD); A4 crosscoder diffing across scale | Laptop / rented | optional | Proposed | — |
| End | Analysis and write-up (post / workshop paper) | — | 3–4 weeks | — | — |

Total: Stages 0–2 ≈ 1–2 months; Stages 0–5 ≈ 4–6 months.

If **G1 fails**, the main line stops and the result ("even a perfect guide cannot beat tuned weight decay in the toy model") is written up; A1 or A2 becomes the main project (alternative_directions.md).

## Stage details

### Stage 0 — Reproduce the baseline
- **Objective:** a trusted re-implementation of the Liu et al. toy model (SUP_2025_002) that reproduces weak/strong superposition and their width-scaling exponents; and the **equivalent-width yardstick** L*(m) used to judge every later stage ("does our width-m model reach L*(1.5·m)?").
- **Plan, decisions and gates:** [phases/stage_0_toy_baseline.md](phases/stage_0_toy_baseline.md).
- **Reading first:** reading_list.md Tier 1.

### Stage 1 — Oracle penalty (go/no-go)
- **Objective:** with the true feature set known, write the best-case "pack more, keep them distinguishable" penalty on the model and test whether *any* such pressure beats tuning weight decay (positive and negative) at fixed width.
- **Why first:** cheapest way to kill or confirm the core idea before building the SAE machinery.
- **Design decisions (to propose):** penalty form (e.g. crosstalk between true feature directions, represented-feature count surrogate), strength sweep, margin for G1 from Stage 0 seed variance.
- **Reading:** Tier 2 (Scherlis et al. capacity; Henighan et al.).
- **Candidates (user-approved for inclusion 2026-11-14; designs still to be proposed in section 9.1 format before any code):**
  - **1a. Overlap penalty on independent features** (frame potential / coherence on the feature directions; packing_ideas.md idea 1). Recommended first.
  - **1b. Tree-structured toy variant (user idea, packing_ideas.md idea 6):** features form a tree (a parent is on whenever any of its children is on) instead of being independent. Test whether a "share the parent, store only child offsets" penalty beats tuned weight decay at the same width. Same gate G1 and same yardstick, run on the tree data. Open design points: tree shape (depth, branching), how common parents are, how the penalty is written, and whether the model already shares parents without any penalty (measure that first).
  - All candidates are compared against tuned weight decay with the seed noise from Stage 0.
- **Reading:** Tier 2 (Scherlis et al. capacity; Henighan et al.); Park et al. 2025 (RAT_2025_013) for 1b.

### Stage 2 — SAE-measured penalty with joint SAE training
- **Objective:** replace the ground truth with an SAE trained in the same loop (D-09: SAE updated on its own reconstruction/sparsity loss with detached activations; penalty gradient never reaches the SAE; penalty gated on reconstruction quality; delta patching for anything inserted, U-03).
- **Controls:** non-SAE penalty of equal cost (e.g. use-all-dimensions / effective-rank pressure); random-dictionary penalty; tuned weight decay.
- **Validity check (HS5):** SAE's superposition estimate vs ground truth.
- **Reading:** Tier 2 SAE papers.

### Stage 3 — Toy model that computes
- **Objective:** repeat Stages 1–2 where packed features must be used downstream (computation in superposition), e.g. a tiny transformer on a synthetic task with known features.
- **Reading:** Tier 3 (Hänni et al.).

### Stage 4 — Small language model + fact storage
- **Objective:** HS3 at small LM scale; a controlled synthetic-facts set to measure knowledge stored per parameter (Allen-Zhu & Li 3.3).
- **Decisions needed:** all architecture/training items (D-08 list), each in CLAUDE.md §9.1 format; reuse standard components (U-07).

### Stage 5 — Quantisation A vs B
- **Objective:** quantise the penalised model and the ≈ 1.5× standard model identically; compare held-out performance and feature survival (QNT_2026_003 protocol).

## Dependencies and risks
- Hardware/budget answers (Q-HW-1/2) gate Stage 4 scale.
- Risks: G1 fails (acceptable, reportable); SAE measurement gamed (mitigated by stop-gradient + independent ruler); denser packing more quantisation-fragile (tested in A1 / Stage 5); baseline under-tuned (mitigated by tuned-γ yardstick).
