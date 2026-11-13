# Research Index

Project: Mechanistically Guided Training of Parameter-Efficient Language Models.
Original proposal: [../PreResearch/README.md](../PreResearch/README.md). Working rules: [../../CLAUDE.md](../../CLAUDE.md).

Last updated: 2026-11-13 (session 2).

## Current research direction

Under review. The original proposal ("substantially better performance per parameter" via MI-guided optimisation + distillation) is **not supported as stated** by the literature. The recommended direction (pending approval, D-01) is a narrowed, controlled study:

> Can auxiliary training objectives derived from an SAE trained online on a small Transformer's own activations improve held-out LM performance, generalisation, or quantisation robustness, relative to compute-matched, tuned baselines and equal-cost non-interpretability regularisers?

Recommendation: **pursue with significant modifications / narrowed scope** ([feasibility_assessment.md](feasibility_assessment.md) §5).

**Update (2026-11-11, later in session 1):** the user clarified that the goal is to push the model to use **more** superposition (store more per dimension, interpretability of the final model not a goal), with an in-loop penalty measured by a jointly trained SAE, judged by a smaller-model-vs-larger-model comparison after quantisation, and tested on a **toy model first**. The current draft question is in [research_question.md](research_question.md) §5; hypotheses HS1–HS5 in [hypotheses.md](hypotheses.md); user directions U-01…U-08 and D-09 in [architectural_decisions.md](architectural_decisions.md). Parts of the gap analysis and feasibility assessment written before this clarification (which assumed interference reduction) need revision.

## Status

| Phase | Status |
|---|---|
| 0 Project understanding | Substantially complete — awaiting user confirmations |
| 1 Literature review | First pass complete — verification pass pending |
| 2 onward | Not started (blocked on decisions) |

Stage 0: environment ready (`.venv`, torch 2.14.1+cu126), git initialised (no commits), Claude's half of the code written and tested ([../../toy/](../../toy/)); all code written (the user asked Claude to write the three core files too), 33 tests pass. No experiments have been run.

## Key documents

| Document | Purpose |
|---|---|
| [literature_review.md](literature_review.md) | Seven cross-area conclusions |
| [research_gap_analysis.md](research_gap_analysis.md) | What exists, candidate gaps G1–G5, narrowed question |
| [feasibility_assessment.md](feasibility_assessment.md) | Mechanism feasibility, compute, scopes, final recommendation |
| [portfolio_assessment.md](portfolio_assessment.md) | Value by role; what a reviewer could verify |
| [research_question.md](research_question.md) | Original vs proposed question, alternatives |
| [hypotheses.md](hypotheses.md) | Candidate hypotheses H1–H7 (not approved) |
| [methodology.md](methodology.md) | Draft principles (controls, compute matching, seeds) |
| [architectural_decisions.md](architectural_decisions.md) | Decision register D-01…D-07 (none approved) |
| [open_questions.md](open_questions.md) | Blocking questions for the user; literature to-dos |
| [research_roadmap.md](research_roadmap.md) | Proposed phases 0–9 |
| [alternative_directions.md](alternative_directions.md) | Side / follow-up directions A1–A4 (not approved) |
| [packing_ideas.md](packing_ideas.md) | Ideas for fitting more into the same space, from the full text of SUP_2025_002 plus a search (candidates, not approved) |
| [reading_list.md](reading_list.md) | User's reading list R1–R15 with questions, by stage |
| [reading_log.md](reading_log.md) | User's notes and answers |
| [phases/stage_0_toy_baseline.md](phases/stage_0_toy_baseline.md) | Stage 0 plan, reference settings, gates, decisions S0-1…S0-7 |
| [paper_library.md](paper_library.md) | Full records + verification levels |
| [citation_index.md](citation_index.md) | Claim → source lookup |
| [papers/](papers/) | Per-area syntheses |
| [phases/](phases/) | Phase documents |
| [../notes/](../notes/) | Session journals |

## Approved decisions

- **S0-1…S0-7** (Stage 0 toy-model baseline), approved 2026-11-11 — see [architectural_decisions.md](architectural_decisions.md). Nothing else approved yet.

## Outstanding questions (blocking)

Q-HW-1 (GPUs), Q-HW-2 (budget), Q-SCOPE-1 (hours/deadline), Q-SCOPE-2 (solo/team), Q-BG-1 (background confirmation), D-01…D-07.

## Next recommended action

1. **Stage 0 (S0-1…S0-7 approved):** user reads R1–R2 and implements `toy/model.py`, `toy/data.py`, `toy/optim.py` until all tests pass; then mutual review, fix gate thresholds, commit, run G-A and G-B (≈ 4.5–5 h), analyse. Details: phases/stage_0_toy_baseline.md §8.
2. Still pending: D-09 approval, restated question wording (research_question.md §5), Q-HW-1/2, Q-SCOPE-1/2.
3. Revise research_gap_analysis.md and feasibility_assessment.md for the "more superposition" goal (banners mark the outdated parts).
4. Close Phase 1: full-text reads of ★ sources (start with SUP_2026_004, the likeliest overlap) and the Q-LIT-2 prior-art searches.
5. Phase 2: pre-register hypotheses; then Phase 3 architecture decisions in §9.1 format.
