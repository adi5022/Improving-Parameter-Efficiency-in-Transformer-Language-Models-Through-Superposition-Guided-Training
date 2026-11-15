# Research Index

Project: Mechanistically Guided Training of Parameter-Efficient Language Models.
Original proposal: [../PreResearch/README.md](../PreResearch/README.md). Working rules: [../../CLAUDE.md](../../CLAUDE.md).

Last updated: 2026-11-15 (session 2).

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

**Toy-model stages (updated 2026-11-15; see [../notes/2.md](../notes/2.md) Addenda 6-20 and [phases/stage_0_toy_baseline.md](phases/stage_0_toy_baseline.md)):**
- **Stage 0 (baseline) done:** gates G0.1-G0.3 passed; 10-seed weight-decay baseline L*(m) established at widths 10-150. A follow-up grid G-E (gamma -3/-4, width 225) is queued on Kaggle and not yet collected.
- **Stage 1 (oracle overlap penalty) done: gate G1 PASS, 3 of 3 widths** (rule fixed before the final run; weighted penalty selected on seeds 0-2, judged on seeds 3-9: +10.8% / +27.5% / +45.9% lower error at widths 15 / 25 / 63, matching a model 1.5x wider at width 63). Oracle case only (true features and frequencies known); width 15 is borderline pair-wise (+8.4%). Details: notes/2.md Addendum 20.
- **Stage 2 (SAE-measured penalty): partly approved 2026-11-15** (D-09, D-S2-1 bake-off, D-S2-2, D-S2-3, D-S2-10 order, 10%/3 s.e. pass rule; see [architectural_decisions.md](architectural_decisions.md)). **Step S2-0 (build) done:** leakage penalty, three SAE types, trainer integration, 177 tests pass. SAE-L steps cost about 3.5-4.4x a baseline step (measured), so the 6-7 GPU-hour estimate is probably 10-12. Next: write the S2-A bake-off script and ask before running it. Still open: D-S2-4..8 details. G-E results collected: width-63 baseline unchanged.
- **Language-model direction (approved 2026-11-15, modified D-LM-1):** no spending; study at 10.6M and 24M parameters with B = 1.5 x A, 70M only as a final confirmation run; first a no-penalty resolution check. Remaining D-LM-2..16 are still proposals: [proposals/README.md](proposals/README.md) (D-LM-1..16, which proposes dropping the literal 2 GB vs 3 GB test for a ratio-preserving ladder). Pipeline code `lm/` built and tested, placeholders only.

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

1. **User decisions (2026-11-15):** approve, change or reject the Stage 2 proposals D-S2-1..10 and the language-model proposals D-LM-1..16 (in particular the ratio-preserving ladder instead of 2 GB vs 3 GB; a spending ceiling, Q-HW-2); collect the Kaggle G-E results and re-check the width-63 baseline; commit the new files.
2. Still pending: D-09 approval, restated question wording (research_question.md §5), Q-HW-1/2, Q-SCOPE-1/2. (Items 1 and 3-5 below are older text; Stage 0 and Stage 1 above are done.)
3. Revise research_gap_analysis.md and feasibility_assessment.md for the "more superposition" goal (banners mark the outdated parts).
4. Close Phase 1: full-text reads of ★ sources (start with SUP_2026_004, the likeliest overlap) and the Q-LIT-2 prior-art searches.
5. Phase 2: pre-register hypotheses; then Phase 3 architecture decisions in §9.1 format.
