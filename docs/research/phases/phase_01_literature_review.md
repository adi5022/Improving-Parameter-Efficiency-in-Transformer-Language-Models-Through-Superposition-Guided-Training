# Phase 1 — Literature Review and Research Gap Identification

Status: **first pass complete; NOT complete** (session 1, 2026-11-11). Completion requires Q-LIT-1 (full-text verification of ★ sources) and Q-LIT-2 (prior-art check for G1).

## Objectives
Establish what exists across CLAUDE.md §4.1 areas A–G; identify a defensible gap; assess feasibility; recommend a direction.

## Questions investigated
1. Has MI been integrated into training? Have SAE-derived features modified optimisation?
2. Is superposition an inefficiency that can be optimised for capacity?
3. Do interpretability-promoting objectives improve capability?
4. Are SAE features reliable enough for training signals?
5. When does KD help small students; is the proposed teacher suitable?
6. Do training-time representation changes affect quantisation?

## Papers examined
See paper_library.md (38 recorded; 10 at level P, 20 at A, 8 at S) and the not-yet-examined list.

## Findings
Seven conclusions C1–C7 in [../literature_review.md](../literature_review.md) §1. Gap analysis in [../research_gap_analysis.md](../research_gap_analysis.md).

## Important technical details
- Toy model and weak/strong superposition scaling: papers/superposition.md §1.
- DecPO regulariser ‖zzᵀ − I‖²_F, λ = 1e-4: papers/representation_learning.md §1a.
- L2E monosemantic scale ϕ(z) = (z − z̄)²/S²: same file.
- Distillation capacity gap and compute thresholds: papers/knowledge_distillation.md §1.
- Regularisers cannot lower the LM training loss at its optimum; gains must come from optimisation, generalisation or unmeasured properties: papers/superposition.md §2.3.

## Conflicting findings
- Encourage (RAT_2024_001) vs inhibit (RAT_2024_002, RAT_2025_003) monosemanticity — both claim gains.
- MoE-X "better than GPT-2" vs mixed against a matched MoE baseline (RAT_2025_005).

## Limitations of this phase so far
- No PDF read page-by-page; level-P extractions went through a summarising tool (one error caught and corrected).
- Searches not exhaustive; 2026 preprints may be missed.
- Language-model-training and quantisation areas covered mainly at A/S level.

## Gaps identified
G1–G5 (research_gap_analysis.md §3).

## Decisions proposed
D-01 … D-07 (architectural_decisions.md). **Decisions approved: none.**

## Outstanding
Q-LIT-1 … Q-LIT-4; user questions Q-HW-1/2, Q-SCOPE-1/2, Q-BG-1.

## References
paper_library.md; citation_index.md.
