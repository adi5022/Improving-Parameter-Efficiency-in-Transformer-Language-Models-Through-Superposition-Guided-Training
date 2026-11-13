# Hypotheses

Status: **CANDIDATES, NOT APPROVED.** To be refined into pre-registered hypotheses with numeric gates in Phase 2. Each states which mechanism it relies on, because at the optimum of the LM loss an added regulariser cannot lower the training LM loss (papers/superposition.md §2.3): gains must come from **(O)** optimisation dynamics, **(G)** generalisation, or **(R)** properties the LM loss does not measure.

| ID | Hypothesis (candidate) | Mechanism | Prior expectation from literature | Key controls |
|---|---|---|---|---|
| H1 | In single-epoch, compute-limited training, an SAE-derived auxiliary loss lowers held-out loss by at least δ vs a tuned baseline at equal FLOPs | O | Low (SUP_2025_002; RAT_2025_004/005; SUP_2022_003) | Tuned LR/weight decay; FLOPs matched incl. SAE overhead; ≥ 3 seeds; equivalence test so a null is informative |
| H2 | In multi-epoch, data-limited training, an SAE-derived auxiliary loss reduces the train/held-out gap and lowers held-out loss more than equal-strength dropout, weight decay, or activation decorrelation | G | Uncertain — plausible because regularisers act through generalisation; untested for SAE losses | Non-MI regularisers tuned to the same budget |
| H3 | Any benefit in H1/H2 is larger with a real online SAE than with a random or PCA dictionary of the same size in the same loss | specificity | Uncertain; FeatureScalpel found SAE > random basis but random basis strong (95% vs 81%) | Random dictionary, PCA dictionary |
| H4 | Models trained with the auxiliary loss lose less held-out performance and keep more SAE features under INT8/INT4 PTQ than baseline, and at least match an outlier-suppression baseline | R | Plausible (QNT_2023_001, QNT_2025_002, QNT_2026_003) | Outlier-suppression baseline (e.g. clipped softmax or single-scale RMSNorm); same PTQ method |
| H5 | Across all runs, representation metrics measured mid-training (interference statistics, SAE fidelity, feature stability) predict final held-out loss beyond what training loss at the same step predicts | diagnostic | Unknown | Partial correlation controlling for current loss; random-init metric baselines |
| H6 | The auxiliary loss improves the trained model's SAE quality (fidelity at fixed L0, seed stability, absorption) | R | High (SAE_2025_002) — expected, not a contribution alone | Must be reported together with H1/H2/H4 outcomes |
| H7 | The sign of the monosemanticity effect: encouraging decorrelation vs inhibiting monosemantic neurons (L2E-style) — which, if either, helps held-out loss in our setting? | O/G | Contested (RAT_2024_001 vs RAT_2025_003) | Both signs implemented with equal tuning; seeds |

## Revised after the user's clarification (2026-11-11) — the central hypotheses

The goal is **more** superposition (research_question.md §5). H1–H7 above remain as secondary/diagnostic hypotheses; the primary ones are:

| ID | Hypothesis (candidate) | Mechanism | Prior expectation | Key controls |
|---|---|---|---|---|
| HS1 (toy) | In the toy model of superposition, an in-loop penalty driven by an SAE's estimate of superposition makes a model of width m represent more ground-truth features and reach lower loss than standard training at width m | O (pack features the optimiser has not yet packed) | Uncertain; SUP_2025_002 suggests the coefficient is movable | Standard training; weight decay sweep incl. negative (SUP_2025_002's knob); same penalty from ground-truth features (upper bound); random-dictionary penalty |
| HS2 (toy) | The penalised width-m model matches a standard model of width ≈ 1.5·m | O | Low–uncertain | Width sweep of standard models gives the "equivalent width" |
| HS3 (LM) | A small LM trained with the penalty matches the held-out loss of a ≈ 1.5× larger standard LM trained on the same data | O / G | Low | Tuned baselines; FLOPs reported; ≥ 3 seeds |
| HS4 (quantisation) | After identical quantisation, the gap from HS3 persists (denser packing does not make the model more fragile) | R | Uncertain; denser features may be *more* quantisation-sensitive (QNT_2026_003) | Same PTQ method for both; feature-survival measurement |
| HS5 (measurement validity) | The jointly trained SAE's superposition estimate tracks the true number of represented features (toy) / a fresh independent SAE's estimate (LM) | — | Must be shown, not assumed (SAE_2025_003, SAE_2025_009) | Ground truth (toy); fresh SAEs with other seeds |

Rejected / postponed from README:
- "Substantially better performance per parameter" as a target — not a testable hypothesis as stated.
- KD × MI interaction (postponed; third-order).
- Instruction-following improvements (removed).
