# Methodology

Status: **DRAFT PRINCIPLES ONLY.** No method is approved. Concrete methodology is written in Phase 2 after decisions D-01 to D-06.

## 1. Principles derived from the literature review

1. **Fair baselines.** Tune learning rate, weight decay and dropout for the baseline with the same budget given to any regulariser's strength sweep (LMT_2024_004 shows seed variance and tuning dominate small differences).
2. **Compute matching.** Count FLOPs of the online SAE and auxiliary loss; give the baseline the same FLOPs (extra steps/tokens) in the main comparison, and also report the equal-steps comparison.
3. **Non-MI controls of equal cost** for every MI-derived signal: raw-activation decorrelation (DeCov/DecPO style), activation L1/TopK, weight decay, random dictionary, PCA dictionary (RAT_2025_009 used random and PCA baselines; FeatureScalpel used random directions).
4. **Interpretability metric controls:** random-init and early-checkpoint SAEs as a floor (SAE_2025_009); ≥ 2 SAE seeds when reporting feature-level metrics (SAE_2025_003).
5. **Seeds and statistics:** ≥ 3 training seeds per arm; paired comparisons where possible; equivalence margins declared in advance so a null is interpretable.
6. **Pre-registration:** gates and analysis rules committed before each run (continuing the user's FeatureScalpel practice).
7. **Outcome hierarchy:** primary = held-out LM loss (and benchmark accuracy if a suite is approved); secondary = generalisation gap, PTQ degradation, feature survival; descriptive = representation metrics. Representation metrics are never primary outcomes.

## 2. Experiment record template (CLAUDE.md §14)

Each experiment gets an ID and a record with: date; git commit; config file; dataset + version + tokenizer; seed; optimiser; LR & schedule; batch size; steps & tokens; loss configuration (aux losses, strengths, refresh schedule); hardware; wall-clock and measured FLOPs; checkpoint path; evaluation method; results; classification (preliminary / controlled / statistically supported / hypothesis).

## 3. To be decided in Phase 2–3

Dataset and regime, model sizes, tokenizer, architecture, SAE architecture and placement, auxiliary-loss formulations, refresh schedule, evaluation suite, PTQ methods. See [architectural_decisions.md](architectural_decisions.md).
