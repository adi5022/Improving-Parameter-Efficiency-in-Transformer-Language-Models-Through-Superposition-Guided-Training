# Mechanistic Interpretability: Methods Relevant to Training-Time Use

Session 1 (2026-11-11). This file covers interpretability *methods* this project might instrument (SAEs are covered separately in [sparse_autoencoders.md](sparse_autoencoders.md)).

## 1. What the user already does well (verified in FeatureScalpel, FS_2026_001, level L)

- Residual-stream hooks (`blocks.L.hook_resid_pre`) through TransformerLens; SAE encode/decode through SAELens.
- Causal ranking of SAE features by single-feature ablation, batched into one forward call (Δp_k(v) = P_ablate(k)(v) − P_clean(v)).
- Delta patching (add only the decoded change, not the reconstruction error).
- Gradient-based optimisation of intervention parameters (per-feature multipliers m_k = 1 + a_k, a_k = −1 + 3σ(z_k), Adam, hinge-margin + restricted-KL + size loss) with the model frozen.
- Linear and MLP "translators" between residual streams of two models; output-matching training of a map.
- Controls: random target word of matched rank, random vector, another record's edit, random-direction basis; exact McNemar, Wilcoxon, bootstrap; pass bars fixed before runs.

These transfer directly: measuring features, attributing outputs to features, and optimising through hooks are the same operations an interpretability-guided training loop needs. **What changes is that the model's weights will now receive the gradients, not an edit vector.**

## 2. Methods likely needed and their status in the literature

| Method | Use in this project | Evidence status |
|---|---|---|
| SAE on own model at checkpoints | Measure feature formation, interference, stability (Phase 4) | Established (SAE_2024_011, SAE_2025_012) |
| Crosscoders across checkpoints | Track features through training | Established (SAE_2025_012, ICLR 2026) |
| Attribution / ablation of features | Identify "functionally important" vs "redundant" features | Established for analysis (user's own work; Marks et al. not yet examined) |
| Projection-ablation during training | Steer what is learned (CAFT) | Established in fine-tuning (RAT_2025_009) |
| Gradient masking | Localise representations (gradient routing) | Established at small scale (RAT_2024_008) |
| Linear probes / diff-in-means | Cheap non-SAE baseline for every SAE-based signal | Established as strong baselines (SAE_2025_007, SAE_2025_008) |

## 3. Cautions

- Random-init controls for any interpretability score (SAE_2025_009).
- Basis-dependence of neuron-level metrics (SoLU, SUP_2022_003).
- Interpretability improvements are not capability improvements (all of [representation_learning.md](representation_learning.md) §1b).

## 4. Open items

- Read Marks et al. (Sparse Feature Circuits) and transcoder/crosscoder papers before Phase 4 decisions on which activations to instrument.
