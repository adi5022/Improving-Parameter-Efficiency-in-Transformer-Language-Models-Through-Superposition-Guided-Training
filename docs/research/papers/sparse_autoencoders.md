# Sparse Autoencoders: Reliability and Training-Time Use

Session 1 (2026-11-11). Records: [../paper_library.md](../paper_library.md). Most entries here are at level **A** (abstract only).

## 1. Standard formulation (for reference)

For an activation x ∈ ℝ^d, a ReLU SAE computes f = ReLU(W_enc(x − b_dec) + b_enc), x̂ = W_dec f + b_dec, trained on ‖x − x̂‖² + λ‖f‖₁ (or a TopK constraint replacing the L1 term; SAE_2024_010, snippet level). The user's FeatureScalpel project used exactly this ReLU form with the pretrained `gpt2-small-res-jb` release (FS_2026_001).

## 2. Reliability evidence relevant to "SAE signals in the gradient"

| Problem | Evidence | Consequence for an SAE-derived training loss |
|---|---|---|
| Seed instability | ~30% of features shared across seeds for a 131k-latent Llama-3-8B MLP SAE; 42% on Pythia-160M 2^15 latents; **overlap is much higher for small SAEs (32 latents ≈ 80-100%, Fig. 4) and rises with training tokens** (SAE_2025_003, read in full 2026-11-14) | The loss depends on which SAE run you happened to train, but this is mainly a large-SAE problem. A small, long-trained SAE on a toy model is far more stable. Whether a feature *count* is stable was not tested. |
| Wrong L0 → wrong features | A too-low L0 gives latents that mix correlated features and a lower MSE than the true solution; true L0 is known only in toy models (SAE_2025_014, level P) | In the toy model the true number of active features is known, so the SAE's k/L1 can be calibrated against ground truth; with real LMs it cannot. |
| Absorption / splitting | Sparsity penalty favours absorption when features co-occur hierarchically (SAE_2024_004 App. A.2); absorption rises with width and lower L0 (Fig. 9b); Matryoshka cuts first-letter absorption 0.49 → 0.05 at L0 = 40 for ~50% more training time (SAE_2025_005 Sec. 4.3) | Penalties on "redundant" or "co-active" features will partly penalise SAE artefacts, not model structure. Absorption under-counts parent features, splitting over-counts. |
| Proxy metrics mislead | Proxy metrics do not predict usefulness; the claim rests on rankings, no correlation coefficients (SAE_2025_006 Sec. 4.1); absorption/SCR worsen with SAE width for non-hierarchical SAEs | Do not optimise the model toward SAE proxy metrics and call it an improvement. The number of SAE latents is not a model property. |
| Weak downstream performance | SAEs lose to simple baselines on steering, detection, probing (SAE_2025_007, SAE_2025_008, SAE_2025_013) | An SAE-derived signal must beat a cheap non-SAE signal (PCA, diff-in-means, raw-activation decorrelation) to justify its cost. |
| Metric validity | Auto-interp and reconstruction scores similar on random/Step-0 and trained Pythia (SAE_2025_009, read in full 2026-11-14); in toy data SAE recovery works only in a band of L1 coefficients that shifts with n | "More interpretable features" must be measured against random-init and early-checkpoint controls; the SAE sparsity setting must be calibrated against ground truth before any count is trusted. |
| Non-stationarity | Features begin forming around step ~1,000 in Pythia-160M, directions keep drifting afterwards (SAE_2025_012, SAE_2024_011, level P); warm-starting SAE[k] from SAE[k−1] is workable for analysis, but the authors report no dictionary size or numeric quality/cost and call it a study tool, not an engineering framework | An SAE trained at step t is stale at step t + k; any online scheme needs a refresh schedule, and early training has few formed features to act on. Warm-starting is a plausible design but unproven as a gradient-signal source. |

**The user's own evidence points the same way.** In FeatureScalpel, a free residual edit of the same size reached rank 1 in 21/21 failure cells against 1 for the SAE edit; and in the cross-model study the SAE basis gave 95% top-1 against 81% for 200 random directions (FS_2026_001). The SAE was useful but not necessary. This "SAE vs random/dense basis" control should be carried into this project.

## 3. Existing ways SAEs and model training have been coupled

| Direction | Work | What changes | Claimed effect | Lvl |
|---|---|---|---|---|
| SAE adapts to model output | e2e SAE (SAE_2024_001) | SAE only (KL objective) | Lower CE increase at equal L0; 2-3.5× runtime, e2e features not seed-stable | P |
| Model adapts to frozen SAE | LoRA for SAEs (SAE_2025_002, ICML 2025) | LM weights via LoRA, SAE frozen | CE gap of spliced SAE −30–55% in 15M tokens; adapted LM not worse without the SAE; e2e still better on Llama | F |
| Concurrent SAE + LM training (prior art pointer only) | Lai & Heimersheim 2024; Lai & Huang 2024, as described by SAE_2025_002 | LM and SAE trained together from scratch | Improves spliced loss; toy settings only (neither paper read) | — |
| SAE-derived directions ablated during fine-tuning | CAFT (RAT_2025_009) | LM weights; forward pass projects out chosen directions | Controls OOD generalisation; PCA often ≥ SAE | P |
| Dictionary inside the architecture | Monet (RAT_2024_006), MoE-X (RAT_2025_005) | Architecture itself is a sparse dictionary | Interpretability up; capability ≈ baseline | A / P |
| Dictionary orthogonality | SUP_2026_004 (COLM 2026) | SAE decoder penalised, then LM LoRA-adapted around it (fine-tuning) | Isolated interventions improve at the strongest penalty; GSM8K unchanged within noise | F |

**Not found in session-1 searches** (non-exhaustive, see [../research_gap_analysis.md](../research_gap_analysis.md)): pretraining an LM **from random initialisation** with an auxiliary loss computed from an **online-trained SAE on its own activations**, evaluated for held-out loss against a compute-matched, tuned baseline and against non-SAE regularisers.

## 4. Engineering facts needed for feasibility (to verify in Phase 4)

- Cost of an online SAE at one layer: encoder + decoder matmuls d × (k·d) per token, plus its own backward pass. With expansion k = 8 at d = 256, that is ≈ 2·(256 × 2048) ≈ 1M MACs per token forward — comparable to one MLP block of the base model. Training it every step at one layer ≈ +1 block of compute; every N steps on cached activations is cheaper. **Estimate only; must be measured.**
- Dead latents and TopK/AuxK recipes (SAE_2024_010) — to read.
- SAELens supports training SAEs on custom TransformerLens-compatible models; the user already uses SAELens 6.x and TransformerLens 3.x (FS_2026_001 requirements). Compatibility with a custom from-scratch model needs a check.

## 5. Open items

- Done 2026-11-14: SAE_2025_002, SAE_2025_003, SAE_2025_009 read in full (F); SAE_2024_001, SAE_2024_010, SAE_2024_004, SAE_2025_005, SAE_2025_006, SAE_2024_011, SAE_2025_012 read at P; new SAE_2025_014. Still at A/S: SAE_2025_007 (AxBench), SAE_2025_008 (sparse probing), SAE_2025_013 (blog).
- Engineering note from the full reads: TopK SAEs need AuxK at scale (10% extra compute, SAE_2024_010); the e2e variant (SAE_2024_001) is the least seed-stable; absorption/width interactions (SAE_2025_006) mean latent counts grow with SAE size independently of the model.
- Determine whether any 2025–2026 work trains SAE-regularised models from scratch (one more targeted search with alternative terms: "dictionary learning regulariser pretraining", "sparse coding auxiliary loss transformer pretraining").
