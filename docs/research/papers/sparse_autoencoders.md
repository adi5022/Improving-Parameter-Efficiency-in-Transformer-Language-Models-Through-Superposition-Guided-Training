# Sparse Autoencoders: Reliability and Training-Time Use

Session 1 (2026-11-11). Records: [../paper_library.md](../paper_library.md). Most entries here are at level **A** (abstract only).

## 1. Standard formulation (for reference)

For an activation x ∈ ℝ^d, a ReLU SAE computes f = ReLU(W_enc(x − b_dec) + b_enc), x̂ = W_dec f + b_dec, trained on ‖x − x̂‖² + λ‖f‖₁ (or a TopK constraint replacing the L1 term; SAE_2024_010, snippet level). The user's FeatureScalpel project used exactly this ReLU form with the pretrained `gpt2-small-res-jb` release (FS_2026_001).

## 2. Reliability evidence relevant to "SAE signals in the gradient"

| Problem | Evidence | Consequence for an SAE-derived training loss |
|---|---|---|
| Seed instability | ~30% of features shared across seeds (SAE_2025_003) | The loss depends on which SAE run you happened to train. Needs ensembles or a seed-robust formulation. |
| Absorption / splitting | Parent features absorbed by children; not fixed by size or sparsity (SAE_2024_004); Matryoshka reduces it (SAE_2025_005) | Penalties on "redundant" or "co-active" features will partly penalise SAE artefacts, not model structure. |
| Proxy metrics mislead | Proxy metrics do not predict usefulness (SAE_2025_006) | Do not optimise the model toward SAE proxy metrics and call it an improvement. |
| Weak downstream performance | SAEs lose to simple baselines on steering, detection, probing (SAE_2025_007, SAE_2025_008, SAE_2025_013) | An SAE-derived signal must beat a cheap non-SAE signal (PCA, diff-in-means, raw-activation decorrelation) to justify its cost. |
| Metric validity | Auto-interp scores similar on random and trained transformers (SAE_2025_009) | "More interpretable features" must be measured against random-init and early-checkpoint controls. |
| Non-stationarity | Features form at specific points in pretraining (SAE_2025_012); SAEs must be re-fitted along checkpoints (SAE_2024_011) | An SAE trained at step t is stale at step t + k; any online scheme needs a refresh schedule, and early training has few formed features to act on. |

**The user's own evidence points the same way.** In FeatureScalpel, a free residual edit of the same size reached rank 1 in 21/21 failure cells against 1 for the SAE edit; and in the cross-model study the SAE basis gave 95% top-1 against 81% for 200 random directions (FS_2026_001). The SAE was useful but not necessary. This "SAE vs random/dense basis" control should be carried into this project.

## 3. Existing ways SAEs and model training have been coupled

| Direction | Work | What changes | Claimed effect | Lvl |
|---|---|---|---|---|
| SAE adapts to model output | e2e SAE (SAE_2024_001) | SAE only (KL objective) | Better SAEs | A |
| Model adapts to frozen SAE | LoRA for SAEs (SAE_2025_002) | LM weights via LoRA, SAE frozen | CE gap of spliced SAE −30–55%; no general harm | A |
| SAE-derived directions ablated during fine-tuning | CAFT (RAT_2025_009) | LM weights; forward pass projects out chosen directions | Controls OOD generalisation; PCA often ≥ SAE | P |
| Dictionary inside the architecture | Monet (RAT_2024_006), MoE-X (RAT_2025_005) | Architecture itself is a sparse dictionary | Interpretability up; capability ≈ baseline | A / P |
| Dictionary orthogonality | SUP_2026_004 | Unclear (abstract only) | Isolated interventions, performance preserved | A |

**Not found in session-1 searches** (non-exhaustive, see [../research_gap_analysis.md](../research_gap_analysis.md)): pretraining an LM **from random initialisation** with an auxiliary loss computed from an **online-trained SAE on its own activations**, evaluated for held-out loss against a compute-matched, tuned baseline and against non-SAE regularisers.

## 4. Engineering facts needed for feasibility (to verify in Phase 4)

- Cost of an online SAE at one layer: encoder + decoder matmuls d × (k·d) per token, plus its own backward pass. With expansion k = 8 at d = 256, that is ≈ 2·(256 × 2048) ≈ 1M MACs per token forward — comparable to one MLP block of the base model. Training it every step at one layer ≈ +1 block of compute; every N steps on cached activations is cheaper. **Estimate only; must be measured.**
- Dead latents and TopK/AuxK recipes (SAE_2024_010) — to read.
- SAELens supports training SAEs on custom TransformerLens-compatible models; the user already uses SAELens 6.x and TransformerLens 3.x (FS_2026_001 requirements). Compatibility with a custom from-scratch model needs a check.

## 5. Open items

- Upgrade SAE_2025_002, SAE_2024_001, SAE_2025_003, SAE_2025_009, SAE_2024_010 from A/S to P.
- Determine whether any 2025–2026 work trains SAE-regularised models from scratch (one more targeted search with alternative terms: "dictionary learning regulariser pretraining", "sparse coding auxiliary loss transformer pretraining").
