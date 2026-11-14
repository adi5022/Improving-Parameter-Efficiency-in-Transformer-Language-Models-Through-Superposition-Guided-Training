# Representation-Aware and Interpretability-Aware Training

Session 1 (2026-11-11). Records: [../paper_library.md](../paper_library.md).

## 1. Four families already in the literature

**(a) Activation-statistic regularisers (no interpretability tool needed).**
- DecPO (RAT_2024_001, EMNLP 2024, level P): L = L_DPO + λ‖zzᵀ − I‖²_F, z = W_up·h (MLP up-projection), λ = 1e-4. Llama-2-7B / Llama-3-8B preference fine-tuning; improves GPT-3.5-judged scores in most but not all cells (e.g. Llama-3 toxicity: L1-reg 58.0 > DecPO 57.0). Best layer differs by task. No variance reported in what was read.
- MEmeL (RAT_2024_002) and L2E (RAT_2025_003) push the **other way**: penalise neurons with high online "monosemantic scale" ϕ(z) = (z − z̄)²/S². L2E applies it during Pythia pretraining (70M–2.8B, 10% of schedule) and reports ≈ +2 points 0-shot on ARC-C/PIQA/SciQ, **single runs, no loss reported**.
- The classical ancestor is DeCov (Cogswell et al., ICLR 2016 — **not yet examined**), a covariance penalty to reduce overfitting. A decorrelation loss labelled "monosemanticity" is mathematically this family.

**Reading:** two groups publish opposite-sign claims (encourage vs inhibit monosemanticity → better performance), each with small effects and weak statistics. The honest conclusion is that **neither sign is established**, which is itself a gap a careful, seeded, compute-matched study could address.

**(b) Interpretable-by-design architectures trained from scratch.**
- SoLU (SUP_2022_003): more explainable neurons, ≈ equal loss, hidden features.
- Weight-sparse transformers (RAT_2025_004, OpenAI 2025, level P): ~16× smaller circuits at matched loss; **100–1,000× compute** for equal capability.
- MoE-X (RAT_2025_005, ICML 2025, level P): better chess interpretability; language perplexity better than (non-matched) GPT-2, mixed vs a Switch MoE baseline.
- Monet (RAT_2024_006, read at level P 2026-11-14): average accuracy is **below** a size-matched dense LLaMA at every scale in its Table 2 (0-shot 0.453/0.478/0.511 vs 0.463/0.484/0.520; no perplexity reported), so "without degrading general performance" is a small-cost claim, not parity. Codebook features (RAT_2023_007): interpretability at "modest" capability cost (abstract level, not yet read).

**Reading:** the consistent finding across serious efforts is *interpretability at roughly equal or lower capability*. None demonstrates a capability gain at matched parameters, data and compute.

**(c) Interpretability signals inside the training loop (fine-tuning).**
- CAFT (RAT_2025_009, level P): project out human-selected PCA/SAE directions during fine-tuning; emergent misalignment 7.0% → 0.51% (Qwen-32B, PCA) with MMLU/GSM8K unchanged; random directions do nothing; PCA often beat SAEs.
- Persona vectors / preventative steering (RAT_2025_010, abstract): adding a trait direction during fine-tuning prevents the model from learning to move along it.
- LoRA-for-SAEs (SAE_2025_002): train the LM so a frozen SAE fits it better.

**Reading:** this is the strongest evidence that "interpretability can influence how networks learn" (the README's §9 aspiration) — **already demonstrated**, for steering *what* is learned or *how it generalises*, not for raw capability per parameter.

**(d) Gradient-level control of where representations form (pretraining).**
- Gradient routing (RAT_2024_008, level P): mask gradients by data so chosen concepts land in chosen subspaces; demonstrated on a 28M TinyStories model and 0.3–0.7B models; retain performance falls with routed fraction.

## 2. Mapping the README's proposed mechanisms onto existing work

| README §3–4 mechanism | Existing closest work | Status |
|---|---|---|
| Reduce interference between concepts | SUP_2026_004 orthogonality; DecPO decorrelation; DeCov | Exists (fine-tuning / unclear stage); pretraining from scratch with SAE features not found |
| More consistent, recoverable features | SAE-friendly training (SAE_2025_002); Monet; MoE-X | Exists |
| Better separation where interference is harmful | CAFT (ablate directions); gradient routing | Exists, for targeted concepts |
| Feature-level feedback from SAEs into loss | SAE_2025_002 (frozen SAE); CAFT (SAE directions, human-selected) | Partly exists; **online, automatic, from-scratch version not found** |
| Activation constraints / sparsity | TopK/ReLU activation sparsity (RAT_2025_004 AbsTopK; MoE-X ReLU experts); L1 baselines in DecPO | Exists |
| "Higher capacity per parameter" | SUP_2025_002 suggests LMs are already in strong superposition | **Theoretically doubtful**; no positive evidence found |

## 3. Implications for the experimental design

1. Every "MI-guided" loss must be paired with its **non-MI counterpart of equal cost**: SAE-feature decorrelation vs raw-activation decorrelation (DeCov/DecPO style); SAE-direction projection vs PCA/random projection (CAFT's own baselines); SAE-sparsity pressure vs activation L1 / TopK.
2. **Weight decay and dropout** must be tuned for the baseline before any regulariser is claimed to help (SUP_2025_002 shows weight decay already controls superposition).
3. Expect the capability effect to be small or negative in the compute-limited, single-epoch regime; the plausible positive regimes are data-limited multi-epoch training (overfitting) and quantisation robustness.

## 4. Open items

- RAT_2024_006 Monet done (level P, 2026-11-14). Still open: full read of RAT_2025_004 capability numbers; RAT_2023_007 performance table.
- Packing-type regulariser precedent: RAT_2018_011 (minimum hyperspherical energy, level F partial) penalises normalised weight vectors and is evidenced only on vision classifiers; SUP_2026_004's λ‖tril(DᵀD)‖² on an SAE decoder is the same frame-potential form but is aimed at intervenability, not capacity.
- Examine DeCov (1511.06068) to fix the classical baseline.
- Search once more for "SAE regulariser pretraining from scratch" under alternative terms.
