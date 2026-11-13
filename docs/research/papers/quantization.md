# Quantization and Deployment

Session 1 (2026-11-11). Records: [../paper_library.md](../paper_library.md). All entries at level **S** — a full read is required before decisions.

## 1. Findings so far

- **Training-time representation changes improve quantisation — established.**
  - Bondarenko et al. (QNT_2023_001): activation outliers come from attention heads learning to "do nothing" via extreme softmax inputs; clipped softmax and gated attention shrink outliers/kurtosis and improve W8A8 with equal or better floating-point performance.
  - OSP (QNT_2025_002, ACL 2025): Muon + single-scale RMSNorm + learnable embedding projection removes outliers at 1.4B/1T tokens (excess kurtosis 1818.56 → 0.04); W4A4 average 26.5 → 35.7; ~2% overhead; Muon alone insufficient.
- **Quantisation damages interpretable features even when perplexity looks fine** (QNT_2026_003, preprint): INT7 on Gemma-2-2B degrades 18.7% of active SAE features while lowering perplexity; at INT6, 51.3% (Gemma) and 62.4% (Pythia-70M) survive; survival predictable from FP statistics (peak activation strongest; AUC 0.92–0.97).

## 2. What this means for the project

1. The README's question "is a model trained with representation-level constraints more resilient to aggressive quantisation?" is **the best-grounded hypothesis in the proposal**, because the link between training-time activation statistics and quantisation error is already established. Novelty must come from *which* constraint (SAE-derived vs known outlier-suppression methods) and from feature-level survival measurements, not from the link itself.
2. Any quantisation-robustness claim must be compared against OSP-style or clipped-softmax baselines, not only against a plain Adam baseline.
3. "Preservation of learned representations" has a ready-made measurement protocol (QNT_2026_003: frozen FP SAE applied to FP vs quantised activations).
4. Deployment of a ≤ 30M model locally is trivial on CPU; tokens/s and memory at this scale are not research results. Keep deployment measurements minimal.

## 3. Open items

- Read QNT_2023_001, QNT_2025_002, QNT_2026_003 in full.
- Examine LLM.int8(), GPTQ, AWQ, Massive Activations; decide PTQ methods in Phase 6.
