# Quantization and Deployment

Session 1 (2026-11-11); upgraded to level **P** (HTML full text via summariser) for the three core papers on 2026-11-14 (Q-LIT-1). Records: [../paper_library.md](../paper_library.md). Numbers below were extracted by a summarising tool; spot-check against the PDFs before quoting in a paper. Post-training quantisation methods (LLM.int8, GPTQ, AWQ, Massive Activations) are still unexamined.

## 1. Findings so far

- **Training-time representation changes improve quantisation — established, but at the scales and bit-widths shown.**
  - Bondarenko et al. (QNT_2023_001, level P): activation outliers come from attention heads learning to "do nothing" via extreme softmax inputs (observational evidence, Sec. 3, Fig. 1-3); clipped softmax and gated attention shrink outliers/kurtosis and fix W8A8 on BERT-base, OPT-125m and ViT-S. Gated attention is good on all three and scales to OPT-1.3B (W8A8 ppl 29.95 vs 989.6 vanilla, Table 3). **Clipped softmax failed on OPT-125m** (FP 16.29 vs 15.84; W8A8 37.2 vs 21.2, Table 2). Only 8-bit PTQ in the main text; ≤ 1.3B.
  - OSP (QNT_2025_002, level P): Muon + single-scale RMSNorm + learnable embedding projection at 1.4B. In the 100B-token ablation kurtosis falls 1818.56 → 0.04 and W4A4KV4 average 26.8 → 37.5 (Table 2); at 1T tokens "4-bit" average 26.5 → 35.7 (Table 3). Muon alone is not enough (29.0), and **SSNorm contributes most of the gain** (Muon + SSNorm 36.4). Full-precision quality is slightly lower at 1T (45.9 → 44.4). One run per setting, no seeds, no sub-1B models; Table 2 was garbled in the source HTML.
- **Quantisation damages interpretable features even when perplexity looks fine** (QNT_2026_003, preprint, level P, title on the v2 page: "Perplexity Can Miss SAE Feature Damage Under Quantization"): weight-only RTN. Gemma-2-2B INT7: 18.74% of active SAE features are "degraded" (correlation 0.5-0.9, none damaged) while perplexity falls ~5.7%; at INT6, 51.3% (Gemma) and 62.4% (Pythia-70M) survive; survival predictable from FP statistics (peak activation strongest; AUC 0.924 / 0.971). Single deterministic runs, public SAEs, internal inconsistencies in its appendices.

## 2. What this means for the project

1. The README's question "is a model trained with representation-level constraints more resilient to aggressive quantisation?" is **the best-grounded hypothesis in the proposal**, because the link between training-time activation statistics and quantisation error is already established. Novelty must come from *which* constraint (SAE-derived vs known outlier-suppression methods) and from feature-level survival measurements, not from the link itself.
2. Any quantisation-robustness claim must be compared against OSP-style or clipped-softmax baselines, not only against a plain Adam baseline.
3. "Preservation of learned representations" has a ready-made measurement protocol (QNT_2026_003: frozen FP SAE applied to FP vs quantised activations).
4. Deployment of a ≤ 30M model locally is trivial on CPU; tokens/s and memory at this scale are not research results. Keep deployment measurements minimal.

## 3. Open items

- QNT_2023_001, QNT_2025_002, QNT_2026_003 read at level P on 2026-11-14 (not PDF-level; Bondarenko Appendix C/B.7 and OSP Table 2 need a PDF spot-check).
- Examine LLM.int8(), GPTQ, AWQ, Massive Activations; decide PTQ methods in Phase 6.
