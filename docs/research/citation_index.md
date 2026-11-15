# Citation Index

Lookup table: claim → source → where it is discussed. Full records in [paper_library.md](paper_library.md).

**Rule:** this index is a lookup system, not a substitute for the source. "Lvl" is the verification level of the source (L local, F PDF pages read page by page, P full-text via summarising fetch, A abstract only, S snippet only — definitions in paper_library.md). Rows updated in the 2026-11-14 Q-LIT-1 pass state the supporting section/figure/table inside the claim text. Claims at level A or S must not be used to justify an approved decision until upgraded.

Last updated: 2026-11-15 (session 2; proposals merged at the end).

## Superposition and capacity

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Superposition | Sparse features can be stored in superposition, producing polysemantic neurons; representation undergoes a phase change with sparsity (verified for the ReLU-output toy model: "Superposition as a Phase Change", n=2,m=1 and n=3,m=2 phase diagrams; pp. 3-17 of the arXiv PDF only) | SUP_2022_001 | arXiv:2209.10652 | F (partial) | papers/superposition.md |
| Superposition | A linear readout never gives superposition (learns top-m PCA-like features); the ReLU/bias readout does, loss ≈ feature benefit + interference; number of features learned ≈ ‖W‖²_F and dimensions per feature D* = m/‖W‖²_F sticky at 1 and 1/2 (Basic Results, Mathematical Understanding, Uniform Superposition) | SUP_2022_001 | arXiv:2209.10652 | F (partial) | papers/superposition.md |
| Capacity theory | Optimal capacity allocation: importance above λ gets a full dimension, importance in (2λ/(k−1), λ) partial capacity, the rest none; partial region widens with kurtosis (sparsity). Derived for a quadratic-activation model; ReLU/GeLU only compared numerically (Sec. 3.4-3.7, Eq. 15-18, Fig. 7) | SUP_2022_005 | arXiv:2210.01892 | P | papers/superposition.md |
| Steering polysemanticity | In toy models the same loss is reached with very different numbers of monosemantic neurons; initial bias and learning rate select the basin (Sec. 4.1, Figs. 3-8) | SUP_2022_008 | arXiv:2211.09169 | P | papers/superposition.md |
| Superposition ↔ scaling | In toy models, strong superposition gives loss ∝ 1/width across many feature-frequency distributions; weak superposition gives a power law only if frequencies are power-law | SUP_2025_002 | arXiv:2505.10465 | P | papers/superposition.md |
| Superposition ↔ scaling | LLMs (OPT, GPT-2, Qwen, Pythia) are in the strong-superposition regime; width exponent ≈ 0.91 ± 0.04 | SUP_2025_002 | arXiv:2505.10465 | P | papers/superposition.md |
| Superposition control | Weight decay (incl. negative) controls the degree of superposition in the toy model | SUP_2025_002 | arXiv:2505.10465 | P | papers/superposition.md |
| Superposition control | Encouraging superposition may change scaling coefficients, not exponents; exponent change likely impossible for natural language | SUP_2025_002 | arXiv:2505.10465 | P | papers/superposition.md |
| Architecture vs polysemanticity | SoLU raises explainable-neuron share ~35% → ~60% with ~equal loss, but may hide features via LayerNorm | SUP_2022_003 | transformer-circuits.pub/2022/solu | P | papers/superposition.md |
| Interference control | Penalty λ‖tril(DᵀD)‖²_F on the **decoder of a TopK SAE** (not the LM's weights) lowers decoder coherence; theory bounds post-intervention mismatch by mean self-coherence (Thm. 4.1, not tight, Fig. 6). The LM is then LoRA-fine-tuned around the frozen SAE (Gemma 2 2B, MetaMathQA, 1 epoch): GSM8K accuracy unchanged within noise (Fig. 3); name-swap success 74.9% (λ=1e-4) vs 62.2% (λ=0) (Fig. 4b); intermediate λ (1e-6, 1e-5) are worse than λ=0 on the interference measures (Table 1). Fine-tuning only, no seeds, mathematical reasoning only | SUP_2026_004 | arXiv:2602.04718 (COLM 2026) | F | papers/superposition.md |
| Interference control (direction) | The orthogonality penalty is aimed at **reducing** feature overlap (intervenability); whether it reduces superposition is listed by the authors as an untested follow-up (Sec. 6) | SUP_2026_004 | arXiv:2602.04718 | F | papers/superposition.md |
| Packing regulariser | Minimum hyperspherical energy: Riesz-energy penalty on normalised neuron weight vectors; vision only (CIFAR, ImageNet ResNet-18/34), e.g. CIFAR-100 CNN-9 error 28.13 → 25.61 (half-space MHE), ImageNet ResNet-18 32.95 → 32.45; weights not activations; not tested on language models (Eq. 1-4, Tables 1, 4, 6) | RAT_2018_011 | arXiv:1805.09298 (NeurIPS 2018) | F (pp. 1-3, 6-8) | packing_ideas.md |

## Sparse autoencoders: reliability and training-time use

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| SAE stability | Only ~30% of SAE features shared across seeds (131k latents, Llama-3-8B MLP); on Pythia-160M 2^15 latents 42% shared (Sec. 3, Fig. 1); latents found in only one of 9 seeds ≈ 35%; overlap **falls with SAE size and k and rises with training tokens and for small SAEs** (Fig. 2, Fig. 4); Gated/L1 SAEs more stable than TopK (Fig. 5). Identity of features was tested, not the count | SAE_2025_003 | arXiv:2501.16615 | F | papers/sparse_autoencoders.md |
| SAE pathology | Feature absorption: the sparsity penalty favours absorption whenever parent and child co-occur (App. A.2, simplified proof); absorption rate rises with SAE width and lower L0 (Fig. 9b); no mitigation tested | SAE_2024_004 | arXiv:2409.14507 | P | papers/sparse_autoencoders.md |
| SAE pathology | Matryoshka SAEs reduce absorption (first-letter absorption 0.05 vs 0.49 at L0 = 40) at a small reconstruction cost (variance explained ~70% vs 72%), ~50% longer training; no gain on Chess-GPT (Sec. 4.3, App. E, G) | SAE_2025_005 | arXiv:2503.17547 | P | papers/sparse_autoencoders.md |
| SAE evaluation | Unsupervised SAE proxy metrics do not reliably predict practical usefulness (rankings and examples, no correlation coefficients reported); absorption/SCR worsen with width for non-hierarchical SAEs (Sec. 4.1-4.2) | SAE_2025_006 | arXiv:2503.09532 | P | papers/sparse_autoencoders.md |
| SAE calibration | Too-low SAE L0 produces latents that mix correlated features and a lower MSE than the true solution; true L0 is known only in toy models (Sec. 3.2-3.3, Thm. F.1-F.2) | SAE_2025_014 | arXiv:2508.16560 | P | papers/sparse_autoencoders.md |
| SAE usefulness | SAEs underperform prompting, fine-tuning and diff-in-means on steering and detection | SAE_2025_007 | arXiv:2501.17148 | A | papers/sparse_autoencoders.md |
| SAE usefulness | SAE probes do not consistently beat simple baselines | SAE_2025_008 | arXiv:2502.16681 | A | papers/sparse_autoencoders.md |
| SAE usefulness | Dense probes beat SAE probes OOD; DeepMind de-prioritised SAE research | SAE_2025_013 | LessWrong post (2025) | S | papers/sparse_autoencoders.md |
| Metric validity | Auto-interp (fuzzing/detection AUROC) and reconstruction scores of TopK SAEs on randomised or Step-0 Pythia models are close to those of trained models and far from a Gaussian-embedding control (Figs. 1-2); gap is larger for small models; authors do not claim SAEs fail on trained models | SAE_2025_009 | arXiv:2501.17727 | F | papers/sparse_autoencoders.md |
| Metric validity (toy) | SAE recovery of ground-truth toy features (mean max cosine ≈ 1) happens only in a "Goldilocks" band of the L1 coefficient that moves with n; random MLPs preserve or increase SAE-measured sparsity of superposed inputs (Sec. 4, Figs. 4-5) | SAE_2025_009 | arXiv:2501.17727 | F | papers/sparse_autoencoders.md |
| SAE training recipe | TopK SAEs + AuxK dead-latent loss: L + α·L_aux, α ≈ 1/32, k_aux ≈ 512, dead = 10M tokens inactive, ~10% extra compute; 16M-latent SAE on GPT-4 ~7% dead (Sec. 2.4, App. A.2, B.1) | SAE_2024_010 | arXiv:2406.04093 | P | papers/sparse_autoencoders.md |
| SAE ↔ model objective | End-to-end (KL) SAE training: at equal L0 CE increase < 45% of local SAE's (GPT-2 small layer 6, L0 50-300); costs 2-3.5× runtime, poorer reconstruction, **e2e features not stable across seeds** (Sec. 3.1, 3.3.2, App. B) | SAE_2024_001 | arXiv:2405.12241 | P | papers/sparse_autoencoders.md |
| Model adapted to SAE | LoRA on the LM around a frozen SAE closes 30–55% of the spliced CE gap (Fig. 3) in 15M tokens, 2-20× faster than e2e SAEs (Tables 1-2); the adapted LM without the SAE is not worse than the original (Table 5); e2e SAE reaches lower final CE on Llama; ICML 2025 | SAE_2025_002 | arXiv:2501.19406 | F | papers/sparse_autoencoders.md |
| Joint SAE + LM training (prior art pointer) | SAE_2025_002 states Lai & Heimersheim (2024) and Lai & Huang (2024) trained SAEs and models concurrently, toy settings only; neither read | SAE_2025_002 | arXiv:2501.19406 Sec. 2 | F (for the statement) | papers/sparse_autoencoders.md |
| Features over training | Warm-started SAE series across Pythia checkpoints (SAE[k] from SAE[k-1]) tracks feature formation/drift; dictionary size and numeric quality/cost not reported; authors call it a study tool, not an engineering framework (Sec. 2.3, App. H) | SAE_2024_011 | arXiv:2412.17626 | P | papers/sparse_autoencoders.md |
| Features over training | Crosscoder on Pythia-160M/6.9B: features begin forming around step ~1,000; "two phases" is a rough division/analogy; causal link is attribution+ablation on three simple tasks (Sec. 4.1, 5, 6, 7) | SAE_2025_012 | arXiv:2509.17196 | P | papers/sparse_autoencoders.md |

## Representation-aware / interpretability-aware training

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Monosemanticity regulariser | Decorrelation penalty ‖zzᵀ−I‖² on MLP activations added to DPO improves alignment scores on Llama-2/3 (fine-tuning only; LLM-judged) | RAT_2024_001 | arXiv:2406.17969 | P | papers/representation_learning.md |
| Monosemanticity (opposite sign) | Suppressing monosemantic neurons gives small gains (BERT/Swin ≤ 88M) | RAT_2024_002 | arXiv:2312.11560 | P | papers/representation_learning.md |
| Monosemanticity (opposite sign) | Inhibiting monosemantic neurons during Pythia pretraining gives ~+2 points 0-shot; single runs, no seeds, no loss reported | RAT_2025_003 | arXiv:2503.23298 | P | papers/representation_learning.md |
| Sign of effect unsettled | Published work claims both that encouraging and that inhibiting monosemanticity improves performance | RAT_2024_001, RAT_2024_002, RAT_2025_003 | — | P | research_gap_analysis.md |
| Interpretability–capability trade-off | Weight sparsity yields ~16× smaller circuits at matched loss but costs 100–1000× compute for equal capability | RAT_2025_004 | arXiv:2511.13653 | P | papers/representation_learning.md |
| Interpretable-by-design MoE | MoE-X: better interpretability than dense/SAE on chess; perplexity mixed vs MoE baseline; dense comparison not parameter-matched | RAT_2025_005 | arXiv:2503.07639 (ICML 2025) | P | papers/representation_learning.md |
| Interpretable-by-design MoE | Monet: 512×512 = 262,144 experts/layer; average accuracy **below** the dense LLaMA baseline at all three scales (0-shot 0.453/0.478/0.511 vs 0.463/0.484/0.520), no perplexity reported; monosemanticity shown by examples (Sec. 3, Table 2, Fig. 2) | RAT_2024_006 | arXiv:2412.04139 | P | papers/representation_learning.md |
| Discrete bottleneck | Codebook features: VQ bottleneck in every layer with modest performance loss | RAT_2023_007 | arXiv:2310.17230 | A | papers/representation_learning.md |
| Localising representations | Gradient routing localises capabilities during pretraining; retain performance degrades with routed fraction | RAT_2024_008 | arXiv:2410.04332 | P | papers/representation_learning.md |
| Interp signal in training loop | Ablating interpreted directions during fine-tuning (CAFT) cuts emergent misalignment ~10× with MMLU/GSM8K unchanged; PCA often ≥ SAE | RAT_2025_009 | arXiv:2507.16795 | P | papers/representation_learning.md |
| Interp signal in training loop | Preventative steering with persona vectors during fine-tuning avoids trait shifts | RAT_2025_010 | arXiv:2507.21509 | A | papers/representation_learning.md |

## Language-model training

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Small LMs | Sub-billion: deep-thin, embedding sharing, GQA matter; block-wise weight sharing adds ~0.7–0.8% | LMT_2024_001 | arXiv:2402.14905 | A | papers/language_model_training.md |
| Small LMs | < 10M-parameter models write coherent stories on TinyStories | LMT_2023_002 | arXiv:2305.07759 | A | papers/language_model_training.md |
| Data-limited | BabyLM: architecture and more epochs helped; curriculum learning mostly failed; teacher distillation used by some teams | LMT_2024_003 | arXiv:2412.05149 | S | papers/language_model_training.md |
| Baseline strength | Tuned 124M recipes reach FineWeb val loss 3.28 in minutes on 8×H100; records need p < 0.01 due to seed variance | LMT_2024_004 | github.com/KellerJordan/modded-nanogpt | S | papers/language_model_training.md |

## Knowledge distillation

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Capacity gap | Student loss rises again when the teacher is too strong | KD_2025_001 | arXiv:2502.08606 | P | papers/knowledge_distillation.md |
| When KD wins | KD beats supervised pretraining only below a student-compute threshold and when the teacher already exists | KD_2025_001 | arXiv:2502.08606 | P | papers/knowledge_distillation.md |
| Logit access | Logit KD needs full teacher logits over a shared vocabulary | KD_2025_001 | arXiv:2502.08606 | P | papers/knowledge_distillation.md |
| KD limits of the evidence | Authors' own limitations: language modelling and English C4 only, full-distribution KD only (no sequence-level KD), transformers only, teacher and student on one distribution, λ = 1 and temperature 1 only; students in the read portion are 1B-30B, none below 100M (App. A, B.2-B.3, C) | KD_2025_001 | arXiv:2502.08606 | P | papers/knowledge_distillation.md |
| On-policy KD | Reverse-KL on-policy KD improves generative students (120M–13B) | KD_2024_002 | arXiv:2306.08543 | A | papers/knowledge_distillation.md |
| KD in practice | Gemma 2 2B/9B pretrained with KD instead of next-token prediction | KD_2024_003 | arXiv:2408.00118 | A | papers/knowledge_distillation.md |
| Proposed teacher | gpt-oss-120b: 116.8B total / 5.1B active; MXFP4 fits one 80 GB GPU; 201,088-token vocabulary | KD_2025_004 | OpenAI model card 2025 | S | papers/knowledge_distillation.md |

## Quantization

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Training ↔ quantisation | Gated attention (all of BERT, OPT-125m/350m/1.3B, ViT) and clipped softmax (BERT, ViT) cut outliers and fix W8A8 with equal or better FP quality; **clipped softmax made OPT-125m worse (FP 16.29 vs 15.84; W8A8 37.2 vs 21.2)**; ≤ 1.3B, 8-bit PTQ (Table 2, Table 3, Sec. 6) | QNT_2023_001 | arXiv:2306.12929 | P | papers/quantization.md |
| Training ↔ quantisation | Outlier-safe pretraining at 1.4B: 100B-token ablation W4A4KV4 avg 26.8 (Adam) → 37.5, kurtosis 1818.56 → 0.04 (Table 2); 1T-token "4-bit" avg 26.5 → 35.7 (Table 3); SSNorm adds most of the gain over Muon alone (29.0 → 36.4 with SSNorm); FP16 at 1T slightly lower (45.9 → 44.4, Table 5); one run, no seeds | QNT_2025_002 | arXiv:2506.19697 | P | papers/quantization.md |
| Quantisation ↔ features | Weight-only RTN: Gemma-2-2B INT7 gives 18.74% of active SAE features "degraded" (Pearson 0.5-0.9; 0% damaged) while perplexity falls 5.65%; INT6 survival 51.3% (Gemma) / 62.4% (Pythia-70M); survival predicted from FP stats, AUC 0.924 (Pythia) / 0.971 (Gemma); single deterministic runs, public SAEs (Table 1, Sec. 4.2, Table 2) | QNT_2026_003 | arXiv:2606.03002 | P | papers/quantization.md |

## User's prior work (local evidence)

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| SAE vs free residual edit | A free residual edit of equal size reached rank 1 in 21/21 cells vs 1 for the SAE edit (selected failure prompts) | FS_2026_001 | technical_paper.tex §"How the design was reached" | L | phases/phase_00_project_understanding.md |
| SAE vs random basis | D6 ceiling: SAE directions 95% top-1 vs 200 random directions 81% — the SAE helps but is not required | FS_2026_001 | handoff_2026-10-10_cross_model.md (S1) | L | phases/phase_00_project_understanding.md |
| Weak success metric | Additive edit forces random words to rank 1 on 20/20 prompts — rank-1 is a weak test | FS_2026_001 | technical_paper.tex §"Rank 1 is a weak test" | L | phases/phase_00_project_understanding.md |

## Added 2026-11-13 (packing ideas)

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Superposition geometry | Strongly represented features ~ min(n, m^2/2) (ETF ceiling); Welch bound kappa ~ 1/sqrt(m) | SUP_2025_002 | arXiv:2505.10465 Eq. 5, App. A.1, Fig. 8c/16 | L | packing_ideas.md |
| Superposition geometry | Measured mean squared overlap above the 1/m line at small alpha (~1.5-2.5x, by eye) but scaling as 1/m: headroom in the coefficient | SUP_2025_002 | Fig. 17 | L | packing_ideas.md |
| Packing penalty | Fig. 2c is produced by minimising squared off-diagonals of the normalised W (a frame-potential penalty) | SUP_2025_002 | App. D.2 | L | packing_ideas.md |
| Untested claim | Encouraging superposition could let smaller models match larger ones; cites nGPT and FOCUS; not tested in the paper; exponent fixed for natural language | SUP_2025_002 | p. 10 | L | packing_ideas.md |
| Packing regulariser | Spreading neuron directions uniformly (minimum hyperspherical energy) improves generalisation of vision classifiers (see the F-level row above); not tested on language models | RAT_2018_011 | arXiv:1805.09298 | F (partial) | packing_ideas.md |
| Architecture | Unit-norm transformer: 4×/10×/20× fewer **tokens/iterations** at 1k/4k/8k context, 0.5B and 1B only; time per step +60-80%; only the initial LR tuned; no downstream table in the text; does not discuss superposition (Sec. 3.1, App. A.5, A.7) | LMT_2025_006 | arXiv:2410.01131 | P | packing_ideas.md |
| Knowledge capacity | ~2 bits/parameter is the best case at **1000 exposures** per fact (GPT-2 rotary, 1M-0.5B, synthetic biographies); ≥ 1 bit/parameter at 100 exposures; int8 negligible loss, int4 more than halves capacity (Results 1, 4, 8) | LMT_2024_005 | arXiv:2404.05405 | P | packing_ideas.md |
| Knowledge capacity | Domain-name-like token prefixes recover most of the capacity lost to junk data (100-exposure drop ~20× → ~2×, Result 12); shown only in a 1/8-useful, 7/8-junk mixture | LMT_2024_005 | arXiv:2404.05405 | P | packing_ideas.md |
| Knowledge capacity | ~3.6 bits/parameter (fitted α = 3.64 half precision; Table 1 means 3.83 fp32 / 3.51 bf16) measured on **random uniform tokens** with small GPT-2-style models (≲ 20M params), a lower bound; no capacity number for text (Sec. 3, Table 1, Fig. 6) | LMT_2025_007 | arXiv:2505.24832 | P | packing_ideas.md |

## Added 2026-11-14 (hierarchy)

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Hierarchy in LMs | Parent-child concepts are orthogonal (child-minus-parent perpendicular to parent) in the **final-layer unembedding space** (whitened) of Gemma-2B (LLaMA-3-8B in App. F), 957 WordNet synsets; authors say internal-layer geometry is open (Thm. 8, Sec. 5.1, Fig. 5, Table 1) | RAT_2025_013 | arXiv:2406.01506 | P | packing_ideas.md |
| Optimiser | Muon + decoupled weight decay + update scale 0.2·√max(A,B): fitted ~52% of AdamW FLOPs at 399M-1.5B; AdamW grid-searched, Muon not; single runs; Muon-SFT worse than Adam-SFT on Qwen2.5-7B (Sec. 2.2, 3.2, Table 7) | LMT_2025_010 | arXiv:2502.16982 | P | papers/language_model_training.md |
| Hierarchy in LMs | Parent-child relations linearly recoverable in a low-dim subspace | RAT_2026_014 | arXiv:2604.07886 | A | packing_ideas.md |
| Shared codebooks | Shared concept-vector codebooks compress token embeddings to ~0.5% of parameters with >95% task performance (encoder models, post-hoc) | LMT_2025_009 | arXiv:2509.17737 | A | packing_ideas.md |

## Added 2026-11-14 (Q-LIT-2 prior-art search; details in prior_art_search.md)

Keys in the 101+ block per category are new in this search. Lvl: P = HTML full text via summarising fetch (partial where noted in prior_art_search.md), A = abstract/listing/search summary, S = snippet. Blogs and arXiv preprints are not peer-reviewed.

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Spread-out penalty in LMs | Pairwise-cosine penalty on the tied embedding matrix reduces embedding degeneration; PPL 66.0 to 65.2 (AWD-LSTM WikiText-2), BLEU +0.5 to +1.1 (ICLR 2019) | RAT_2019_101 | arXiv:1907.12009 | P | prior_art_search.md |
| Coherence/ETF penalty | Penalty max-abs(AtA minus ETF Gram) on FC/conv/LSTM weights; modest gains, mainly combined with dropout; slow, large beta hurts (preprint; BMVC 2020 per a search record) | RAT_2018_103 | arXiv:1810.06049 | P | prior_art_search.md |
| Spectrum control | Training output-embedding singular values toward a slow-decay prior improves LM/MT (ICLR 2020) | RAT_2020_102 | ICLR 2020 poster ByxY8CNtvr (OpenReview) | A | prior_art_search.md |
| Hyperspherical energy | CoMHE: naive MHE hard to optimise; compressive version for vision (CVPR 2020) | RAT_2020_104 | arXiv:1906.04892 | A | prior_art_search.md |
| Orthogonality on SAE dictionary | Coherence penalty is applied to the SAE decoder during SAE fine-tuning, then LoRA on the LM; acts on the SAE, not LM weights; GSM8K comparable (answers Q-LIT-4; upgraded to F level in the Q-LIT-1 pass, see the SUP_2026_004 rows in the superposition table above) | SUP_2026_004 | arXiv:2602.04718 | F | prior_art_search.md |
| Orthogonality in a small LM | Neuron-correlation penalty on 30M TinyStories GPT: ~1.5 PPL worse, no monosemanticity gain; single runs; blog | RAT_2026_105 | hisku.substack.com/p/training-for-interpretability-lessons | P | prior_art_search.md |
| Joint SAE + LM training | Char-level GPT-2 (4 layers, width 64): SAE losses added to CE, trained together; better fresh SAEs, slightly lower CE; toy scale; blog | RAT_2025_106 | LessWrong sYFNGRdDQYQrSJAd8; code peterlai/gpt-circuits | P | prior_art_search.md |
| Joint SAE training | Meta-SAE decomposability penalty trained jointly with the SAE only (LM not trained) | RAT_2026_107 | arXiv:2604.03436 | A | prior_art_search.md |
| SAE inside the block | Monosemantic Feature Neurons: concept paper, no experiments | RAT_2025_108 | preprints.org 202510.2064 | A | prior_art_search.md |
| SAE regularisation (fine-tuning) | Frozen SAE feature-space regularisation for LLM continual learning (preprint, search summaries) | RAT_2026_109 | arXiv:2606.26629 | A | prior_art_search.md |
| SAE regularisation (fine-tuning) | SAE-FD: anchoring representations in SAE feature space for continual learning (snippet only) | RAT_2026_110 | arXiv:2605.25525 | S | prior_art_search.md |
| SAE regularisation (fine-tuning) | SAE-FT: penalise added/removed SAE features when fine-tuning CLIP (snippet only) | RAT_2026_111 | arXiv:2605.15961 | S | prior_art_search.md |
| Hierarchy encoders | Hyperbolic clustering + centripetal losses re-train encoder LMs (NeurIPS 2024; snippet) | RAT_2024_112 | arXiv:2401.11374 | A | prior_art_search.md |
| Hyperbolic LLMs | HELM: fully hyperbolic LLMs trained from scratch, up to ~4% over Euclidean (NeurIPS 2025; summary only) | RAT_2025_113 | arXiv:2505.24722 | A | prior_art_search.md |
| Hyperbolic tree regulariser | HypStructure: hyperbolic tree loss for classification features (snippet only) | RAT_2024_114 | arXiv:2412.01023 | S | prior_art_search.md |
| Correlated-feature geometry | Correlated features give constructive interference and clusters in trained superposition autoencoders; more with weight decay; no hierarchy, no regulariser (ICLR 2026 poster listing seen) | SUP_2026_101 | arXiv:2603.09972 | P | prior_art_search.md |
| Coefficients vs exponents | Position paper: exponents fixed by generic mechanisms (incl. superposition: 1/width); coefficients are the target; no method (preprint) | SUP_2026_102 | arXiv:2606.25008 | A | prior_art_search.md |
| Superposition theory | Width phase transition from partial to full representation; non-uniform frequencies delay it (preprint) | SUP_2026_103 | arXiv:2609.36455 | A | prior_art_search.md |
| Training dynamics | Superposition bottleneck gives training power-law exponent ~1, up to ~10x faster (teacher-student; preprint) | SUP_2026_104 | arXiv:2602.01045 | A | prior_art_search.md |
| Superposition geometry | Frame-operator spectrum; capacity saturation gives tight frames (preprint) | SUP_2026_105 | arXiv:2602.02224 | A | prior_art_search.md |
| Superposition theory | Frame theory and compressed sensing; exact recovery threshold for real ETFs (preprint) | SUP_2026_106 | arXiv:2608.27540 | A | prior_art_search.md |
| Superposition hurts? | Toy model: adversarial vulnerability rises with features per dimension (r ~0.99); real-model evidence via SAE proxies (preprint) | SUP_2025_107 | arXiv:2508.17456 | P | prior_art_search.md |
| Superposition metric | SAE-entropy effective-feature count; says superposition-vulnerability sign depends on regime (TMLR listing per abstract page) | SUP_2025_108 | arXiv:2512.13568 | A | prior_art_search.md |
| Geometry vs performance | 108 OLMo-style models: weight decay shapes unembedding effective rank; rank does not reliably predict performance (preprint) | SUP_2026_109 | arXiv:2602.20433 | A | prior_art_search.md |
| Interference vs width | Widening at constant non-zero parameter count reduces feature collisions and raises accuracy (ICML 2026) | SUP_2026_110 | arXiv:2510.04500 | A | prior_art_search.md |
| Interference weights | Anthropic note: interference weights are irrelevant or harmful; no training method (partial read) | SUP_2026_111 | transformer-circuits.pub/2026/interference_effectiveness_helpfulness | A | prior_art_search.md |
| Superposition review | Review of theory and recovery methods (preprint) | SUP_2026_112 | arXiv:2609.06862 | A | prior_art_search.md |
| Optimiser and capacity | One Muon step stores more associations than SGD in a linear associative-memory model (preprint) | SUP_2026_113 | arXiv:2603.26554 | A | prior_art_search.md |
| Forgetting as feature change | Forgetting from shrunken feature capacity or disrupted readout; toy model (CoLLAs 2026) | SUP_2026_114 | arXiv:2601.22012 | A | prior_art_search.md |
| Hierarchy in SAE toys | Hierarchical/correlated toy features (orthogonal directions) cause hedging in narrow SAEs (preprint) | SAE_2025_101 | arXiv:2505.11756 | A | prior_art_search.md |
| Synthetic SAE benchmark | Synthetic data with correlation, hierarchy, superposition (preprint) | SAE_2026_102 | arXiv:2602.14687 | A | prior_art_search.md |
| Rotation quantisation | Rotation removes hidden-state outliers; 4-bit W/A/KV, LLaMA-2-70B within 0.47 PPL (NeurIPS 2024 not confirmed on page) | QNT_2024_101 | arXiv:2404.00456 | A | prior_art_search.md |
| Learned rotations | SpinQuant narrows 4-bit gap to 2.9 zero-shot points on LLaMA-2 7B (ICLR 2025) | QNT_2025_102 | arXiv:2405.16406 | A | prior_art_search.md |
| Quantisation vs data | PTQ degradation grows with tokens per parameter (ICLR 2025 listing seen in search) | QNT_2025_103 | arXiv:2411.04330 | A | prior_art_search.md |
| Quantisation vs training | Low-bit quantisation hurts smaller, heavily trained models more (ACL 2025 listing seen in search) | QNT_2025_104 | arXiv:2411.17691 | A | prior_art_search.md |
| Overtraining | Longer pretraining makes models more sensitive to parameter changes and harder to fine-tune (preprint) | QNT_2025_105 | arXiv:2503.19206 | A | prior_art_search.md |
| Embedding weight decay in practice | OLMo 2 excluded token embeddings from weight decay (norms shrank, spikier training) | LMT_2025_101 | arXiv:2501.00656 | S | prior_art_search.md |

## Added 2026-11-15 from the Stage 2 and language-model proposals (agent-written records; lead spot-checked, not re-verified source by source)

Full records (authors, identifiers, URLs, methods, limits) are in [proposals/stage_2_sources.md](proposals/stage_2_sources.md) (14 new + 12 re-read records) and [proposals/lm_sources.md](proposals/lm_sources.md) (52 records). They are **not yet merged into paper_library.md**. Level "F(text)" in the LM file = downloaded and read in the named sections, not whole papers, figures not viewed (stricter than P, short of F). Author lists marked † in stage_2_sources.md were given from memory and still need checking on the arXiv pages.

### Stage 2: SAE measurement, joint training, ground truth

| Topic | Claim or finding | Key | Lvl | File |
|---|---|---|---|---|
| SAE basics | Sparse autoencoders find interpretable features in LMs | SAE_2023_201 | P | proposals/stage_2_sources.md |
| SAE ground truth | Early MMCS ground-truth recovery measure for SAEs on toy data | SAE_2022_202 | P | proposals/stage_2_sources.md |
| SAE variants | JumpReLU and BatchTopK SAE architectures (reconstruction fidelity vs sparsity) | SAE_2024_203, SAE_2024_204 | P | proposals/stage_2_sources.md |
| SAE inference | Provable amortisation gap between the SAE encoder and optimal sparse inference | SAE_2024_205 | P | proposals/stage_2_sources.md |
| SAE reliability | SAE latents are not canonical units; theory of SAE limits and a reweighted remedy | SAE_2025_206, SAE_2025_207 | P | proposals/stage_2_sources.md |
| SAE ground truth | ToyBench (SAE evaluation with known features across frequency distributions); only the abstract was read | SAE_2026_208 | A | proposals/stage_2_sources.md |
| Gaming a monitor | Models can bypass latent-space monitors when optimised against them (Goodhart risk for SAE-based penalties) | RAT_2024_210 | P | proposals/stage_2_sources.md |
| Gaming a proxy | Reward-model overoptimisation scaling laws (abstract only) | RAT_2022_211 | A | proposals/stage_2_sources.md |
| Coherence bounds | Coherence-based guarantees for sparse recovery (Ben-Haim, Eldar, Elad, IEEE TSP 2010; pages 1-5) | MATH_2010_213 | F | proposals/stage_2_sources.md |
| Welch bound | Welch bound and mutual coherence (secondary pages); the project's measured Welch minima 0.0657 / 0.0390 / 0.0149 (m = 15 / 25 / 63) match the formula (n - m) / (m (n - 1)) | MATH_1974_214 | P | proposals/stage_2_sources.md |

### Language-model training, compute, quantisation

| Topic | Claim | Key | Lvl | File |
|---|---|---|---|---|
| Compute rule | Training is about 6 FLOPs per parameter per token; exact count within 0.99 to 1.10 of 6ND (73M to 6.8B) | LMT_2020_201, LMT_2022_202 (App. F, Table A4) | F(text) partial | proposals/lm_sources.md |
| Tokens per parameter | Compute-optimal about 20 tokens per parameter (400M: 8.0B; 1B: 20.2B) | LMT_2022_202 (Table 3) | F(text) partial | proposals/lm_sources.md |
| Utilisation | Published runs reach 22 to 34 percent (Pythia), 51 (TinyLlama), 55 (llm.c) of A100 peak, computed from reported GPU-hours | LMT_2023_204, LMT_2024_207, LMT_2024_224 | F(text) / P | proposals/lm_sources.md |
| Over-training | SmolLM2-135M trained on 2T tokens (about 14,800 per parameter) | LMT_2025_206 | F(text) partial | proposals/lm_sources.md |
| Embedding share | A 125M-class model has more than 20 percent of its parameters in embeddings; sharing saves 16M parameters for -0.2 accuracy points (45.0 with 2 extra layers) | LMT_2024_001 (Table 1) | F(text) partial | proposals/lm_sources.md |
| Weight decay | No weight decay on embeddings reduced spikes in OLMo 2 (supersedes the S-level entry LMT_2025_101 above) | LMT_2025_101 (Sec. 3.4.2) | F(text) partial | proposals/lm_sources.md |
| Small-model quantisation | OPT-125M WikiText2 perplexity 27.65 (16-bit), 31.12 GPTQ 4-bit, 37.28 RTN 4-bit | QNT_2022_201 (Table 3) | F(text) partial | proposals/lm_sources.md |
| Software state | AutoGPTQ and AutoAWQ archived on GitHub (checked 2026-11-15); bitsandbytes and llama.cpp active | INF_2026_201 | F(text) | proposals/lm_sources.md |
| Free compute | Kaggle docs: 12 h sessions, T4 x2, no weekly quota figure on the page (the "about 30 h/week" figure is from secondary sources, unverified) | INF_2026_203 | F(text) via renderer | proposals/lm_sources.md |
