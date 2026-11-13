# Citation Index

Lookup table: claim → source → where it is discussed. Full records in [paper_library.md](paper_library.md).

**Rule:** this index is a lookup system, not a substitute for the source. "Lvl" is the verification level of the source (L local, P full-text via summarising fetch, A abstract only, S snippet only — definitions in paper_library.md). Claims at level A or S must not be used to justify an approved decision until upgraded.

Last updated: 2026-11-13 (session 2).

## Superposition and capacity

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Superposition | Sparse features can be stored in superposition, producing polysemantic neurons; representation undergoes a phase change with sparsity | SUP_2022_001 | arXiv:2209.10652 | A | papers/superposition.md |
| Superposition ↔ scaling | In toy models, strong superposition gives loss ∝ 1/width across many feature-frequency distributions; weak superposition gives a power law only if frequencies are power-law | SUP_2025_002 | arXiv:2505.10465 | P | papers/superposition.md |
| Superposition ↔ scaling | LLMs (OPT, GPT-2, Qwen, Pythia) are in the strong-superposition regime; width exponent ≈ 0.91 ± 0.04 | SUP_2025_002 | arXiv:2505.10465 | P | papers/superposition.md |
| Superposition control | Weight decay (incl. negative) controls the degree of superposition in the toy model | SUP_2025_002 | arXiv:2505.10465 | P | papers/superposition.md |
| Superposition control | Encouraging superposition may change scaling coefficients, not exponents; exponent change likely impossible for natural language | SUP_2025_002 | arXiv:2505.10465 | P | papers/superposition.md |
| Architecture vs polysemanticity | SoLU raises explainable-neuron share ~35% → ~60% with ~equal loss, but may hide features via LayerNorm | SUP_2022_003 | transformer-circuits.pub/2022/solu | P | papers/superposition.md |
| Interference control | Orthogonality regularisation on a feature dictionary bounds interference propagation; enables isolated interventions while preserving performance | SUP_2026_004 | arXiv:2602.04718 | A | papers/superposition.md |

## Sparse autoencoders: reliability and training-time use

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| SAE stability | Only ~30% of SAE features shared across seeds (131k latents, Llama-3-8B); ReLU+L1 more stable than TopK | SAE_2025_003 | arXiv:2501.16615 | A | papers/sparse_autoencoders.md |
| SAE pathology | Feature absorption: sparsity on hierarchical features makes parent features fail to fire; not fixed by size/sparsity changes | SAE_2024_004 | arXiv:2409.14507 | A | papers/sparse_autoencoders.md |
| SAE pathology | Matryoshka SAEs reduce absorption at a small reconstruction cost | SAE_2025_005 | arXiv:2503.17547 | A | papers/sparse_autoencoders.md |
| SAE evaluation | Unsupervised SAE proxy metrics do not reliably predict practical usefulness | SAE_2025_006 | arXiv:2503.09532 | A | papers/sparse_autoencoders.md |
| SAE usefulness | SAEs underperform prompting, fine-tuning and diff-in-means on steering and detection | SAE_2025_007 | arXiv:2501.17148 | A | papers/sparse_autoencoders.md |
| SAE usefulness | SAE probes do not consistently beat simple baselines | SAE_2025_008 | arXiv:2502.16681 | A | papers/sparse_autoencoders.md |
| SAE usefulness | Dense probes beat SAE probes OOD; DeepMind de-prioritised SAE research | SAE_2025_013 | LessWrong post (2025) | S | papers/sparse_autoencoders.md |
| Metric validity | Auto-interp and reconstruction scores of SAEs are similar on random-init and trained transformers | SAE_2025_009 | arXiv:2501.17727 | A | papers/sparse_autoencoders.md |
| SAE training recipe | TopK SAEs + AuxK dead-latent loss (~10% extra compute) | SAE_2024_010 | arXiv:2406.04093 | S | papers/sparse_autoencoders.md |
| SAE ↔ model objective | End-to-end (KL) SAE training gives a Pareto improvement over MSE SAEs | SAE_2024_001 | arXiv:2405.12241 | A | papers/sparse_autoencoders.md |
| Model adapted to SAE | LoRA on the LM around a frozen SAE cuts the spliced CE gap 30–55% without harming general capability | SAE_2025_002 | arXiv:2501.19406 | A | papers/sparse_autoencoders.md |
| Features over training | Continual series of SAEs across checkpoints tracks feature formation and drift | SAE_2024_011 | arXiv:2412.17626 | A | papers/sparse_autoencoders.md |
| Features over training | Most features form around a specific point in pretraining; statistical-learning then feature-learning phases | SAE_2025_012 | arXiv:2509.17196 | A | papers/sparse_autoencoders.md |

## Representation-aware / interpretability-aware training

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Monosemanticity regulariser | Decorrelation penalty ‖zzᵀ−I‖² on MLP activations added to DPO improves alignment scores on Llama-2/3 (fine-tuning only; LLM-judged) | RAT_2024_001 | arXiv:2406.17969 | P | papers/representation_learning.md |
| Monosemanticity (opposite sign) | Suppressing monosemantic neurons gives small gains (BERT/Swin ≤ 88M) | RAT_2024_002 | arXiv:2312.11560 | P | papers/representation_learning.md |
| Monosemanticity (opposite sign) | Inhibiting monosemantic neurons during Pythia pretraining gives ~+2 points 0-shot; single runs, no seeds, no loss reported | RAT_2025_003 | arXiv:2503.23298 | P | papers/representation_learning.md |
| Sign of effect unsettled | Published work claims both that encouraging and that inhibiting monosemanticity improves performance | RAT_2024_001, RAT_2024_002, RAT_2025_003 | — | P | research_gap_analysis.md |
| Interpretability–capability trade-off | Weight sparsity yields ~16× smaller circuits at matched loss but costs 100–1000× compute for equal capability | RAT_2025_004 | arXiv:2511.13653 | P | papers/representation_learning.md |
| Interpretable-by-design MoE | MoE-X: better interpretability than dense/SAE on chess; perplexity mixed vs MoE baseline; dense comparison not parameter-matched | RAT_2025_005 | arXiv:2503.07639 (ICML 2025) | P | papers/representation_learning.md |
| Interpretable-by-design MoE | Monet: sparse dictionary learning inside MoE pretraining, 262k experts/layer | RAT_2024_006 | arXiv:2412.04139 | A | papers/representation_learning.md |
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
| On-policy KD | Reverse-KL on-policy KD improves generative students (120M–13B) | KD_2024_002 | arXiv:2306.08543 | A | papers/knowledge_distillation.md |
| KD in practice | Gemma 2 2B/9B pretrained with KD instead of next-token prediction | KD_2024_003 | arXiv:2408.00118 | A | papers/knowledge_distillation.md |
| Proposed teacher | gpt-oss-120b: 116.8B total / 5.1B active; MXFP4 fits one 80 GB GPU; 201,088-token vocabulary | KD_2025_004 | OpenAI model card 2025 | S | papers/knowledge_distillation.md |

## Quantization

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Training ↔ quantisation | Clipped softmax / gated attention reduce outliers and improve W8A8 with equal or better FP performance | QNT_2023_001 | arXiv:2306.12929 | S | papers/quantization.md |
| Training ↔ quantisation | Outlier-safe pretraining (Muon + single-scale RMSNorm + embedding projection) raises W4A4 avg 26.5 → 35.7 at 1.4B; kurtosis 1818 → 0.04 | QNT_2025_002 | arXiv:2506.19697 | S | papers/quantization.md |
| Quantisation ↔ features | Quantisation degrades SAE features even when perplexity is unchanged (INT7: 18.7% of active features degraded on Gemma-2-2B); survival predictable (AUC 0.92–0.97) | QNT_2026_003 | arXiv:2606.03002 | S | papers/quantization.md |

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
| Packing regulariser | Spreading neuron directions uniformly (minimum hyperspherical energy) improves generalisation in vision tasks | RAT_2018_011 | arXiv:1805.09298 | A | packing_ideas.md |
| Architecture | Unit-norm transformer reports 4-20x fewer training steps | LMT_2025_006 | arXiv:2410.01131 | A | packing_ideas.md |
| Knowledge capacity | ~2 bits/parameter, holds under int8; domain-name prefixes raise capacity | LMT_2024_005 | arXiv:2404.05405 | A | packing_ideas.md |
| Knowledge capacity | ~3.64 bits/parameter for GPT models in half precision | LMT_2025_007 | arXiv:2505.24832 | A | packing_ideas.md |

## Added 2026-11-14 (hierarchy)

| Topic | Claim or finding | Key | Identifier | Lvl | File |
|---|---|---|---|---|---|
| Hierarchy in LMs | Parent-child concepts are orthogonal (child-minus-parent perpendicular to parent) in Gemma and LLaMA-3 | RAT_2025_013 | arXiv:2406.01506 | A | packing_ideas.md |
| Hierarchy in LMs | Parent-child relations linearly recoverable in a low-dim subspace | RAT_2026_014 | arXiv:2604.07886 | A | packing_ideas.md |
| Shared codebooks | Shared concept-vector codebooks compress token embeddings to ~0.5% of parameters with >95% task performance (encoder models, post-hoc) | LMT_2025_009 | arXiv:2509.17737 | A | packing_ideas.md |
