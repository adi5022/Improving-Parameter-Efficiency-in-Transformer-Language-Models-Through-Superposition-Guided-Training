# Paper Library

Full bibliographic records for every source examined so far. Claims are indexed in
[citation_index.md](citation_index.md); topic syntheses are in [papers/](papers/).

Last updated: 2026-11-11 (session 1).

## Verification levels (read this before trusting any entry)

| Code | Meaning |
|---|---|
| **L** | Local copy read directly from disk (user's own work). |
| **P** | Paper HTML full text fetched and read through a summarising fetch tool. Usually the first ~100k characters were covered (introduction, method, experiments, main results); appendices and some limitations sections may be missing. Numbers were extracted by the tool and **must be spot-checked against the PDF before being quoted in a paper.** |
| **A** | Abstract and arXiv/venue metadata only. **Methodology and experimental findings NOT verified.** |
| **S** | Search-result snippets or secondary summaries only. **Treat as unverified pointers.** |

No PDF was read page-by-page in session 1. Every central paper (marked ★) needs a full-text read before it is used to justify an approved decision (tracked in [open_questions.md](open_questions.md), Q-LIT-1).

Peer-review status is taken from the arXiv comments field or the proceedings page. "Not stated" means the arXiv page lists no venue; it does not mean the paper was rejected.

---

## Superposition and representational capacity

### SUP_2022_001 — Toy Models of Superposition
- Authors: N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, et al., C. Olah
- Year / venue: 2022, Anthropic technical report (Transformer Circuits Thread); arXiv preprint
- Peer review: **Not peer-reviewed** (technical report)
- ID: arXiv:2209.10652 — https://arxiv.org/abs/2209.10652 — open access
- Code: none official
- Level: **A**
- Main findings (abstract): polysemanticity can arise when sparse features are stored in superposition; phase change between superposed and non-superposed representations; geometry linked to uniform polytopes; evidence linking superposition to adversarial examples.
- Relevance: the conceptual foundation of the README's §3. Full sections on computation in superposition and phase diagrams still to be read.

### SUP_2025_002 ★ — Superposition Yields Robust Neural Scaling
- Authors: Yizhou Liu, Ziming Liu, Jeff Gore
- Year / venue: 2025, **NeurIPS 2025** (Best Paper Runner-up per arXiv page)
- Peer review: Peer-reviewed
- ID: arXiv:2505.10465 — https://arxiv.org/abs/2505.10465 — open access
- Code: https://github.com/liuyz0/SuperpositionScaling
- Level: **L** (upgraded 2026-11-13: user-supplied PDF `Research References/Superposition Yields Robust Neural Scaling.pdf`, all 37 pages read; the earlier level P came from a summarising fetch)
- Method: Anthropic toy model (h = Wᵀx, y = ReLU(Wh + b), squared error), feature frequencies p_i ∝ i^(−α); weight decay (including *negative* weight decay) used as a knob to control the degree of superposition. LLM analysis of unembedding matrices of OPT, GPT-2, Qwen, Pythia (≈100M–70B).
- Findings: weak superposition → loss is a power law in width m only if feature frequencies are power-law (exponent α−1). Strong superposition → loss ∝ 1/m over broad frequency distributions, because squared overlaps of feature vectors scale ~1/m. LLMs are in the strong regime; fitted width exponent α_m = 0.91 ± 0.04 (0.88 ± 0.06 from Chinchilla sizes).
- Stated implications: changing the width-scaling exponent may be impossible for natural language; encouraging superposition might change coefficients, not exponents; encouraging superposition could make interpretability harder.
- Limitations (authors): toy model not rigorously solved; tokens treated as atomic features; data/steps scaling not studied; squared error vs cross-entropy.
- Relevance: **the strongest single piece of evidence against the README's framing that superposition is an inefficiency to remove.** It implies trained LMs already exploit superposition in the way that drives scaling.
- Full-text additions (2026-11-13, level L): (1) strongly represented features number about min(n, m^2/2), the ETF ceiling (Fig. 8c, 16; App. A.1); (2) the Welch bound kappa ~ 1/sqrt(m) is the best worst-case overlap (Eq. 5); (3) measured mean squared overlap sits above the 1/m line at small alpha while still scaling as 1/m (Fig. 17; ratio ~1.5-2.5x read by eye); (4) Fig. 2c is built by minimising the squared off-diagonals of the normalised W (App. D.2), i.e. a frame-potential penalty; (5) p. 10: encouraging superposition 'could enable smaller models to match the performance of larger ones', citing nGPT and FOCUS, **untested in the paper**; the exponent cannot change for natural language, only coefficients; (6) the large toy models (n = 10240) scale lr as lr*(8/m)^0.25 with b-lr 2/m, the small ones (our Stage 0) use fixed 1e-2 (App. B); (7) negative decay Eq. 2 matches the reference code's formula; (8) the LLM analysis uses only pretrained models' unembedding rows: **no LLM was trained with negative weight decay**.

### SUP_2022_003 ★ — Softmax Linear Units (SoLU)
- Authors: N. Elhage, T. Hume, C. Olsson, N. Nanda, C. Olah, et al.
- Year / venue: 2022, Anthropic technical report (Transformer Circuits Thread)
- Peer review: **Not peer-reviewed**
- URL: https://transformer-circuits.pub/2022/solu/index.html — open access
- Level: **P** (first 100k of 429k characters; Section 4 formula and detailed loss tables not covered)
- Findings: replacing the MLP activation with SoLU raised the fraction of quickly-explainable neurons from ~35% to ~60% (blinded), smaller gain for the largest models; test loss and NLP evals "approximately the same". Caveat: SoLU may *hide* non-neuron-aligned features by shrinking them and recovering them through the following LayerNorm — "no free lunch".
- Relevance: an architectural intervention that improves neuron-level interpretability **without improving capability**, and whose interpretability gain is partly illusory. Direct precedent for README §3–4.

### SUP_2026_004 — Superposition Without Interference? Towards Isolated Interventions via Almost Orthogonal Features in Language Models
- Authors: Moritz Miller, Florent Draye, Bernhard Schölkopf
- Year / venue: 2026, **COLM 2026** (arXiv comments)
- Peer review: Peer-reviewed
- ID: arXiv:2602.04718 — https://arxiv.org/abs/2602.04718 — open access
- Level: **A**
- Abstract: bounds the propagation of feature interference by the self-coherence of the feature dictionary; adds orthogonality regularisation on the dictionary; enables more isolated interventions on mathematical-reasoning concepts "while preserving model performance". Not verified whether the dictionary is an SAE or model weights, nor whether training is pretraining or fine-tuning.
- Relevance: **closest found work to "control interference by optimisation"**. Must be read in full (Q-LIT-1). Its stated outcome is intervenability, not capability.

---

## Sparse autoencoders and their reliability

### SAE_2024_001 — Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning
- Authors: Dan Braun, Jordan Taylor, Nicholas Goldowsky-Dill, Lee Sharkey
- Year / venue: 2024; arXiv page lists no venue (believed NeurIPS 2024 — **unverified**)
- ID: arXiv:2405.12241 — open access
- Level: **A**
- Abstract: train SAEs to minimise KL between original and SAE-spliced model outputs; Pareto improvement (more performance explained, fewer total and active features).
- Relevance: SAE objectives that use the model's output — the inverse direction of this project (SAE adapts to model).

### SAE_2025_002 ★ — Low-Rank Adapting Models for Sparse Autoencoders
- Authors: Matthew Chen, Joshua Engels, Max Tegmark
- Year / venue: 2025, arXiv preprint (no venue listed)
- ID: arXiv:2501.19406 — open access
- Level: **A**
- Abstract: freeze a trained SAE and LoRA-fine-tune the *language model* so that splicing the SAE costs less; CE-loss gap reduced 30–55%; reaches e2e-SAE downstream CE 3–20× faster (Gemma), 2–10× (Llama); no harm to general capabilities reported.
- Relevance: **the closest existing case of "modify the LM's weights with an SAE-derived loss".** It shows the model can be made more SAE-decomposable cheaply; it does not claim the model becomes more capable.

### SAE_2025_003 ★ — Sparse Autoencoders Trained on the Same Data Learn Different Features
- Authors: Gonçalo Paulo, Nora Belrose
- Year / venue: 2025, arXiv preprint
- ID: arXiv:2501.16615 — open access
- Level: **A**
- Abstract: changing only the SAE seed, only ~30% of features shared (131k-latent SAE on Llama-3-8B MLP); ReLU+L1 SAEs more seed-stable than TopK.
- Relevance: an SAE-derived training signal is partly an artefact of one SAE run. Any loss computed from one SAE inherits this noise.

### SAE_2024_004 — A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders
- Authors: David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, Joseph Bloom
- Year / venue: 2024 (v6), **NeurIPS 2025 oral** (arXiv comments)
- ID: arXiv:2409.14507 — open access
- Level: **A**
- Abstract: features that look monosemantic fail to fire when a more specific child feature absorbs them; caused by sparsity optimisation on hierarchical features; varying SAE size or sparsity does not fix it.

### SAE_2025_005 — Learning Multi-Level Features with Matryoshka Sparse Autoencoders
- Authors: Bart Bussmann, Noa Nabeshima, Adam Karvonen, Neel Nanda
- Year / venue: 2025, arXiv preprint (no venue on page)
- ID: arXiv:2503.17547 — open access
- Level: **A**
- Abstract: nested dictionaries reduce absorption; better probing and concept erasure; minor reconstruction cost. Includes TinyStories experiments.

### SAE_2025_006 — SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability
- Authors: Adam Karvonen, Can Rager, Johnny Lin, et al., Samuel Marks, Neel Nanda
- Year / venue: 2025, **ICML 2025** (arXiv comments)
- ID: arXiv:2503.09532 — open access
- Level: **A**
- Abstract: eight metrics, >200 SAEs, eight architectures; gains on unsupervised proxy metrics do not reliably translate to practical performance.
- Relevance: if this project uses "SAE quality" as an outcome, SAEBench metrics are the reference.

### SAE_2025_007 — AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders
- Authors: Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, Christopher Potts
- Year / venue: 2025; arXiv page lists no venue (believed ICML 2025 — **unverified**)
- ID: arXiv:2501.17148 — open access; code https://github.com/stanfordnlp/axbench
- Level: **A** (numbers from the user's FeatureScalpel note are snippet-level)
- Abstract: on Gemma-2-2B/9B, prompting > fine-tuning for steering; difference-in-means best for detection; SAEs not competitive on either.

### SAE_2025_008 — Are Sparse Autoencoders Useful? A Case Study in Sparse Probing
- Authors: Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, Neel Nanda
- Year / venue: 2025, arXiv preprint (no venue on page)
- ID: arXiv:2502.16681 — open access
- Level: **A**
- Abstract: SAE probes do not consistently beat baselines; apparent SAE advantages are matched by simple non-SAE baselines.

### SAE_2025_009 ★ — Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers
- Authors: Thomas Heap, Tim Lawson, Lucy Farnik, Laurence Aitchison
- Year / venue: 2025 (v2 Jan 2026), arXiv preprint. Earlier circulated as "Sparse Autoencoders Can Interpret Randomly Initialized Transformers".
- ID: arXiv:2501.17727 — open access
- Level: **A**
- Abstract: SAEs on randomly initialised Pythia models get auto-interp and reconstruction scores similar to SAEs on trained models.
- Relevance: **"more interpretable features" measured by auto-interp is not evidence of better learned representations.** Every interpretability metric in this project needs a random-init / untrained-checkpoint control.

### SAE_2024_010 — Scaling and Evaluating Sparse Autoencoders
- Authors: Leo Gao, Tom Dupré la Tour, et al. (OpenAI)
- Year / venue: 2024, **ICLR 2025** (proceedings link seen in search)
- ID: arXiv:2406.04093 — open access
- Level: **S**
- Snippet-level: TopK SAEs; AuxK loss to revive dead latents (~10% extra compute, α ≈ 1/32); 16M-latent SAE on GPT-4 with ~7% dead latents.
- Relevance: standard modern SAE recipe if we train our own SAEs. Needs reading.

### SAE_2024_011 — Tracking the Feature Dynamics in LLM Training: A Mechanistic Study (SAE-Track)
- Authors: Yang Xu, Yi Wang, Hengguan Huang, Hao Wang
- Year / venue: 2024 (v3 Jun 2025), arXiv preprint
- ID: arXiv:2412.17626 — open access
- Level: **A** (search snippet adds: recurrent initialisation of each checkpoint's SAE from the previous one; Pythia 160M/410M/1.4B — unverified)
- Relevance: method for keeping an SAE current while the model trains — the core engineering problem of "SAE in the training loop".

### SAE_2025_012 — Evolution of Concepts in Language Model Pre-Training
- Authors: Xuyang Ge, Wentao Shu, Jiaxing Wu, Yunhua Zhou, Zhengfu He, Xipeng Qiu
- Year / venue: 2025, **ICLR 2026** (arXiv comments)
- ID: arXiv:2509.17196 — open access
- Level: **A**
- Abstract: crosscoders across pretraining checkpoints; most features form around a specific point; complex features later; two stages (statistical learning, then feature learning); feature evolution causally linked to downstream performance.
- Relevance: a baseline method for the README's "feature stability across checkpoints" and a warning that interventions early in training act on a model that has not yet formed most features.

### SAE_2025_013 — Negative Results for SAEs on Downstream Tasks and Deprioritising SAE Research (Google DeepMind)
- Year / venue: 2025, blog post (LessWrong / DeepMind Safety Research Medium) — **not peer-reviewed**
- URL: https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks
- Level: **S** (secondary — taken from the user's own FeatureScalpel literature note dated 2026-10-01, which marks it "Fetched")
- Content: dense linear probes beat SAE probes on OOD harmful-intent detection; DeepMind de-prioritised SAE work.

---

## Representation-aware and interpretability-aware training

### RAT_2024_001 ★ — Encourage or Inhibit Monosemanticity? Revisit Monosemanticity from a Feature Decorrelation Perspective
- Authors: Hanqi Yan, Yanzheng Xiang, Guangyi Chen, Yifei Wang, Lin Gui, Yulan He
- Year / venue: 2024, **EMNLP 2024 main** (arXiv comments)
- ID: arXiv:2406.17969 — open access
- Level: **P**
- Method: DecPO = DPO + λ‖zzᵀ − I‖²_F, z = MLP up-projection output; λ = 1e-4; best layer varies by task. Models: Llama-2-7B (base, chat), Llama-3-8B-Instruct. Tasks: toxicity, cognition reframing, sycophancy; GPT-3.5-judged.
- Results: e.g. Llama-2-7B-base toxicity 44.0 (DPO) → 56.0 (DecPO); not uniformly best (Llama-3 toxicity: L1-reg 58.0 > DecPO 57.0). No seeds/variance reported in what was read.
- Limitations: proxy metrics for monosemanticity; Llama family only; preference fine-tuning only (not pretraining).
- Relevance: **closest "monosemanticity regulariser improves performance" evidence**, but in fine-tuning, with an LLM judge and small evaluation sets.

### RAT_2024_002 — Learning from Emergence: A Study on Proactively Inhibiting the Monosemantic Neurons of Artificial Neural Networks (MEmeL)
- Authors: Jiachuan Wang, Shimin Di, Lei Chen, Charles Wang Wai Ng
- Year / venue: 2023/2024, arXiv preprint (venue placeholder only)
- ID: arXiv:2312.11560 — open access
- Level: **P**
- Method: online Monosemantic Scale ϕ(z) = (z − z̄)²/S²; "Reverse Deactivation" to suppress monosemantic neurons.
- Results: small gains (BERT GLUE avg 79.6 → 80.0/80.5; Swin-T 80.9 → 81.1); largest model 88M; no LLMs.
- Relevance: the **opposite** claim to RAT_2024_001 (inhibiting monosemanticity helps), on weak evidence.

### RAT_2025_003 — Learning Towards Emergence: Paving the Way to Induce Emergence by Inhibiting Monosemantic Neurons on Pre-trained Models (L2E)
- Authors: Jiachuan Wang, Shimin Di, Tianhao Tang, Haoyang Li, Charles Wang-wai Ng, Xiaofang Zhou, Lei Chen
- Year / venue: 2025, arXiv preprint
- ID: arXiv:2503.23298 — open access
- Level: **P**
- Method: during **pretraining** of Pythia 70M/410M/2.8B (10% of the Pythia schedule, deduplicated Pile), penalise the top-2% most monosemantic neurons in two middle blocks (log-MS penalty, weight 1e-9…1e-11).
- Results: 0-shot gains +2.3/+1.7/+1.9 points (ARC-C, PIQA, SciQ); overhead 1.5–3.4%; **single runs, no seeds or variance; no loss/perplexity reported**; authors themselves decline to attribute gains conclusively.
- Relevance: the only found **pretraining-time** monosemanticity regulariser on LMs — and it pushes *against* monosemanticity. Shows the literature has not settled even the sign of the effect.

### RAT_2025_004 ★ — Weight-sparse transformers have interpretable circuits
- Authors: Leo Gao, Achyuta Rajaram, Jacob Coxon, Soham V. Govande, Bowen Baker, Dan Mossing (OpenAI)
- Year / venue: 2025, arXiv preprint
- ID: arXiv:2511.13653 — open access; code https://github.com/openai/circuit_sparsity
- Level: **P** (first 100k of 124k characters)
- Method: GPT-2-style decoder; keep largest-magnitude weights after each AdamW step, L0 annealed over 50–80% of training; ~1/1000 nonzero weights; AbsTopK activations (~1/4 nonzero); trained on GPT-4-generated Python; 2048-token BPE.
- Results: at matched pretraining loss, pruned task circuits ~16× smaller than in dense models; sparsity trades capability for interpretability; larger total params improve the trade-off.
- Limitations: **100–1,000× more train/inference compute than dense models of comparable capability**; dead neurons; still some polysemantic features.
- Relevance: the most rigorous recent "train for interpretability from scratch" work. Its explicit finding is a **capability cost**, not a gain.

### RAT_2025_005 ★ — Mixture of Experts Made Intrinsically Interpretable (MoE-X)
- Authors: Xingyi Yang, Constantin Venhoff, Ashkan Khakzar, Christian Schroeder de Witt, Puneet K. Dokania, Adel Bibi, Philip Torr
- Year / venue: 2025, **ICML 2025** (PMLR v267, https://proceedings.mlr.press/v267/yang25ag.html)
- ID: arXiv:2503.07639 — open access
- Level: **P**
- Method: ReLU experts + sparsity-aware routing; upcycled from dense; chess (8L/512d) and FineWeb-10BT language models (S: 12L/768d, 180M active; M: 24L/1024d, 555M active).
- Results: chess val loss 0.211 vs 0.213 (GELU dense), interpretability coverage 0.428 vs 0.356. Language perplexity (Table 2, lower is better), recomputed by hand from the extracted table: MoE-X-S beats GPT-2 Small on all four sets (OpenWeb 19.42 vs 22.83; LAMBADA 28.11 vs 32.71; WT103 43.80 vs 49.89; WT2 42.58 vs 44.36), and MoE-X-M beats GPT-2 Medium on all four; **against the Switch-Transformer MoE baseline the result is mixed** (MoE-X-S worse on 3 of 4). The dense comparison is confounded: 180M active vs 124M parameters, and different training data (FineWeb vs WebText).
- Note on verification: the fetch tool's own summary stated the opposite direction for the dense comparison; the reading above comes from the raw numbers. This is a concrete example of why level-P numbers must be checked against the PDF.
- Relevance: interpretable-by-design architecture with capability at least comparable to a matched MoE baseline; the "better than GPT-2" claim is not a fair parameter- and data-matched comparison.

### RAT_2024_006 — Monet: Mixture of Monosemantic Experts for Transformers
- Authors: Jungwoo Park, Young Jin Ahn, Kee-Eung Kim, Jaewoo Kang
- Year / venue: 2024 (v4 Jun 2025); arXiv page lists no venue (believed ICLR 2025 — **unverified**)
- ID: arXiv:2412.04139 — open access
- Level: **A**
- Abstract: sparse dictionary learning built into end-to-end MoE pretraining, 262,144 experts per layer; knowledge manipulation "without degrading general performance".

### RAT_2023_007 — Codebook Features: Sparse and Discrete Interpretability for Neural Networks
- Authors: Alex Tamkin, Mohammad Taufeeque, Noah D. Goodman
- Year / venue: 2023, arXiv preprint
- ID: arXiv:2310.17230 — open access
- Level: **A**
- Abstract: vector-quantisation bottlenecks in every layer (fine-tuned, up to 410M); "only modest degradation in performance"; codes usable for steering.

### RAT_2024_008 ★ — Gradient Routing: Masking Gradients to Localize Computation in Neural Networks
- Authors: Alex Cloud, Jacob Goldman-Wetzler, Evžen Wybitul, Joseph Miller, Alexander Matt Turner
- Year / venue: 2024, arXiv preprint (no venue on page)
- ID: arXiv:2410.04332 — open access
- Level: **P**
- Method: data-dependent masks on gradients so chosen data update chosen subregions; forward pass unchanged. LM experiments: 28M TinyStories model, 303M FineWeb-Edu model, 0.7B virology model.
- Results: localisation and robust unlearning; **retain-set performance degrades in proportion to how much data is routed**.
- Relevance: a working method for **shaping where representations form during pretraining**; demonstrates the representational-control idea at small scale, with a capability cost.

### RAT_2025_009 ★ — Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning (CAFT)
- Authors: Helena Casademunt, Caden Juang, Adam Karvonen, Samuel Marks, Senthooran Rajamanoharan, Neel Nanda
- Year / venue: 2025, arXiv preprint (no venue in read text)
- ID: arXiv:2507.16795 — open access
- Level: **P**
- Method: find concept directions (PCA on activation differences, or SAE latents — own SAEs for Qwen/Mistral, Gemma Scope for Gemma-2-2B); humans select directions; project them out **during fine-tuning**; normal inference.
- Results: emergent misalignment on Qwen2.5-Coder-32B 7.0% → 0.51% (PCA), 2.4% (SAE); Mistral-Small-24B 6.6% → 1.2% (PCA), 4.6% (SAE); MMLU/GSM8K unchanged; random directions no effect.
- Relevance: **the clearest existing proof that interpretability-derived directions can usefully change what gradient descent learns.** But it controls *what generalises*, not capacity; and PCA often beat SAEs.

### RAT_2025_010 — Persona Vectors: Monitoring and Controlling Character Traits in Language Models
- Authors: Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey
- Year / venue: 2025, arXiv preprint
- ID: arXiv:2507.21509 — open access
- Level: **A**
- Abstract: activation directions predict trait shifts during fine-tuning; "preventative steering" during fine-tuning avoids them; can flag training data.

---

## Language-model training

### LMT_2024_001 — MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases
- Authors: Zechun Liu, Changsheng Zhao, Forrest Iandola, et al., Vikas Chandra
- Year / venue: 2024, **ICML 2024**
- ID: arXiv:2402.14905 — open access
- Level: **A**
- Abstract: for sub-billion models, architecture matters; deep-and-thin, embedding sharing, GQA; +2.7%/+4.3% over prior 125M/350M SOTA; block-wise weight sharing +0.7/+0.8%.

### LMT_2023_002 — TinyStories: How Small Can Language Models Be and Still Speak Coherent English?
- Authors: Ronen Eldan, Yuanzhi Li
- Year / venue: 2023, arXiv preprint
- ID: arXiv:2305.07759 — open access
- Level: **A**
- Abstract: GPT-3.5/4-generated stories with a 3–4-year-old vocabulary; models below 10M parameters produce fluent multi-paragraph stories; GPT-4-graded evaluation.
- Relevance: cheapest credible from-scratch setting; also a sequence-level-distillation dataset by construction.

### LMT_2024_003 — Findings of the Second BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora
- Year / venue: 2024, CoNLL 2024 BabyLM workshop (ACL Anthology 2024.conll-babylm.1); arXiv:2412.05149
- Level: **S**
- Snippets: 10M / 100M-word budgets; LTG-BERT / GPT-BERT hybrids won; curriculum learning mostly failed; more epochs helped; some teams used teacher distillation.
- Relevance: an established data-limited benchmark and publication venue suited to a student project.

### LMT_2024_004 — modded-nanogpt (NanoGPT speedrun)
- Author/maintainer: Keller Jordan and contributors
- Type: **official code repository / community benchmark**, not a paper
- URL: https://github.com/KellerJordan/modded-nanogpt
- Level: **S**
- Content: 124M model to FineWeb val loss 3.28 on 8×H100; baseline ~45 min (llm.c), now minutes; Muon optimiser introduced in a 2024 record; records must show p < 0.01 below 3.28 because of run-to-run variance.
- Relevance: a strong, cheap, well-tuned baseline recipe — and a reminder that untuned baselines are easy to "beat".

---

## Knowledge distillation

### KD_2025_001 ★ — Distillation Scaling Laws
- Authors: Dan Busbridge, et al. (Apple)
- Year / venue: 2025, **ICML 2025** (PMLR 267:5977–6045)
- ID: arXiv:2502.08606 — open access
- Level: **P** (first 100k of 339k characters; Appendix A limitations and G.4 truncation study not read)
- Method: pure logit distillation (λ = 1, τ = 1); students 143M–12.6B; C4 English.
- Findings: **capacity gap** — student loss first falls then rises as teacher grows; distillation beats supervised pretraining only below a student-compute threshold that grows with student size, and only when the teacher already exists (teacher cost excluded); requires full teacher logits over the vocabulary.
- Relevance: argues against "biggest available teacher" and quantifies when KD can help a small student.

### KD_2024_002 — MiniLLM: On-Policy Distillation of Large Language Models
- Authors: Yuxian Gu, Li Dong, Furu Wei, Minlie Huang
- Year / venue: 2023, **ICLR 2024**
- ID: arXiv:2306.08543 — open access
- Level: **A**
- Abstract: reverse-KL, on-policy distillation; 120M–13B students; better instruction following, calibration, less exposure bias.

### KD_2024_003 — Gemma 2: Improving Open Language Models at a Practical Size
- Authors: Gemma Team, Google DeepMind
- Year / venue: 2024, technical report
- ID: arXiv:2408.00118 — open access
- Level: **A**
- Abstract: 2B and 9B trained with knowledge distillation instead of next-token prediction; best performance for size.

### KD_2025_004 — gpt-oss-120b and gpt-oss-20b Model Card
- Author: OpenAI
- Year / venue: 2025, model card (technical report)
- URL: https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf ; https://huggingface.co/openai/gpt-oss-120b
- Level: **S** (search summary of the official card)
- Content: 116.8B total, ~5.1B active per token, 36 layers; MoE weights in MXFP4 → fits one 80 GB GPU; tokenizer o200k_harmony, **201,088 tokens**.
- Relevance: the README's proposed teacher. Vocabulary size makes logit-level KD into a small student impractical without cross-tokenizer methods (see papers/knowledge_distillation.md).

---

## Quantization

### QNT_2023_001 ★ — Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing
- Authors: Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort (Qualcomm AI Research)
- Year / venue: 2023; NeurIPS 2023 per secondary sources (**unverified**)
- ID: arXiv:2306.12929 — open access; code in qualcomm-ai-research GitHub
- Level: **S**
- Content: outliers arise from heads learning "no-op" updates through softmax; clipped softmax and gated attention reduce outlier magnitude/kurtosis and improve W8A8 quantisation while keeping or improving FP performance (BERT, OPT, ViT).
- Relevance: **established precedent that a training-time representation intervention improves quantisation robustness.**

### QNT_2025_002 ★ — Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models (OSP)
- Authors: Park et al.
- Year / venue: 2025, **ACL 2025** (aclanthology 2025.acl-long.618)
- ID: arXiv:2506.19697 — open access; code https://github.com/dmis-lab/Outlier-Safe-Pre-Training
- Level: **S**
- Content: Muon + single-scale RMSNorm + learnable embedding projection; 1.4B model, 1T tokens; W4A4 average 35.7 vs 26.5 (Adam); excess kurtosis 0.04 vs 1818.56; ~2% overhead; Muon alone insufficient.

### QNT_2026_003 — How Quantization Changes Interpretable Features: A Sparse Autoencoder Analysis of Language Models
- Author: Duan
- Year / venue: 2026, arXiv preprint
- ID: arXiv:2606.03002 — open access
- Level: **S**
- Content: Pythia-70M and Gemma-2-2B, round-to-nearest; INT7 on Gemma lowers perplexity yet degrades 18.7% of active SAE features; at INT6 51.3% (Gemma) / 62.4% (Pythia) of active features survive; survival predicted from FP statistics (AUC 0.92–0.97, peak activation strongest).
- Relevance: perplexity alone does not show features survive quantisation — directly relevant to README §8 "preservation of learned representations".

---

## User's own prior work

### FS_2026_001 — FeatureScalpel: transient SAE-feature editing in GPT-2 Small and cross-model transfer (technical paper draft + repository)
- Authors: four-person student team (MBCET), per the paper's author block
- Year / status: 2026, **unpublished** technical paper draft; repository with 144 commits and 37 research-journal entries
- Location: `D:\Work\PROJECTS\FeatureScalpel` (paper: `papers/technical_paper/technical_paper.tex`)
- Level: **L**
- Summary: see [phases/phase_00_project_understanding.md](phases/phase_00_project_understanding.md).

---

## Added 2026-11-11 for the reading list (level A — abstract/metadata checked)

### SUP_2022_005 — Polysemanticity and Capacity in Neural Networks
- Authors: Adam Scherlis, Kshitij Sachan, Adam S. Jermyn, Joe Benton, Buck Shlegeris
- Year / venue: 2022 (rev. 2025), arXiv preprint — arXiv:2210.01892 — open access
- Abstract: capacity = fraction of embedding space a feature uses; important features get dedicated dimensions, less important ones share in proportion to loss impact, least important are ignored.

### SUP_2023_006 — Superposition, Memorization, and Double Descent
- Authors: Tom Henighan, Shan Carter, Tristan Hume, Nelson Elhage, Robert Lasenby, Stanislav Fort, Nicholas Schiefer, Christopher Olah
- Year / venue: 2023, Anthropic technical report — https://transformer-circuits.pub/2023/toy-double-descent/index.html
- Summary (first part of page): with limited data, toy models store datapoints (overfitting) or features (generalising) in superposition; double descent between regimes.

### SUP_2024_007 — Mathematical Models of Computation in Superposition
- Authors: Kaarel Hänni, Jake Mendel, Dmitry Vaintrob, Lawrence Chan
- Year / venue: 2024, ICML 2024 Mechanistic Interpretability Workshop (arXiv comments) — arXiv:2408.05451
- Abstract: one-layer MLP computes pairwise AND over many features with ≈ m^(2/3) neurons; error-correction layers for deep circuits.

### LMT_2024_005 — Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws
- Authors: Zeyuan Allen-Zhu, Yuanzhi Li
- Year / venue: 2024, arXiv preprint — arXiv:2404.05405
- Abstract: knowledge measured in bits via synthetic factual tuples; ≈ 2 bits/parameter, holding under int8 quantisation.
- Venue: ICLR 2025 (proceedings.iclr.cc). Level A (abstract via fetch, 2026-11-13).
- Abstract points: five factors studied incl. architecture and data signal-to-noise (effect direction/size not read); prepending domain names to training data significantly increases capacity; GPT-2 + rotary matches or surpasses LLaMA/Mistral (GatedMLP) at storing knowledge, especially with shorter training; ~2 bits/param needs ~1000 exposures per fact (secondary summaries, unverified).

### RES_ARENA — ARENA curriculum, chapter 1.3 (Superposition & SAEs exercises)
- Type: educational material (not a paper); mirrored at https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/ ; official site arena.education (current version not checked). Level S.

### Code reference — liuyz0/SuperpositionScaling (official code for SUP_2025_002)
- Files read 2026-11-11: `exp/exp-10.py`, `exp/exp-10-3.py`, `exp/adamw.py`. **No licence.** Toy settings recorded in phases/stage_0_toy_baseline.md §2. Notable: no random seeds set; negative weight decay implemented as W += lr·γ·W·(1 − 1/‖W_i‖) row-wise.

## Not yet examined (to read; not cited as evidence anywhere)

Identified as relevant but not opened in session 1. Do not cite until examined.

| Topic | Work | Identifier (to verify) | Why |
|---|---|---|---|
| Superposition | Jermyn, Schiefer, Hubinger — Engineering Monosemanticity in Toy Models | arXiv:2211.09169 | Reducing polysemanticity in toy models without loss cost |
| Superposition | Scherlis et al. — Polysemanticity and Capacity in Neural Networks | arXiv:2210.01892 | Capacity allocation theory |
| Superposition | Henighan et al. — Superposition, Memorization, and Double Descent | Transformer Circuits 2023 | Data-limited regime + superposition |
| Regularisation | Cogswell et al. — Reducing Overfitting by Decorrelating Representations (DeCov) | arXiv:1511.06068 (ICLR 2016) | The classical non-MI baseline for a decorrelation loss |
| SAEs | Bricken et al. — Towards Monosemanticity | Transformer Circuits 2023 (PDF in FeatureScalpel/docs) | Foundational SAE method |
| SAEs | Cunningham et al. — SAEs Find Highly Interpretable Features | arXiv:2309.08600 (ICLR 2024) | Foundational |
| SAEs | Rajamanoharan et al. — JumpReLU SAEs | arXiv:2407.14435 | SAE architecture choice |
| SAEs | Marks et al. — Sparse Feature Circuits | arXiv:2403.19647 | Attribution with SAE features |
| SAEs | Dunefsky et al. — Transcoders; Lindsey et al. — Crosscoders | arXiv:2406.11944; Transformer Circuits 2024 | Alternatives to residual SAEs |
| Training | Hoffmann et al. — Chinchilla | arXiv:2203.15556 | Token/parameter budget |
| Training | Biderman et al. — Pythia | arXiv:2304.01373 | Checkpointed reference suite |
| Training | Muon optimiser papers (Jordan blog; Liu et al. "Muon is Scalable") | arXiv:2502.16982 | Optimiser choice; OSP link |
| Training | Geiping & Goldstein — Cramming | arXiv:2212.14034 | Single-GPU pretraining |
| Training | Wortsman et al. — Small-scale proxies for large-scale instabilities | arXiv:2309.14322 | Stability at small scale |
| KD | Hinton et al. 2015; Kim & Rush 2016 (sequence-level KD) | arXiv:1503.02531; arXiv:1606.07947 | Foundations |
| KD | Cross-tokenizer KD (e.g. Universal Logit Distillation) | arXiv:2402.12030 | Needed if teacher vocab ≠ student vocab |
| Quantisation | LLM.int8(); GPTQ; AWQ; Massive Activations | arXiv:2208.07339; 2210.17323; 2306.00978; 2402.17762 | PTQ methods and outlier theory |
| Side directions | tmsq (toy superposition + RTN quantisation, GitHub); Muon vs Adam features; Anthropic crosscoders; distilled-reasoning crosscoders; SAE-FD | github.com/fqybaba/tmsq; arXiv:2606.09658; transformer-circuits.pub/2024/crosscoders; arXiv:2503.03730; arXiv:2605.25525 | Located at snippet level for alternative_directions.md A1–A4 |
| Knowledge capacity | Allen-Zhu & Li — Physics of Language Models 3.3 (knowledge capacity scaling laws) | to locate | Facts-per-parameter test (Q-LIT-5) |

## Added 2026-11-13 for packing ideas (level A: search-result summaries only; NOT verified against full text)

### RAT_2018_011 - Learning towards Minimum Hyperspherical Energy
- Authors: Weiyang Liu, Rongmei Lin, Zhen Liu, Lixin Liu, Zhiding Yu, Bo Dai, Le Song. Year / venue: 2018, NeurIPS (peer-reviewed). arXiv:1805.09298 - https://arxiv.org/abs/1805.09298
- Idea: regulariser that spreads neuron directions evenly on the unit sphere (Thomson problem); half-space and angular variants; reported generalisation gains on vision tasks.
- Relevance: general form of a "pack directions better" penalty (packing_ideas.md idea 1). Not found: used for language-model feature packing.

### LMT_2025_006 - nGPT: Normalized Transformer with Representation Learning on the Hypersphere
- Authors: Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, Boris Ginsburg. Year / venue: 2025, ICLR (peer-reviewed). arXiv:2410.01131 - https://arxiv.org/abs/2410.01131
- Claim (abstract): unit-norm embeddings, weights and hidden states with learnable per-dimension step sizes; 4-20x fewer training steps to the same accuracy depending on sequence length. No independent reproduction seen. The speedup is in steps, not capacity per parameter.

### LMT_2025_007 - How much do language models memorize?
- Authors: John X. Morris et al. (Meta, Google DeepMind, Cornell, NVIDIA). Year: 2025, arXiv:2505.24832 (peer-review status not checked) - https://arxiv.org/abs/2505.24832
- Claim: ~3.64 bits/parameter for GPT models in half precision; capacity measured by separating unintended memorisation from generalisation; an empirical estimate, not a theoretical limit.

### LMT_2025_008 - FOCUS: First Order Concentrated Updating Scheme
- Authors: Yizhou Liu, Ziming Liu, Jeff Gore. Year: 2025, arXiv:2501.12243 (preprint) - https://arxiv.org/abs/2501.12243
- Claim (abstract): Signum plus an attraction term toward a moving average; faster than Adam and more stable than Signum on GPT-2. Cited by SUP_2025_002 as an optimiser that works without weight decay.

### RAT_2025_012 - SAFR: Superposition-Aware Feature Regularization
- arXiv:2501.16374 (preprint; authors not recorded). Regularises superposition via a modified loss for interpretability (monosemantic on important tokens, polysemantic on correlated pairs). Different goal from ours. Level A.

## Added 2026-11-14 for the hierarchy idea (level A: search-result summaries only; NOT verified against full text)

### RAT_2025_013 - The Geometry of Categorical and Hierarchical Concepts in Large Language Models
- Authors: Kiho Park, Yo Joong Choe, Yibo Jiang, Victor Veitch. Year / venue: ICLR 2025 (peer-reviewed; earlier ICML 2024 workshop version). arXiv:2406.01506 - https://arxiv.org/abs/2406.01506
- Claim (summary): hierarchically related concepts are orthogonal in a precise sense; categorical concepts are polytopes; tested on Gemma and LLaMA-3 with 900+ WordNet concepts. Code: github.com/KihoPark/LLM_Categorical_Hierarchical_Representations.

### LMT_2025_009 - Breaking Token Into Concepts: Extreme Compression in Token Representation via Compositional Shared Semantics
- Authors: Kavin R V, Pawan Goyal. Year / venue: 2025, EMNLP Findings. arXiv:2509.17737 - https://arxiv.org/abs/2509.17737
- Claim (summary): product-quantisation-style shared codebooks (Aggregate Semantic Grouping) compress embeddings to 0.4-0.5% of embedding parameters with >95% performance retained on mBERT, XLM-R, mT5 tasks. Post-hoc, embedding table only.

### RAT_2026_014 - Linear Representations of Hierarchical Concepts in Language Models
- arXiv:2604.07886 (preprint; authors not recorded). Claim (summary): parent-child relations are linearly recoverable in a low-dimensional (~150-250 dim), largely domain-specific subspace. Analysis, not a training method.
