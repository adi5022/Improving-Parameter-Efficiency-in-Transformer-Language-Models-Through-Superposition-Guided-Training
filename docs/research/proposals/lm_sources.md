# Sources for the standard-LM proposal (model B and shared pipeline)

Date: 2026-11-15. Companion to [lm_baseline_proposal.md](lm_baseline_proposal.md). Records follow CLAUDE.md section 11.2. **citation_index.md was not edited**; the lead merges these.

**Verification levels** (scheme of [../citation_index.md](../citation_index.md) / paper_library.md):
- **F(text)**: the document was downloaded and its text read directly by me (PDFs through `pdftotext`; web pages and API output as raw text). It is not a summarising tool, so this is stronger than P, but **no page images or figures were viewed**, which is why I do not write plain F. The "Read" line of each record names the sections or tables actually read; where it says "partial", the rest of the document was not read.
- **P**: read through a summarising fetch tool; numbers need spot-checking.
- **A**: abstract / first page only; methodology and experiments NOT verified.
- **S**: snippet only (none used here).

Keys: existing keys are reused (marked "existing"); new keys use the 2xx block per category. Peer-review status is taken from the arXiv comment field or the paper's own header where visible; otherwise "not stated on the arXiv page" (this does not mean it was rejected). Open access: all papers are on arXiv (free).

**Counts: 52 records. F(text): 35. P: 7. A: 10. S: 0.** (21 of the 31 papers are F(text), 10 are A.)

---

## Part 1. Papers

### LMT_2020_201: Scaling Laws for Neural Language Models
- Authors: J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, et al. Year: 2020. Venue: arXiv preprint (no venue on the arXiv page). Peer review: **not stated** (preprint).
- ID: arXiv:2001.08361. URL: https://arxiv.org/abs/2001.08361. Code: none found.
- Level: **F(text), partial.** Read: Sec. 1 summary and Eq. 1.1 to 1.4, Sec. 2.1 (parameter and compute counting, Table 1), Sec. 2.2 (training setup), the critical-batch passages.
- Findings: non-embedding parameters N approx 12 n_layer d_model^2; forward cost approx 2N + 2 n_layer n_ctx d_attn; training approx 6N FLOPs per token (embeddings excluded from N). Critical batch size B_crit(L) = B*/L^(1/alpha_B) with B* approx 2e8 tokens, alpha_B approx 0.21 (Eq. 1.4). Setup: Adam, 3,000 warmup steps, cosine decay to zero, batch 512 x 1,024 tokens, runs from 22M to 23B tokens.
- Limitations: N excludes embeddings (unlike Chinchilla), which matters for small models with large vocabularies; many runs below 100M parameters; cosine length fixed (Chinchilla argues this biases conclusions).
- Supports: the 6ND rule and attention extra term (A1), B_crit numbers (D-LM-9). Proposal section: 3.1, D-LM-6, D-LM-9.

### LMT_2022_202: Training Compute-Optimal Large Language Models (Chinchilla)
- Authors: J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, et al. Year: 2022. Venue: NeurIPS 2022 per common citation; **not stated on the arXiv page, unverified here**. Peer review: unverified.
- ID: arXiv:2203.15556. URL: https://arxiv.org/abs/2203.15556. Code: none.
- Level: **F(text), partial.** Read: abstract, Sec. 1 to 3 (three approaches), Table 2 (exponents), Table 3 and Table A3 (tokens for model sizes), App. B (cosine length), App. D.3 to D.4, App. E (frontier curvature), App. F (FLOP counting incl. embeddings, Table A4).
- Findings: parameters and tokens should scale about equally (exponents 0.50/0.50, 0.49/0.51, 0.46/0.54); Table 3: 400M gets 8.0B tokens (20.0 per parameter), 1B gets 20.2B; cosine cycle should match the number of steps; models from under 70M to over 16B; exact FLOPs / 6ND is 0.99 to 1.10 for 73M to 6.8B (Table A4); slight curvature in the frontier, "even smaller models may be optimal" (App. E).
- Limitations: smallest model about 70M, most above 500M; Approach 3 constants later shown inconsistent (LMT_2024_203); one data distribution.
- Supports: 20 tokens per parameter (D-LM-7), cosine length matching (D-LM-8), 6ND accuracy (A1).

### LMT_2024_203: Chinchilla Scaling: A replication attempt
- Authors: T. Besiroglu, E. Erdil, M. Barnett, J. You (Epoch AI). Year: 2024. Venue: arXiv preprint. Peer review: not peer reviewed (preprint).
- ID: arXiv:2404.10102. URL: https://arxiv.org/abs/2404.10102. Code: none checked.
- Level: **F(text), partial.** Read: abstract, Sec. 1 to 3 (re-fit), Eq. 3 and 4.
- Findings: Hoffmann et al.'s Approach 3 constants are inconsistent with their other approaches and implausibly precise; re-fit L(N, D) = 1.8172 + 482.01/N^0.3478 + 2085.43/D^0.3658 (Eq. 3), compatible with Approaches 1 and 2.
- Limitations: re-fit from points reconstructed from the paper's plots (the authors list extraction errors); not re-run on raw data.
- Supports: the loss-gap illustration for P vs 1.5P (proposal section 3.6), which is an extrapolation below the fitted range.

### LMT_2023_204: Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling
- Authors: S. Biderman, H. Schoelkopf, Q. Anthony, H. Bradley, et al. Year: 2023. Venue: ICML 2023 (PMLR 202; stated in the paper header). Peer review: peer reviewed.
- ID: arXiv:2304.01373. URL: https://arxiv.org/abs/2304.01373. Code: https://github.com/EleutherAI/pythia (not opened).
- Level: **F(text), partial.** Read: Sec. 2.2 to 2.4 (architecture choices, training, checkpoints), Table 1 (sizes), Table 3, App. D (Table 5 GPU hours); App. E configuration only partly.
- Findings: 8 sizes from 70M (18.9M non-embedding, d = 512, 6 layers, 50,304 untied vocabulary) to 12B; rotary embeddings, parallel attention and MLP, Flash Attention, untied embeddings; 300B tokens; batch 1,024 x 2,048 = 2,097,152 tokens for all sizes with "no convergence issues" for sub-billion models; 154 checkpoints per model; Table 5: 70M 510, 160M 1,030, 410M 2,540, 1.0B 4,830, 1.4B 7,120 A100-40GB hours (total 136,070 for one suite).
- Limitations: one tokenizer and corpus (the Pile); 2023-era training stack; the GPU-hour table extracted from PDF text was cross-checked against its total and TinyLlama's quoted 4,830 hours for Pythia-1.0B.
- Supports: utilisation calibration (22 to 34 percent, computed), batch size, checkpoint policy, embedding share of a 70M model (D-LM-9, D-LM-11, section 3.3).

### LMT_2024_205: OLMo: Accelerating the Science of Language Models
- Authors: D. Groeneveld, I. Beltagy, P. Walsh, et al. Year: 2024. Venue: arXiv; **not stated on the arXiv page** (published at ACL 2024 per common citation, unverified). Peer review: unverified.
- ID: arXiv:2402.00838. URL: https://arxiv.org/abs/2402.00838. Code: Apache 2.0 licence stated in the paper (repository not opened).
- Level: **F(text), partial.** Read: Sec. 2.1 (architecture choices), tokenizer/vocabulary paragraph (50,280, padded to 50,304, a multiple of 128), Table 1 optimiser settings (AdamW, betas 0.9 and 0.95), hardware paragraph.
- Findings: no biases, non-parametric layer norm, SwiGLU, RoPE.
- Limitations: 1B to 7B; superseded by OLMo 2 for the recipe.
- Supports: vocabulary padding, bias-free design, AdamW betas (D-LM-2, D-LM-8).

### LMT_2025_101: 2 OLMo 2 Furious (existing key; was S, now F(text) partial)
- Authors: Team OLMo (Walsh, Soldaini, Groeneveld, et al.). Year: 2024 (v3 2025). Venue: shorter version accepted to COLM 2025 (arXiv comment). Peer review: stated as accepted (shorter version).
- ID: arXiv:2501.00656. URL: https://arxiv.org/abs/2501.00656. Code: https://github.com/allenai/OLMo (not opened).
- Level: **F(text), partial.** Read: Sec. 2.1 (RMSNorm, reordered norm, QK-norm, z-loss, RoPE theta = 500,000), Sec. 2.2 and Table 2 (tokenizer), Sec. 3.2 (initialisation: truncated normal, std 0.02), Sec. 3.4.1 to 3.4.2 and Fig. 10 (epsilon 1e-5 to 1e-8; no weight decay on embeddings).
- Findings: excluding embeddings from weight decay stops embedding norms shrinking and reduces gradient spikes (spike score 0.16 vs 0.092); cl100k-based 100k vocabulary slightly better than the OLMo 1 tokenizer on a 1B/100B model (OLMES CF 60.6 vs 59.8), and the authors note, citing Tao et al., that the larger vocabulary is at a slight disadvantage at this size and compute.
- Limitations: 1B and above; the weight-decay study is shown for large runs.
- Supports: D-LM-4 (tokenizer), D-LM-8 (no decay on embeddings, epsilon, init).

### LMT_2025_206: SmolLM2: When Smol Goes Big: Data-Centric Training of a Small Language Model
- Authors: L. Ben Allal, A. Lozhkov, E. Bakouch, et al. Year: 2025. Venue: arXiv preprint (no venue on the page). Peer review: not stated.
- ID: arXiv:2502.02737. URL: https://arxiv.org/abs/2502.02737. Code: released with datasets (https://github.com/huggingface/smollm; not opened).
- Level: **F(text), partial.** Read: Sec. 2 (dataset ablation setup and FineWeb-Edu vs DCLM), Sec. 4.1 (training setup), Sec. 6 (135M and 360M), App. A (Table 6 architecture).
- Findings: 1.7B model: 256 H100, nanotron, AdamW (0.9, 0.95), WSD schedule with 2,000 warmup steps and peak LR 5e-4, tokenizer of 49,152 entries, 11T tokens, cost "around 250,000 USD" of GPU compute. 135M trained on 2T tokens and 360M on 4T, single-stage, WSD with 20 percent decay, LR 3e-3, Grouped-Query Attention. Architecture: 24 layers, d = 2,048, FFN 8,192, 2,048 tokens, 2M tokens per batch, tied embeddings, RoPE theta = 10,000, SwiGLU. Data ablations use 1.7B models with GPT-2 tokenizer and cosine LR 3e-4 on 350B tokens.
- Limitations: data-centric, small models trained far beyond compute-optimal; hyperparameters of the 135M/360M models given only in brief.
- Supports: over-training ratios (14,815 and 11,111 tokens per parameter, computed), tied embeddings, RoPE theta, LR range, FineWeb-Edu as a main corpus (D-LM-3, 5, 7, 8).

### LMT_2024_207: TinyLlama: An Open-Source Small Language Model
- Authors: P. Zhang, G. Zeng, T. Wang, W. Lu. Year: 2024. Venue: technical report (stated in arXiv comment). Peer review: not peer reviewed.
- ID: arXiv:2401.02385. URL: https://arxiv.org/abs/2401.02385. Code: https://github.com/jzhang38/TinyLlama (not opened).
- Level: **F(text), partial.** Read: Sec. 2.1 to 2.4 (data, architecture Table 1, speed optimisation, training).
- Findings: 1.1B, 22 layers, hidden 2,048, intermediate 5,632, 32 heads with 4 key-value groups, vocabulary 32,000; RoPE, RMSNorm, SwiGLU, FlashAttention-2; 24,000 tokens per second per A100-40G; AdamW (0.9, 0.95), cosine LR 4e-4 to 4e-5, 2,000 warmup steps, batch 2M tokens, weight decay 0.1, gradient clip 1.0; 16 A100-40G; about 3 epochs over about 950B tokens (3T target). These settings are the v1.0 description (Sec. 2.4); Sec. 2.5 says v1.1 was retrained from scratch after fixing learning-rate-scheduler and data-loading issues, with about 1.8M tokens per batch and 2T tokens instead of 3T.
- Limitations: technical report; the quoted throughput and hyperparameters are for v1.0; one size.
- Supports: utilisation of 51 percent (computed), recipe values, parameter check of the formula (D-LM-2, D-LM-8).

### LMT_2023_002: TinyStories: How Small Can Language Models Be and Still Speak Coherent English? (existing key; was A, now F(text) partial)
- Authors: R. Eldan, Y. Li. Year: 2023. Venue: arXiv preprint. Peer review: not stated.
- ID: arXiv:2305.07759. URL: https://arxiv.org/abs/2305.07759. Code/data: https://huggingface.co/datasets/roneneldan/TinyStories.
- Level: **F(text), partial.** Read: abstract, Sec. 1 to 3 (dataset, GPT-Eval), model/training description (GPT-Neo, window 256, context 512, top-10K tokens of the GPT-Neo tokenizer), scaling observations.
- Findings: stories generated by GPT-3.5/GPT-4 from a 1,500-word child vocabulary; models below 10M parameters produce coherent text; all models in the study train in at most 30 hours on a single V100; evaluation by GPT-4 grading.
- Limitations: synthetic narrow domain; GPT-4 grading needs a paid API; conclusions about capabilities do not transfer to web text.
- Supports: TinyStories as a development corpus, the 10k truncated vocabulary option, 512 context option (D-LM-4, 5, 6, 12).

### LMT_2024_001: MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases (existing key; was A, now F(text) partial)
- Authors: Z. Liu, C. Zhao, F. Iandola, et al. Year: 2024. Venue: ICML 2024 (arXiv comment). Peer review: stated.
- ID: arXiv:2402.14905. URL: https://arxiv.org/abs/2402.14905. Code: https://github.com/facebookresearch/MobileLLM (stated).
- Level: **F(text), partial.** Read: Sec. 1 to 2.3 (ablations: SwiGLU, deep-and-thin, embedding sharing Table 1, GQA), layer-sharing Table 2.
- Findings: embeddings are more than 20 percent of a 125M model; sharing input and output embeddings removes 16M parameters (135M to 119M) at an average zero-shot accuracy change from 44.8 to 44.6, and spending the saving on two extra layers gives 45.0 at 125M; 30 or more layers beat 12 at 125M; 32 A100 GPUs, 0.25T-token exploratory runs and 1T-token final runs.
- Limitations: accuracy averages over a handful of benchmarks without seed spread; sub-billion only.
- Supports: tied embeddings (D-LM-3), embedding share (section 3.2), depth/width choice noted as a postponed option.

### LMT_2024_208: The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale
- Authors: G. Penedo, H. Kydlicek, L. Ben Allal, et al. Year: 2024. Venue: arXiv; **venue not stated on the page** (NeurIPS 2024 Datasets and Benchmarks per common citation, unverified). Peer review: unverified.
- ID: arXiv:2406.17557. URL: https://arxiv.org/abs/2406.17557. Code: https://github.com/huggingface/datatrove (not opened); models and data on Hugging Face.
- Level: **F(text), partial.** Read: Sec. 1, Sec. 3.1 (ablation methodology, benchmark selection), start of Sec. 3.2 to 3.5, Sec. 5 summary of FineWeb-Edu.
- Findings: ablation models 1.71B parameters, Llama architecture, sequence 2,048, batch about 2M tokens, GPT-2 tokenizer; filtering ablations on about 28B tokens (about Chinchilla-optimal for that size); two models per dataset version trained with different data subsets and seeds and averaged; benchmarks chosen for low variance across runs, monotonic improvement and above-chance scores; about 80,000 H100 hours in total; released under ODC-By; education-classifier threshold 3.
- Limitations: web-data focus; 1.7B ablations.
- Supports: corpus choice, replication with 2 seeds as an existing minimal practice, benchmark criteria (D-LM-5, D-LM-12).

### LMT_2024_209: Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws
- Authors: N. Sardana, J. Portes, S. Doubov, J. Frankle. Year: 2024. Venue: ICML 2024 (arXiv comment). Peer review: stated.
- ID: arXiv:2401.00448. URL: https://arxiv.org/abs/2401.00448. Code: none found.
- Level: **A** (abstract and first page of the introduction). Methodology and experiments not verified.
- Findings (abstract): with large inference demand (about 1B requests or more) train smaller and longer than Chinchilla-optimal; 47 models show quality still improving up to about 10,000 tokens per parameter.
- Limitations: abstract only.
- Supports: over-trained regime as normal practice (D-LM-7); to be upgraded before it grounds a decision.

### LMT_2024_210: Language models scale reliably with over-training and on downstream tasks
- Authors: S. Y. Gadre, G. Smyrnis, V. Shankar, S. Gururangan, et al. Year: 2024. Venue: arXiv; **not stated on the arXiv page**. Peer review: unverified.
- ID: arXiv:2403.08540. URL: https://arxiv.org/abs/2403.08540. Code: https://github.com/mlfoundations/scaling.
- Level: **F(text), partial.** Read: abstract, Sec. 1 to 3 (testbed, token multiplier M = D/N, configurations, qk-LayerNorm and learning-rate remarks, compute used).
- Findings: 104 models from 0.011B to 6.9B on C4, RedPajama, RefinedWeb with token multipliers from 20 to 640; loss predicted for a 1.4B/900B-token run from 300x less compute; qk-LayerNorm made models insensitive to learning rate and they fixed LR 3e-3 for sweeps; 100 A100-hours to fit the loss law.
- Limitations: models at or above 0.011B; the paper writes the multiplier once as N/D (typo) and otherwise D/N.
- Supports: over-training range 20 to 640 (D-LM-7), LR range (D-LM-8), scaling-law use from small models (D-LM-1).

### LMT_2023_211: Scaling Data-Constrained Language Models
- Authors: N. Muennighoff, A. M. Rush, B. Barak, T. Le Scao, et al. Year: 2023 (v5 2025). Venue: NeurIPS 2023 per common citation, not stated on the arXiv page, unverified.
- ID: arXiv:2305.16264. URL: https://arxiv.org/abs/2305.16264. Code: https://github.com/huggingface/datablations.
- Level: **A** (abstract only).
- Findings (abstract): up to 4 epochs of repeated data give negligible loss change compared with unique data; beyond that the value of compute decays; up to 9B parameters and 900B tokens.
- Limitations: abstract only; large models.
- Supports: tolerance of 2 to 4 passes over TinyStories (D-LM-5); to be verified before use as a ground.

### LMT_2024_212: Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies
- Authors: C. Tao, Q. Liu, L. Dou, N. Muennighoff, et al. Year: 2024. Venue: NeurIPS 2024 (arXiv comment). Peer review: stated.
- ID: arXiv:2407.13623. URL: https://arxiv.org/abs/2407.13623. Code: https://github.com/sail-sg/scaling-with-vocab.
- Level: **F(text), partial.** Read: abstract, Sec. 1, Sec. 2 setup (N = N_nv + N_v, N_v = V d, Eq. 5), Sec. 3, Sec. 4.1 (Approach 1, power-law results), Fig. 1 to 5 captions.
- Findings: 33M to 3B non-vocabulary parameters, vocabulary 4K to 96K, up to 500B characters; the compute-optimal vocabulary grows with compute but slower than the non-vocabulary parameters (exponent gamma approx 0.83); Approach 1 fits N_nv = 0.08 C^0.50, N_v = 0.20 C^0.42, H = 6.42 C^0.50; most LLMs use too small a vocabulary at large scale; at 3B, vocabulary 32K to 43K improved ARC-Challenge from 29.1 to 32.0.
- Limitations: assumes Chinchilla-optimal data; I applied the Approach 1 fit below its smallest N_nv (33M) and counted N_v as V x d (how the paper counts tied/untied tables was not checked); results are for their tokenizer and data.
- Supports: vocabulary range (about 5k to 13k at 17M to 113M body parameters, computed) in D-LM-4.

### LMT_2018_213: An Empirical Model of Large-Batch Training
- Authors: S. McCandlish, J. Kaplan, D. Amodei, OpenAI Dota Team. Year: 2018. Venue: arXiv preprint. Peer review: not peer reviewed.
- ID: arXiv:1812.06162. URL: https://arxiv.org/abs/1812.06162. Code: none.
- Level: **A** (abstract and introduction).
- Findings: the gradient noise scale predicts the largest useful batch size across domains; it rises during training as loss falls.
- Limitations: not read beyond the introduction.
- Supports: why small batches suit small token budgets (D-LM-9; numbers come from Kaplan Eq. 1.4).

### LMT_2023_214: Small-scale proxies for large-scale Transformer training instabilities
- Authors: M. Wortsman, P. J. Liu, L. Xiao, et al. Year: 2023. Venue: arXiv; not stated on the page (ICLR 2024 per common citation, unverified).
- ID: arXiv:2309.14322. URL: https://arxiv.org/abs/2309.14322. Code: none found.
- Level: **F(text), partial.** Read: abstract, Sec. 1 to 3.1 (LR-sensitivity metric, attention-logit growth, output-logit divergence, default settings).
- Findings: instabilities seen at large scale (attention-logit growth, output-logit divergence) can be reproduced at small scale; qk-layernorm and z-loss reduce learning-rate sensitivity; longer warmup, independent weight decay and mu-Param also help; defaults AdamW (0.9, 0.95), epsilon 1e-8, clip 1, cosine decay, z-loss coefficient 1e-4.
- Limitations: model sizes about 1M to several B; sensitivity metric depends on the range of LRs tested.
- Supports: QK-norm and z-loss as switchable stability features (D-LM-8).

### LMT_2020_215: GLU Variants Improve Transformer
- Authors: N. Shazeer. Year: 2020. Venue: arXiv preprint. Peer review: not peer reviewed.
- ID: arXiv:2002.05202. URL: https://arxiv.org/abs/2002.05202. Code: none.
- Level: **F(text)** (short paper; abstract, definitions, Table 1 and results tables read).
- Findings: with d_ff reduced from 3,072 to 2,048 to keep parameters and compute equal, held-out log-perplexity (T5 segment-filling) of FFN-ReLU 1.997, GELU 1.983, GEGLU 1.942, SwiGLU 1.944 (Table 1); gains in downstream tables; author gives no explanation ("divine benevolence").
- Limitations: T5 base, one task; single-run standard errors of 0.003 to 0.010 shown.
- Supports: SwiGLU with d_ff approx 2/3 of the 4d (D-LM-2).

### LMT_2019_216: Root Mean Square Layer Normalization
- Authors: B. Zhang, R. Sennrich. Year: 2019. Venue: NeurIPS 2019 (arXiv comment). Peer review: stated.
- ID: arXiv:1910.07467. URL: https://arxiv.org/abs/1910.07467. Code: https://github.com/bzhangGo/rmsnorm.
- Level: **A** (abstract and introduction start).
- Findings: RMSNorm matches LayerNorm quality while cutting running time by 7 to 64 percent in their models.
- Limitations: tasks and architectures in the paper are not decoder-only language models trained from scratch at our scale; adoption evidence comes from LMT_2024_207 and LMT_2025_101.
- Supports: normalisation choice (D-LM-2).

### LMT_2021_217: RoFormer: Enhanced Transformer with Rotary Position Embedding
- Authors: J. Su, Y. Lu, S. Pan, et al. Year: 2021 (v5 2023). Venue: arXiv; journal publication per common citation, not stated on the page.
- ID: arXiv:2104.09864. URL: https://arxiv.org/abs/2104.09864. Code: Hugging Face integration stated.
- Level: **A** (abstract).
- Findings (abstract): rotary embeddings encode absolute position with a rotation matrix and give relative-position dependence; evaluated on long-text classification benchmarks.
- Limitations: the paper's own evidence is not language-model pre-training from scratch; adoption by Pythia, TinyLlama, SmolLM2, OLMo 2, PaLM (all read) is the practical evidence.
- Supports: position encoding (D-LM-2).

### LMT_2017_218: Using the Output Embedding to Improve Language Models
- Authors: O. Press, L. Wolf. Year: 2016 (v3 2017). Venue: EACL 2017 (arXiv comment). Peer review: stated.
- ID: arXiv:1608.05859. URL: https://arxiv.org/abs/1608.05859. Code: none checked.
- Level: **A** (abstract and introduction).
- Findings (abstract): tying input and output embeddings reduces perplexity of neural language models and can halve translation-model size.
- Limitations: LSTM-era evidence; not transformers.
- Supports: tied embeddings (D-LM-3), secondary to MobileLLM's ablation.

### LMT_2022_219: PaLM: Scaling Language Modeling with Pathways
- Authors: A. Chowdhery, S. Narang, J. Devlin, et al. Year: 2022. Venue: arXiv; journal version per common citation, unverified.
- ID: arXiv:2204.02311. URL: https://arxiv.org/abs/2204.02311. Code: none.
- Level: **F(text), partial.** Read: Sec. 2 architecture bullets (SwiGLU, RoPE, no biases, LayerNorm), Sec. 4.1 (MFU definition and Table 3), training-setup bullets (z-loss 1e-4 x log^2 Z, Adafactor, batch-size schedule).
- Findings: MFU is observed throughput over theoretical peak, excluding rematerialisation; PaLM 540B reached 46.2 percent (45.7 without rematerialisation); GPT-3 21.3 and Gopher 32.5 percent per its Table 3; auxiliary z-loss keeps the softmax normaliser near 1.
- Limitations: TPU training of a 540B model.
- Supports: the MFU concept used in the tables (section 1), z-loss definition (D-LM-8).

### LMT_2019_220: Decoupled Weight Decay Regularization
- Authors: I. Loshchilov, F. Hutter. Year: 2017 (v3 2019). Venue: ICLR 2019 (arXiv comment). Peer review: stated.
- ID: arXiv:1711.05101. URL: https://arxiv.org/abs/1711.05101. Code: none checked.
- Level: **A** (first page only).
- Findings: the AdamW formulation decouples weight decay from the gradient update.
- Limitations: not read; cited only to name the optimiser.
- Supports: D-LM-8 naming.

### LMT_2020_221: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- Authors: S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. Year: 2019 (v3 2020). Venue: SC20 per common citation, not stated on the arXiv page, unverified.
- ID: arXiv:1910.02054. URL: https://arxiv.org/abs/1910.02054. Code: DeepSpeed (not opened).
- Level: **F(text), partial.** Read: Sec. 3.1 (memory consumption of model states).
- Findings: mixed-precision Adam needs 2 (fp16 weights) + 2 (fp16 gradients) + K (K = 12: fp32 weights, momentum, variance) = 16 bytes per parameter; a 1.5B GPT-2 needs at least 24 GB for states.
- Limitations: excludes activations and fragmentation.
- Supports: memory rule 16N (A8, section 3.3).

### EVL_2025_201: Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation
- Authors: D. Heineman, V. Hofmann, I. Magnusson, et al. (Allen Institute for AI). Year: 2025. Venue: arXiv preprint. Peer review: not stated.
- ID: arXiv:2508.13144. URL: https://arxiv.org/abs/2508.13144. Code/data: allenai/signal-and-noise (stated).
- Level: **F(text), partial.** Read: abstract, Sec. 1, parts of Sec. 4 (SNR vs decision accuracy, R = 0.791) and Sec. 5 (interventions: subtask filtering, checkpoint averaging, bits-per-byte).
- Findings: signal-to-noise ratio of a benchmark predicts whether small-scale decisions hold at larger scale; switching to a metric such as perplexity rather than accuracy improves reliability; 375 models from 60M to 32B; HellaSwag low noise but low signal, ARC-Challenge high noise and high signal, MMLU low noise and high signal (their Fig. 1 example).
- Limitations: models trained by the authors' group and others; relevance to 10M to 126M models with our data not tested.
- Supports: bits per byte as the primary metric (D-LM-12).

### QNT_2022_201: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- Authors: E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. Year: 2022. Venue: ICLR 2023 (arXiv comment). Peer review: stated.
- ID: arXiv:2210.17323. URL: https://arxiv.org/abs/2210.17323. Code: https://github.com/IST-DASLab/gptq (not opened).
- Level: **F(text), partial.** Read: abstract, Sec. 1, Sec. 5 (setup, Tables 1 to 4, runtime), start of the 175B discussion.
- Findings: 128 random C4 segments of 2,048 tokens as calibration; one A100; OPT-125M WikiText2 perplexity 27.65 (16-bit), 4-bit RTN 37.28, 4-bit GPTQ 31.12, 3-bit RTN 1,300 (1.3e3), 3-bit GPTQ 53.85; at OPT-175B 4-bit GPTQ costs 0.03 perplexity; 1 to 3B models quantise in minutes; embeddings and output layer kept at fp16 in the large-model timing experiments.
- Limitations: weight-only; the library behind it (AutoGPTQ) is archived (INF_2026_201); perplexity-only evaluation.
- Supports: small models are much more fragile than large ones under 4-bit quantisation (D-LM-15).

### QNT_2023_202: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
- Authors: J. Lin, J. Tang, H. Tang, et al. Year: 2023. Venue: MLSys 2024, Best Paper (arXiv comment). Peer review: stated.
- ID: arXiv:2306.00978. URL: https://arxiv.org/abs/2306.00978. Code: https://github.com/mit-han-lab/llm-awq.
- Level: **F(text), partial.** Read: abstract, Sec. 1, Sec. 3.1 (salient weights, Table 1), Fig. 2.
- Findings: only about 0.1 to 1 percent of weight channels (chosen by activation magnitude) are salient; scaling them before quantisation protects them without mixed precision (OPT-6.7B INT3-g128: RTN perplexity 43.2 vs 13.0); TinyChat inference system.
- Limitations: evidence on models from about 1B up (the toy-size regime is not covered); the AutoAWQ library is archived (INF_2026_201).
- Supports: why GPTQ/AWQ are not proposed as the primary method (D-LM-15).

### QNT_2022_203: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- Authors: T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. Year: 2022. Venue: NeurIPS 2022 (arXiv comment). Peer review: stated.
- ID: arXiv:2208.07339. URL: https://arxiv.org/abs/2208.07339. Code: bitsandbytes (https://github.com/bitsandbytes-foundation/bitsandbytes; active per INF_2026_201).
- Level: **F(text), partial.** Read: abstract, Sec. 1.
- Findings: vector-wise quantisation plus mixed-precision decomposition for outlier features; systematic large-magnitude outliers emerge in all layers from about 6.7B parameters, a phase shift; below that, vector-wise int8 suffices.
- Limitations: evidence for large models; int8 only.
- Supports: int8 is expected to be benign at our sizes (D-LM-15; an expectation, not a measurement).

### QNT_2023_204: QLoRA: Efficient Finetuning of Quantized LLMs
- Authors: T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. Year: 2023. Venue: "extended NeurIPS submission" (arXiv comment); acceptance not verified here.
- ID: arXiv:2305.14314. URL: https://arxiv.org/abs/2305.14314. Code: https://github.com/artidoro/qlora.
- Level: **A** (abstract and introduction).
- Findings: 4-bit NormalFloat (NF4) data type, double quantisation, paged optimisers for fine-tuning; implemented in bitsandbytes.
- Limitations: fine-tuning method, not pretraining; not used for a number.
- Supports: naming NF4 as a bitsandbytes option (D-LM-15).

### QNT_2025_103: Scaling Laws for Precision (existing key; still A)
- Authors: T. Kumar, Z. Ankner, B. F. Spector, et al. Year: 2024. Venue: arXiv; not stated on the page. 
- ID: arXiv:2411.04330. URL: https://arxiv.org/abs/2411.04330. Code: none checked.
- Level: **A** (abstract and first page of the introduction re-read this session).
- Findings (abstract): degradation from post-training quantisation increases as models are trained on more data, eventually making extra pretraining harmful; fitted on over 465 runs, validated up to 1.7B parameters and 26B tokens.
- Limitations: abstract only; the existing index also lists it at A.
- Supports: D-LM-7 and D-LM-15 (both flagged A).

### QNT_2025_104: Low-Bit Quantization Favors Undertrained LLMs (existing key; still A)
- Authors: X. Ouyang, T. Ge, T. Hartvigsen, Z. Zhang, H. Mi, D. Yu. Year: 2024. Venue: arXiv ("Work in Progress"). Peer review: not peer reviewed.
- ID: arXiv:2411.17691. URL: https://arxiv.org/abs/2411.17691. Code/checkpoints: https://huggingface.co/Xu-Ouyang (stated).
- Level: **A** (abstract and Fig. 1 caption).
- Findings (abstract): larger models or fewer training tokens suffer less quantisation-induced degradation, smaller models with extensive training suffer more; over 1,500 checkpoints, GPTQ 2- and 3-bit.
- Limitations: abstract only; low-bit (2 and 3) focus.
- Supports: D-LM-7, D-LM-15 (flagged A).

---

## Part 2. Software, data, hardware and pricing sources

### LMT_2024_004: modded-nanogpt (NanoGPT speedrun) README (existing key; was S, now F(text) of the README)
- Authors: K. Jordan and contributors. Year: 2024 to 2026. Venue: GitHub repository. Peer review: none. Licence: MIT (GitHub API).
- URL: https://github.com/KellerJordan/modded-nanogpt. Code: same.
- Level: **F(text)**: raw README read (rules, record table, introduction); no run logs opened.
- Findings: task is validation loss at most 3.28 on FineWeb with 8 H100; the llm.c baseline took 45 minutes and 10B tokens; the current record uses under 330M tokens; records after modernising the architecture (rotary, QK-norm, ReLU-squared), Muon, FP8, embedding-skip connections; record 8 "untied embedding and head" took 12.0 to 10.8 minutes; the README lists record 79 at 1.411 minutes (2026-04-04); submissions must show p < 0.01 that mean validation loss is at most 3.28 because of inter-run variance.
- Limitations: a competitive benchmark, not a controlled study; specific to the 124M GPT-2-sized setting and the GPT-2 vocabulary.
- Supports: baseline strength caveat, untied-embedding counter-evidence, significance practice (D-LM-2, 3, 14).

### LMT_2024_223: nanoGPT README
- Authors: A. Karpathy. Year: 2022 to 2025. Venue: GitHub. Licence: MIT; last push 2025-11-12 (GitHub API 2026-11-15).
- URL: https://github.com/karpathy/nanoGPT. Level: **F(text)** (raw README).
- Findings: reproducing GPT-2 124M on OpenWebText needs an 8 x A100 40 GB node and "about 4 days", reaching about 2.85; GPT-2 124M scores 3.11 train/3.12 val loss on OpenWebText and about 2.85 after fine-tuning; code is "readable, hackable"; `torch.compile` in PyTorch 2.0 noted as not available on all platforms such as Windows (a December 2022 statement).
- Limitations: README statements, not experiments by me; the Windows remark is old and was not re-tested.
- Supports: code-base option (D-LM-16), context 1,024 reference, Windows caveat (section 3.7).

### LMT_2024_224: llm.c GPT-2 (124M) reproduction, discussion #481
- Authors: A. Karpathy et al. Year: 2024. Venue: GitHub discussion. Peer review: none.
- URL: https://github.com/karpathy/llm.c/discussions/481. Level: **P** (summarising fetch).
- Findings: 124M on 10B FineWeb tokens, 8 x A100 80GB SXM, about 90 minutes, about 20 USD at about 14 USD/h for the node; batch 524,288 tokens, 18,865 steps, peak LR 6e-4, warmup 700, weight decay 0.1; HellaSwag 29.9 vs 29.4 for GPT-2 124M; MFU stated "up to about 60 percent"; 350M: 14 hours, about 200 USD, about 30B tokens ("10B tokens was not enough"). Commenter claims (for example 8 H100 in about 40 minutes) were not used.
- Limitations: summariser output; thread is partly unverifiable user comments.
- Supports: utilisation calibration (55 percent computed), LR/batch/weight-decay reference, HellaSwag value at 124M (A4, D-LM-8, D-LM-9, D-LM-12).

### QNT_2026_201: llama.cpp `llama-quantize` documentation
- Authors: llama.cpp contributors (ggml-org). Year: 2026 (checked 2026-11-15). Venue: GitHub documentation. Licence: MIT; last push 2026-11-14 (GitHub API).
- URL: https://github.com/ggml-org/llama.cpp (tools/quantize/README.md). Level: **F(text)** (raw README).
- Findings: converts Hugging Face models to GGUF (`convert_hf_to_gguf.py`) and quantises (`llama-quantize`); measured on Llama-3.1-8B: bits per weight F16 16.0005, Q8_0 8.5008, Q6_K 6.5633, Q5_K_M 5.7036, Q4_K_M 4.8944, Q3_K_M 3.9960, Q2_K 3.1593; quality measured as perplexity and KL divergence; options to set the token-embedding and output-tensor types, importance matrices, per-tensor types.
- Limitations: bits per weight were measured on one 8B model; small models with large embeddings and other shapes will differ.
- Supports: file-size arithmetic (Tables 1b, 3.5) and the quantisation toolchain (D-LM-15).

### QNT_2026_202: llama.cpp conversion modules for GPT-2, GPT-NeoX, Llama
- Source: GitHub contents API listing of `conversion/` in ggml-org/llama.cpp, 2026-11-15 (files `gpt2.py`, `gptneox.py`, `llama.py` present). Level: **F(text)** (API JSON read directly).
- Findings: converters for GPT-2, GPT-NeoX and Llama architectures exist in the current conversion tree.
- Limitations: presence of a converter file, not a conversion test of our model.
- Supports: D-LM-2 and D-LM-15 (a Llama-style model is the safest target).

### INF_2026_201: GitHub repository status checks (software maintenance and licences)
- Source: GitHub REST API, queried 2026-11-15. Level: **F(text)** (JSON read directly).
- Findings: AutoAWQ (casper-hansen): archived, last push 2025-05-11, MIT. AutoGPTQ: archived, last push 2025-04-11, MIT. bitsandbytes: not archived, last push 2026-11-02, MIT. llama.cpp: not archived, 2026-11-14, MIT. flash-attention: BSD-3-Clause, 2026-11-14. nanoGPT: MIT, 2025-11-12. modded-nanogpt: MIT, 2026-09-28. nanotron: Apache-2.0, 2026-10-08. lm-evaluation-harness: MIT, 2026-11-04.
- Limitations: a point-in-time check; says nothing about quality. Forks of AutoGPTQ/AutoAWQ may exist (not checked).
- Supports: D-LM-15, D-LM-16.

### INF_2026_202: FlashAttention README (hardware requirements)
- Source: https://github.com/Dao-AILab/flash-attention (README.md, raw). Licence BSD-3-Clause. Level: **F(text)**.
- Findings: FlashAttention-2 requires Ampere, Ada or Hopper GPUs; for Turing (T4, RTX 2080) a separate project supports a subset; bf16 requires Ampere, Ada or Hopper.
- Limitations: about the library, not about PyTorch's built-in attention kernels (not checked).
- Supports: D-LM-10.

### HW_2026_201: NVIDIA T4 product page
- Source: https://www.nvidia.com/en-us/data-center/tesla-t4/. Level: **P** (summarising fetch).
- Findings: FP32 8.1 TFLOPS; mixed precision FP16/FP32 65 TFLOPS; INT8 130 TOPS; 16 GB GDDR6; memory bandwidth 320+ GB/s; 320 tensor cores; 70 W.
- Limitations: peak, not achieved, numbers; a search for measured T4 transformer-training throughput found nothing usable.
- Supports: assumption A3 (section 3.1).

### HW_2026_202: NVIDIA A100 product page
- Source: https://www.nvidia.com/en-us/data-center/a100/. Level: **P**.
- Findings (A100 80GB table): FP32 19.5 TFLOPS; TF32 156 dense / 312 with sparsity; BF16 and FP16 tensor core 312 dense / 624 with sparsity; HBM2e 80 GB, 1,935 GB/s (PCIe) or 2,039 GB/s (SXM). The summariser read the first of each pair as dense; the page marks the second as "with sparsity".
- Limitations: no 40 GB table on the page (Pythia and TinyLlama used 40 GB cards; the peak compute is the same family figure).
- Supports: MFU denominators (section 3.3, A4).

### HW_2026_203: NVIDIA H100 product page
- Source: https://www.nvidia.com/en-us/data-center/h100/. Level: **P**.
- Findings: H100 SXM: FP32 67; BF16/FP16 tensor 1,979 "with sparsity"; 80 GB; 3.35 TB/s. The page gives no dense figure; **my derivation of 989.5 TFLOP/s dense is half of 1,979.**
- Limitations: derivation not confirmed on the page.
- Supports: A5.

### HW_2026_204: Wikipedia, GeForce 16 series (GTX 1660 Ti specification)
- Source: https://en.wikipedia.org/wiki/GeForce_16_series. Level: **P**; secondary source.
- Findings: desktop GTX 1660 Ti: 1,536 shader processors, boost 1,770 MHz, 288 GB/s bandwidth, 5,437 GFLOPS single precision (boost), 10,875 GFLOPS half precision, no tensor cores.
- Limitations: secondary and desktop variant; the exact card in your machine (laptop or desktop) is unverified; TechPowerUp refused access (HTTP 403) and NVIDIA's product URL returned 404.
- Supports: A2.

### INF_2026_203: Kaggle Notebooks documentation
- Source: https://www.kaggle.com/docs/notebooks, read on 2026-11-15 as text through a third-party renderer (r.jina.ai) because the page is script-driven; Kaggle's own page returned only a title to the summariser. Level: **F(text)** (via renderer).
- Findings: each session has 12 hours of execution time for CPU and GPU notebooks (9 for TPU); 20 GB auto-saved disk; T4 x2 option = 2 Tesla T4, 4 CPU cores, 29 GB RAM; P100 option 1 GPU; 20 minutes of idle time while editing; "Save and Run All" must finish within 12 hours; free GPU availability is limited and sessions may queue at busy times; a Colab Pro link promotion gives 15 or 30 extra GPU hours per week, and quotas are shown in the account settings.
- Limitations: **the page gives no weekly GPU quota figure**; the commonly repeated "about 30 hours per week" is from secondary sources and unverified. A web search the same day returned only secondary sources and noted that one of them says the docs give no number.
- Supports: section 3.7, D-LM-11, D-LM-13.

### INF_2026_204: Colab FAQ
- Source: https://research.google.com/colaboratory/faq.html. Level: **P**.
- Findings: free-tier GPU types "vary over time"; limits are dynamic and unpublished; runtime at most 12 hours (24 hours with Pro+ and enough compute units); idle timeouts exist; paid plans give more compute and premium GPUs subject to availability.
- Limitations: no prices, compute-unit rates or GPU names on the page.
- Supports: section 3.7, D-LM-13.

### INF_2026_205: GPU rental price pages (Lambda, RunPod, Vast.ai)
- Source: https://lambda.ai/pricing, https://www.runpod.io/pricing (page "published" 2026-11-13), https://vast.ai/pricing, read 2026-11-15 as text through a third-party renderer. Level: **F(text)** (via renderer).
- Findings: RunPod: A100 SXM 80 GB 2.19 USD/h, A100 PCIe 2.09, H100 SXM 4.39, H100 PCIe 2.89, RTX 4090 0.89, L4 0.69, RTX A5000 0.27. Lambda (stated to take effect 2026-11-16): per-GPU-hour A100 40 GB 1.99, HGX A100 80 GB 3.19, H100 SXM 4.69 to 4.89 depending on instance size, HGX H100 4.59. Vast.ai marketplace "from / median": A100 SXM4 80 GB from 0.37 / 0.84, H100 SXM from 1.73 / 2.47, RTX 4090 0.14 / 0.49, T4 0.05 / 0.15.
- Limitations: list prices only; availability, storage, egress and interruption not checked; marketplace prices fluctuate; one RunPod page section shows a second, higher price list that I did not interpret.
- Supports: A6, D-LM-13. Not an endorsement; no account created.

### DAT_2023_201: TinyStories dataset card and file listing
- Source: https://huggingface.co/datasets/roneneldan/TinyStories (raw README, HF API tree listing, datasets-server row counts), 2026-11-15. Level: **F(text)**.
- Findings: licence cdla-sharing-1.0; TinyStories-train.txt 1,924,281,556 bytes; TinyStoriesV2-GPT4-train.txt 2,227,753,162 bytes; viewer reports 2,119,719 training and 21,990 validation rows.
- Limitations: token count not given (my estimate 0.43 to 0.48B, 4 to 4.5 bytes per token); which file the viewer's default configuration corresponds to was not checked.
- Supports: D-LM-5.

### DAT_2024_202: FineWeb-Edu dataset card
- Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu (raw README), 2026-11-15. Level: **F(text)**.
- Findings: licence ODC-By 1.0 plus Common Crawl terms of use; 1.3T tokens (threshold 3), 5.4T tokens in the score-2 variant; samples sample-10BT, 100BT, 350BT of "around" 10B, 100B, 350B GPT-2 tokens; classifier F1 82 percent at threshold 3; threshold 3 gave the best overall results; the authors' ablation model is 1.8B on 350B tokens.
- Limitations: token counts are for the GPT-2 tokenizer; "around" figures.
- Supports: D-LM-4, D-LM-5.

### DAT_2024_203: FineWeb dataset card
- Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb (raw README), 2026-11-15. Level: **F(text)**.
- Findings: ODC-By 1.0; more than 18.5T tokens; sample-10BT about 10B GPT-2 tokens (27.6 GB), sample-100BT 277.4 GB, sample-350BT 388 GB.
- Supports: D-LM-5 (rejected in favour of FineWeb-Edu).

### DAT_2023_204: Wikipedia dataset card (wikimedia/wikipedia)
- Source: https://huggingface.co/datasets/wikimedia/wikipedia (raw README), 2026-11-15. Level: **F(text)**.
- Findings: licences cc-by-sa-3.0 and gfdl; configuration 20231101.en has 20,200,062,385 bytes in the training split.
- Limitations: tokens not counted (about 4.7B at 4.3 bytes per token is my estimate).
- Supports: D-LM-5.

### DAT_2016_205: WikiText dataset card
- Source: https://huggingface.co/datasets/Salesforce/wikitext (raw README), 2026-11-15. Level: **F(text)**.
- Findings: over 100 million tokens from Good and Featured Wikipedia articles; Creative Commons Attribution-ShareAlike. The original dataset paper (Merity et al.) was not opened.
- Supports: D-LM-5 (optional held-out set).

### DAT_2023_206: SlimPajama announcement
- Source: https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama. Level: **P**.
- Findings: 627B tokens after removing 49.6 percent of bytes from RedPajama's 1.21T; "released under the Apache 2.0 license" (the post's statement).
- Limitations: the Hugging Face dataset card could not be read (HTTP 401 and "Invalid username or password" for my requests); underlying source licences and composition not verified here.
- Supports: why SlimPajama is not recommended (D-LM-5).

### DAT_2025_207: Benchmark sizes
- Source: Hugging Face datasets-server `size` endpoint for Rowan/hellaswag, allenai/ai2_arc, allenai/winogrande, cais/mmlu, EleutherAI/lambada_openai, nyu-mll/blimp, queried 2026-11-15. Level: **F(text)** (JSON read directly).
- Findings: HellaSwag validation 10,042; ARC-Easy test 2,376; ARC-Challenge test 1,172; WinoGrande validation 1,267; MMLU test 14,042; LAMBADA test 5,153; BLiMP 1,000 rows per phenomenon. PIQA could not be read (dataset runs custom code), so PIQA is not in the proposal.
- Limitations: row counts, not a check of which split lm-evaluation-harness uses by default.
- Supports: standard-error table in D-LM-12 (computed).

---

## Existing keys reused but not re-opened (no claim in the proposal rests on them)

- LMT_2025_010 (Muon is Scalable): cited only as the existing source for "about 52 percent of AdamW FLOPs" in D-LM-8 (existing level P).
- QNT_2026_003 (feature survival under RTN quantisation): cited only as the origin of the RTN protocol (existing level P).
- LMT_2024_005 (knowledge capacity): mentioned as the Stage 4 synthetic-fact test, not read here.
- SUP_2025_002 (superposition and scaling), D-S1-3 (Stage 1 seeds): referenced for project context only.

## Papers or documents not obtainable or not used

- No paywalled paper was needed. SlimPajama's dataset card, TechPowerUp's GTX 1660 Ti page and NVIDIA's GTX 1660 Ti page were not readable (HTTP 401, 403, 404); alternatives are noted above. No PDF from you is requested.

## Suggested rows for citation_index.md (for the lead; not applied)

| Topic | Claim | Key | Lvl |
|---|---|---|---|
| Compute rule | Training approx 6 FLOPs per parameter per token; exact count within 0.99 to 1.10 of 6ND (73M to 6.8B) | LMT_2020_201, LMT_2022_202 (App. F, Table A4) | F(text) partial |
| Tokens per parameter | Compute-optimal about 20 tokens per parameter (400M: 8.0B; 1B: 20.2B) | LMT_2022_202 (Table 3) | F(text) partial |
| Utilisation | Published runs reach 22 to 34 percent (Pythia), 51 (TinyLlama), 55 (llm.c) of A100 peak, computed from reported GPU-hours | LMT_2023_204, LMT_2024_207, LMT_2024_224 | F(text) / P |
| Over-training | SmolLM2-135M trained on 2T tokens (about 14,800 per parameter) | LMT_2025_206 | F(text) partial |
| Embedding share | A 125M-class model has more than 20 percent of its parameters in embeddings; sharing saves 16M parameters for -0.2 accuracy points (45.0 with 2 extra layers) | LMT_2024_001 (Table 1) | F(text) partial |
| Weight decay | No weight decay on embeddings reduced spikes in OLMo 2 | LMT_2025_101 (Sec. 3.4.2) | F(text) partial |
| Small-model quantisation | OPT-125M WikiText2 perplexity 27.65 (16-bit), 31.12 GPTQ 4-bit, 37.28 RTN 4-bit | QNT_2022_201 (Table 3) | F(text) partial |
| Software state | AutoGPTQ and AutoAWQ archived on GitHub (2026-11-15); bitsandbytes and llama.cpp active | INF_2026_201 | F(text) |
| Free compute | Kaggle docs: 12 h sessions, T4 x2, no weekly quota figure on the page | INF_2026_203 | F(text) via renderer |
