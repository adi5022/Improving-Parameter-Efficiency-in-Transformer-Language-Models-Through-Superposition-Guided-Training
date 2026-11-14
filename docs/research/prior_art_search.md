# Prior-Art Search for the "Pack More Per Dimension" Idea (Q-LIT-2)

Session 3, 2026-11-14. Question: has the project's idea already been done, and how novel are the two Stage 1 candidates (1a: a penalty on feature-direction overlap; 1b: tree-structured toy features with a shared parent direction)?

Status: **findings only, nothing here is an approved decision.** Hand-off notes: no code, no GPU, no git. Citation keys added to [citation_index.md](citation_index.md) use a **101+ numbering block per category** (for example `RAT_2019_101`) so they cannot collide with keys added by other sessions.

## 0. Verification levels used here

Same scale as [paper_library.md](paper_library.md), plus a note on how this session actually read things.

| Level | Meaning in this document |
|---|---|
| **L** | Full text read locally. Only SUP_2025_002 (user-supplied PDF, earlier session). |
| **P** | HTML full text opened through a summarising fetch tool (the tool returns a model-written summary of the page, not raw text). Where the page was longer than the tool window, the share actually read is stated. Quotable numbers must still be spot-checked against the paper before they go in a write-up. |
| **A** | arXiv abstract page, venue listing or search-result summary only. Method and experiments **not** verified. |
| **S** | Search snippet only. |

PDFs fetched directly came back as raw bytes (not readable), so the HTML or ar5iv versions were used. Nothing in this search was paywalled.

Peer-review wording: "peer-reviewed" only when a venue listing for acceptance was seen in the search results (the listing page was usually **not** opened). arXiv preprints are called preprints.

## 1. Search log (so nobody repeats it)

Tools: `WebSearch` (a summarising web search, US-only; each query returns summaries plus links), `WebFetch` (page to summary), arXiv abstract and HTML pages, the arXiv export API, Semantic Scholar citation endpoint, ar5iv.

**Not done or not possible:** Google Scholar was not queried directly. OpenReview was only reached through web-search results (one search restricted to openreview.net) and not through its own search page. GitHub was searched only through web search, not through GitHub code search, so "no repo found" means "none surfaced". The Semantic Scholar *search* endpoint returned HTTP 429 three times, so it was not used for keyword search (its *citations* endpoint worked once). The arXiv export API returned mostly irrelevant results for the combined queries (it is a poor tool for fuzzy phrases) and is not evidence of absence.

### Queries run (web search unless stated)

| # | Query (shortened) | Result |
|---|---|---|
| 1 | Liu/Liu/Gore follow-up "encourage superposition" regularizer | no follow-up regulariser found |
| 2 | frame potential regularizer weights, increase superposition, packing, LM | only SAFR, toy-model refs, an SAE continual-learning paper |
| 3 | Semantic Scholar citations of arXiv:2505.10465 | 37 citing papers; abstracts of 6 read (section 3) |
| 4 | "negative weight decay" embedding/unembedding LM superposition | only SUP_2025_002 and the OLMo-style weight-decay study |
| 5 | regularizer to "encourage superposition", feature density, capacity | SAFR (different goal); blog posts |
| 6 | joint training SAE and LM, SAE loss added to LM objective | MetaSAEs (SAE-only); LessWrong SAE-regularization post |
| 7 | equiangular tight frame regularization, Welch bound, coherence penalty | Bank and Giryes; ETF theory papers |
| 8 | toy superposition with hierarchical / tree features, shared parent direction | absorption papers; one forum question; no trained-superposition tree study |
| 9 | superposition + correlated / anticorrelated features, hierarchy | Prieto et al., spectral superposition |
| 10 | hyperbolic embeddings, hierarchy loss, LM pretraining | He et al. (encoders), HELM, HypStructure |
| 11 | frame / compressed-sensing theory of superposition (Mixon) | Ivanitskiy et al. 2026 |
| 12 | Representation Degeneration Problem (cosine regularisation) | Gao et al. ICLR 2019 |
| 13 | Spectrum Control (ICLR 2020) | Wang et al. |
| 14 | Bank and Giryes dropout / ETF | arXiv:1810.06049 |
| 15 | fixed ETF / Grassmannian unembedding in LM pretraining | neural-collapse papers; nothing for LM pretraining |
| 16 | SAE-guided regularisation, continual learning | arXiv:2606.26629, 2605.25525 |
| 17 | SAE in the loop / auxiliary loss during pretraining, interpretable-by-design | LessWrong post; MFN concept paper |
| 18 | SAE reconstruction as regulariser for fine-tuning | SAE-FT (CLIP), feature drift papers |
| 19 | OpenReview ICLR 2026, SAE jointly trained with transformer | nothing matching |
| 20 | LessWrong "SAE regularization produces more interpretable models" | page found and read |
| 21 | "Training for Interpretability: anti-superposition" blog | page found and read |
| 22 | quantisation + SAE features / superposition robustness | Duan 2026; Bereska et al. |
| 23 | superposition + forgetting / interference harms | Masip et al.; Henighan et al. (known) |
| 24 | scaling laws vs model architectures (exponent vs offset) | Tay et al.; a 2026 preprint on constants vs exponents (not opened) |
| 25 | orthogonality regularisation embedding matrix LM pretraining | nothing for embeddings; POET/Muon-type results are about weights or optimisers |
| 26 | Minimum Hyperspherical Energy in LMs | no LM-pretraining use found; Hyper-SET is an architecture |
| 27 | critique or replication of SUP_2025_002 | none found |
| 28 | scaling laws for precision (Kumar et al.) | confirmed |
| 29 | low-bit quantisation favors undertrained LLMs | confirmed |
| 30 | outliers, massive activations vs superposition | no direct link found |
| 31 | mechanistic-interpretability-guided training, circuit-aware regularisers | OpenAI weight-sparse, post-training sparse attention; none raises capability |
| 32 | "negative weight decay" (generic) | nothing on negative decay |
| 33 | "fit more features than dimensions", smaller model matches larger (extended search) | SUP_2025_002; name-collision papers |
| 34 | GitHub: TMS repos with frame potential or coherence loss | none (anthropics/toy-models-of-superposition, zroe1, tms-kit have none) |
| 35 | weight decay on token embeddings, ablations | OLMo 2 report |
| 36 | Muon + superposition / effective rank | arXiv:2603.26554 and others |
| 37 | HELM hyperbolic LLM | NeurIPS 2025 |
| 38 | feature hedging / absorption, tree-structured toy models | Chanin et al. |
| 39 | arXiv API: superposition AND regularization etc. (3 queries) | mostly noise; no new relevant hits |
| 40 | Prieto et al. venue check | ICLR 2026 poster listing seen |

## 2. Findings by question

### Q1. Has anyone trained with a regulariser that deliberately increases superposition / packing, or lowers the coherence of feature directions?

**Not found: any work whose stated purpose is to increase superposition or packing so a model needs fewer parameters.** The nearest statement of the idea is the one the project already knows: SUP_2025_002 (level L) says encouraging superposition "could enable smaller models to match the performance of larger ones" and names nGPT and weight-decay-free optimisers as possible examples, without testing it. A search of the 37 papers that cite it (list in section 3) turned up no paper that tests this.

**Found: many regularisers that lower coherence or spread directions, but for other goals.** These matter because they cover the *form* of 1a.

- **Gao et al., ICLR 2019 (RAT_2019_101), level P.** Adds γ · (1/N²) Σ_{i≠j} cos(w_i, w_j) to the loss on the **tied** word-embedding matrix (γ = 1, said to be insensitive). It is a pairwise-cosine penalty on token vectors in real LMs (AWD-LSTM on WikiText-2; Transformer base and big on WMT14 En-De). Gains are small: perplexity 66.0 to 65.2 without fine-tuning, BLEU +0.5 to +1.1. Goal: fix "representation degeneration" (embeddings collapsing into a narrow cone), not storage. This is the closest thing to a 1a-style penalty on a real language model. It uses the plain (not squared) cosine, which is a different function from a frame potential, but the aim (spread the rows apart) is the same.
- **Bank and Giryes, arXiv 2018 (BMVC 2020 per a search record, not verified) (RAT_2018_103), level P.** Penalty ‖AᵀA − |G_ETF|‖_∞ (max entry of the difference between the weight Gram matrix and an equiangular-tight-frame Gram matrix); the squared and absolute-sum versions were tried and the max version was best on Fashion-MNIST. Tested on FC, conv and LSTM layers (Penn Treebank word-level LSTMs, perplexity gains mostly via combination with dropout). Authors state gains are modest, large β hurt optimisation, the coherence term did not reach the Welch bound in linear autoencoders, and training is slower. So a coherence/ETF-seeking penalty has been used as a **regulariser for generalisation** already. It is a different goal from packing.
- **Minimum Hyperspherical Energy (RAT_2018_011 in the index, level A) and its CVPR 2020 follow-up (RAT_2020_104, level A).** Spread neuron directions uniformly on the sphere. Vision classifiers. The CVPR 2020 follow-up reports that naive MHE has optimisation difficulties. No LM-pretraining use found.
- **Wang et al., ICLR 2020, Spectrum Control (RAT_2020_102), level A.** Trains the singular values of the output embedding towards a slow-decay prior to counter anisotropy; reports gains on Transformer-XL and translation (about 1.5 BLEU on IWSLT14 De-En). Another "spread out the output embeddings" method.
- **SUP_2026_004 (Miller, Draye, Schölkopf, COLM 2026; level P, ≈58% of the text read).** Penalty λ‖tril(DᵀD)‖²_F on the **decoder of a TopK SAE** (65,536 latents, Gemma 2 2B, layer 13), used to make interventions more isolated. This answers **Q-LIT-4: it acts on an SAE dictionary during SAE fine-tuning, not on model weights during pretraining.** The LM is then LoRA-adapted with the frozen SAE inserted. Reported: GSM8K accuracy comparable across λ; first-name recovery after intervention about 74.9% at λ = 1e-4 versus about 62.2% at λ = 0; the authors' intermediate λ values were inconclusive. Opposite goal to ours (they want less coupling), same mathematical object (squared off-diagonals of a Gram matrix). Code: github.com/mrtzmllr/sae-icm.
- **Orthogonality penalty on neuron activation correlations in a 30M-parameter TinyStories GPT (Hisku, blog, 2026-03-14; RAT_2026_105; level P).** Single runs, no seeds, no code. Orthogonality (λ = 0.01) cost about 1.5 perplexity points from a ≈ 19.8 baseline and decorrelated neurons without making them monosemantic. Not peer-reviewed, tiny evidence, but it is a direct "frame-style penalty in a small LM" data point (it goes in the anti-superposition direction).
- **SAFR (RAT_2025_012, already in the index, level A).** Different goal (interpretability).
- **Looked for and not found:** a fixed or regularised ETF/Grassmannian *unembedding* in LM pretraining (neural-collapse papers do this for classifiers only; Zhu et al. reported fixing the classifier saves memory without hurting ResNet accuracy, level S). Also not found: a toy-model-of-superposition repo with a frame-potential or coherence loss (searched via web search only).

**What 1a's penalty would do, by reasoning (mine, not from a paper):** a frame potential Σ(Ŵ_i·Ŵ_j)² is *minimised* by well-spread, ETF-like directions. In strong superposition SUP_2025_002 says the task loss already contains these squared overlaps (loss ∝ squared overlaps ∝ 1/m), so the penalty duplicates part of what the loss already pushes on. It targets lever A (the coefficient) and can only help if gradient descent is stuck in a worse packing than the penalised optimum. It does **not** by itself make more features get a direction (lever C); that is what negative decay or norm-pulling does, by keeping row norms near 1. A penalty that reduces overlap and a term that keeps norms from shrinking are two different forces.

### Q2. Negative weight decay or norm-pulling on embedding/unembedding rows in real LMs

- **Found:** SUP_2025_002 (level L) applies decoupled weight decay (negative values pull row norms toward 1) to the embedding/unembedding rows in the **toy** model only. It analyses open LLMs read-only (row-normalised LM-head overlaps follow the 1/m pattern).
- **Not found:** any paper that trains a real LM with negative weight decay, or deliberately norm-pulls embedding rows, to gain capacity. The generic search "negative weight decay" returned only ordinary weight-decay literature (norm growth, effective learning rate, level S).
- **Related practice (level S, snippet):** the OLMo 2 report (arXiv:2501.00656) excluded token embeddings from weight decay because decay pushed embedding norms down and made training spikier (spike score 0.16 versus 0.092 in their figure). So embedding-norm control is already an engineering knob in real pretraining, but pointed at stability, not capacity.
- **Related study (level A):** Kulkarni et al., arXiv:2602.20433 (SUP_2026_109): 108 OLMo-style models; lower weight decay reduces effective rank of the unembedding, but effective rank does not reliably predict performance. This is a warning that geometry metrics can move without performance moving.
- **Adjacent architecture (already indexed):** nGPT (LMT_2025_006, level A) keeps everything unit norm. The speedup claim is in steps; no independent reproduction seen.

### Q3. Control of superposition in toy models beyond Liu et al.; follow-ups, critiques, replications

- **Critiques/replications of SUP_2025_002:** none found (one targeted search plus the citation list). The only stated limitations are the authors' own (exponents depend on the data distribution; mechanism-to-LLM link needs more work).
- **Follow-ups from the same group (all preprints, level A):**
  - Liu and Gore, "Neural Scaling Universality: If Exponents Are Fixed, Time to Understand Coefficients," arXiv:2606.25008 (SUP_2026_102): position paper; says the exponents are fixed by generic mechanisms (inverse width from superposition, inverse depth, one-third time exponent) and **coefficients are where near-term gains are**. It does not propose a method to change coefficients. This supports the project's framing (coefficient, not exponent) and weakens any hope of a new exponent.
  - Guo, Liu, Gore, "Emergent phases of superposition," arXiv:2609.36455 (SUP_2026_103): width phase transition from partial to full representation; non-uniform firing probabilities delay the transition and lower loss. Analysis, no regulariser.
  - Chen et al., "Superposition unifies power-law training dynamics," arXiv:2602.01045 (SUP_2026_104): with a superposition bottleneck the training power-law exponent is about 1 and training is up to about 10x faster than sequential learning (their teacher-student setup).
- **Other toy-model work:**
  - Ivanov et al., "Spectral Superposition," arXiv:2602.02224 (SUP_2026_105, level A): frame-operator spectrum; capacity saturation gives tight frames; diagnostic, no training method.
  - Ivanitskiy, Jasper, King, Mixon, "Towards a mathematical theory of superposition," arXiv:2608.27540 (SUP_2026_106, level A): frame theory and compressed sensing; exact recovery threshold for real ETFs. Background for the Welch-bound/coherence maths.
  - Prieto et al., ICLR 2026 poster (SUP_2026_101, level P, ≈81% of the page read): for correlated features, interference can be **constructive**; more common with weight decay; produces semantic clusters and cyclic structures. Only weight decay is studied as a control; no new regulariser. Code: github.com/LucasPrietoAl/correlations-feature-geometry.
  - Gorton and Lewis, arXiv:2508.17456 (SUP_2025_107, level P): in the Elhage toy model (100 features, 20 dims), vulnerability to adversarial attacks rises with features per dimension (reported r ≈ 0.99). Real-model evidence only via SAE-based proxies.
  - Bereska et al., TMLR (SUP_2025_108, level A): defines superposition as effective features per neuron via SAE entropy; **disagrees** with Gorton and Lewis: adversarial training can raise or lower superposition depending on task and capacity regime.
  - Review: Shi et al., arXiv:2609.06862 (SUP_2026_112, level A) surveys the field; no venue.
  - Turner, Wu, Batson, "Characterizing interference weights in a tiny language model," Transformer Circuits Thread, 2026-08-21 (SUP_2026_111, level A, truncated read): interference weights are described as irrelevant or harmful; not a training method.
- **Optimiser lever:** Kim et al., arXiv:2603.26554 (SUP_2026_113, level A): in a linear associative-memory model where stored associations can exceed the embedding dimension, one Muon step stores substantially more associations than SGD. Relevant to the project's A2 (Muon vs AdamW) side direction. Preprint, toy theory.
- **Name collisions (only superficially similar, level S):** "Token Superposition Training" (Nous Research, arXiv:2605.06546; a pretraining speed-up that bags tokens, not feature packing); "Superposition in Transformers: A Novel Way of Building Mixture of Experts" (arXiv:2501.00530); "Superposition of many models into one" (arXiv:1902.05522; packs several tasks into one network). None is about packing features per dimension.

### Q4. Training-time use of SAE-derived signals; joint SAE+LM training; what happened

| Work | What it does | Outcome reported |
|---|---|---|
| LessWrong post, Lai and StefanHex, 2025-01-28 (RAT_2025_106; blog; level P; code github.com/peterlai/gpt-circuits) | 4-layer character-level GPT-2 (width 64) on Shakespeare, an SAE before and after every block, SAE losses added to the cross-entropy so the LM and SAEs train together; SAE reconstruction loss rescaled by activation magnitude to stop activations shrinking; SAEs discarded afterwards and fresh SAEs trained to evaluate | Fresh SAEs on regularised models had better reconstruction at fixed L0 ≈ 50 (layer 3 inconsistent); regularised models reached slightly lower cross-entropy; coefficients were set large to amplify the effect, large values hurt model performance; authors say it is expensive and untested at scale |
| SUP_2026_004 (above) | Coherence penalty on SAE decoder, SAE inserted into the LM, LoRA | isolation of interventions, performance preserved on GSM8K |
| MetaSAEs, Levinson, arXiv:2604.03436 (RAT_2026_107, level A) | Jointly trains a small "meta SAE" with the primary SAE to penalise blended latents | GPT-2 large: 7.5% lower mean |φ|, +7.6% fuzzing score; Gemma results "directional". SAE-only, the LM is not trained |
| Monosemantic Feature Neurons, Dev, Preprints.org 202510.2064 (RAT_2025_108, level A) | Sparse-autoencoding layer inside each block, loss backpropagating into the Transformer weights | Concept paper, **no experiments**; not peer-reviewed |
| SAE-guided activation regularisation for LLM continual learning, Ning et al., arXiv:2606.26629 (RAT_2026_109, level A via search summaries) | Frozen SAE; per-task feature masks; penalises drift of features in a protected region | Reported to beat EWC/SI/MAS/ELLA-type baselines on TRACE and MedCL; frozen SAE, fine-tuning |
| SAE-FD, Zhang et al., arXiv:2605.25525 (RAT_2026_110, level S) | Anchors LM representations in a pretrained SAE's sparse feature space for continual learning | Claims first SAE-space regularisation method for continual learning; preprint |
| SAE-FT, Morelli et al., arXiv:2605.15961 (RAT_2026_111, level S) | Penalises added/removed SAE features while fine-tuning CLIP | Claims parity or better vs robust fine-tuning on ImageNet shifts; preprint |
| Already indexed: LoRA-for-SAE (SAE_2025_002), CAFT (RAT_2025_009), gradient routing (RAT_2024_008), Persona vectors (RAT_2025_010), end-to-end SAE (SAE_2024_001) | see paper_library.md | — |

**Not found:** an SAE trained *online and jointly* with an LM **from scratch** at a scale larger than a toy, with capability or efficiency results; any work using an SAE-derived loss to *increase* superposition; any measured SAE-instability-in-the-gradient study for from-scratch training. The LessWrong post is the only joint LM+SAE training found, and it is a blog-level toy result with the opposite aim (better SAEs).

### Q5. Hierarchical / shared-parent features (relevant to 1b)

- **Park, Choe, Jiang, Veitch, ICLR 2025 oral (RAT_2025_013, in the index at level A; venue checked this session from the arXiv page, comments confirm ICLR 2025 oral).** Concept hierarchy is encoded so that child minus parent is orthogonal to the parent in Gemma and LLaMA-3 (900+ WordNet concepts). So real LMs already show "parent plus offset" geometry. Code: github.com/KihoPark/LLM_Categorical_Hierarchical_Representations.
- **Feature absorption (SAE_2024_004) and feature hedging** (Chanin, Dulka, Garriga-Alonso, arXiv:2505.11756, SAE_2025_101, level A): toy models with hierarchical features (a child fires only if its parent fires) and correlated features, but with **orthogonal** feature directions and an SAE as the object of study. They show how hierarchy breaks SAEs, not how a model packs hierarchy. They also note that narrow SAEs mix parent and child components, depending on correlation sign.
- **Matryoshka SAEs (SAE_2025_005, level A):** nested dictionaries as a fix. A LessWrong analysis disputes the mechanism (search snippet only, level S).
- **SynthSAEBench** (Chanin and Garriga-Alonso, arXiv:2602.14687, SAE_2026_102, level A): synthetic data with correlation, hierarchy and superposition for evaluating SAEs; page did not say how parent-child is modelled; no code link on the arXiv page.
- **Prieto et al.** (SUP_2026_101): correlated features arrange by co-activation; **does not study hierarchy** (confirmed on the page read).
- **Anthropic's Toy Models of Superposition** (SUP_2022_001, level A, not re-read this session): secondary summaries say correlated features get orthogonal directions when possible, otherwise placed close together, and collapse onto a principal component when space runs out. If correct, this is the existing toy-model finding closest to the "shared parent" idea. Re-check against the primary text before citing (the page was too large for the tool).
- **Hyperbolic / tree embeddings (level A, snippets):** "Language Models as Hierarchy Encoders" (NeurIPS 2024, arXiv:2401.11374, RAT_2024_112) re-trains encoder LMs with hyperbolic clustering and centripetal losses so parents sit nearer the origin; HELM (NeurIPS 2025, arXiv:2505.24722, RAT_2025_113) trains fully hyperbolic LLMs from scratch and reports gains up to about 4% over Euclidean architectures on a small corpus (third-party summary: 5B Wikipedia tokens, 1.5 to 1.8x longer training; not verified in the paper); HypStructure (arXiv:2412.01023, RAT_2024_114, level S) is a hyperbolic tree regulariser for classification features. These push hierarchy into the *geometry* with an architectural or loss change, not by sharing a parent direction in Euclidean superposition.
- **Related compression (already indexed):** shared concept codebooks for token embeddings (LMT_2025_009); parent-child linear recoverability (RAT_2026_014).
- **A forum question (level S):** a LessWrong shortform comment asks what geometry a model adopts for features respecting a binary-tree hierarchy; I could not open it, so its content beyond the snippet is unknown.
- **Not found:** a *trained toy model of superposition* with tree-structured ground-truth features at n ≫ m measuring whether children share a parent direction and whether that gives a loss-versus-width gain; any training-time objective that encourages parent sharing to save capacity.

### Q6. Representation compression for quantisation (overlaps with the project)

- **Rotation methods (level A):** QuaRot (Ashkboos et al., arXiv:2404.00456, QNT_2024_101; NeurIPS 2024 as I recall it, not confirmed on the arXiv page; code github.com/spcl/QuaRot) rotates the model so hidden-state outliers vanish without changing the output, enabling 4-bit weights, activations and KV cache (LLaMA-2-70B within 0.47 WikiText-2 perplexity). SpinQuant (Liu et al., ICLR 2025, arXiv:2405.16406, QNT_2025_102; code github.com/facebookresearch/SpinQuant) learns the rotations (narrows the 4-bit W/A/KV gap on LLaMA-2 7B to 2.9 zero-shot points).
- **Why they matter to the project (reasoning, mine):** a rotation changes the coordinate system but not the angles between feature directions. So rotation methods cure *outlier coordinates*, not the *interference between crowded directions* that quantisation noise adds to. Whether a packed (denser) representation is more fragile than an unpacked one under these methods was **not found tested**. Practical rule: apply identical PTQ, rotations included, to model A and model B, so that outlier handling is not a confound in the 2 GB versus 3 GB comparison.
- **Training-time outlier suppression (already indexed):** clipped softmax (QNT_2023_001), Outlier-Safe Pre-Training (QNT_2025_002).
- **Relevant fragility results:** see Q7.
- **No direct link between outlier features and superposition was found**; the only bridge is Elhage et al.'s privileged-basis discussion (not read).

### Q7. Evidence that more superposition hurts, or that the exponent cannot be changed

**Hurts (all indirect; none tests a packing regulariser):**
- Gorton and Lewis (SUP_2025_107, level P): in toy models, adversarial vulnerability rises with features per dimension. Real-model support is proxy-based. Bereska et al. (SUP_2025_108, level A) report the opposite direction can occur depending on regime. So the sign is **contested**.
- Kumar et al., "Scaling Laws for Precision," ICLR 2025 (QNT_2025_103, level A): post-training-quantisation degradation grows with the number of training tokens per parameter, to the point extra data can hurt (fit on 465+ runs, up to 1.7B parameters and 26B tokens). Ouyang et al., ACL 2025 (QNT_2025_104, level A): low-bit quantisation hurts smaller, heavily trained models more; more tokens means more degradation. The "packs more information" intuition is a secondary-summary gloss, not something these papers measured as superposition. But the direction (more efficient use of parameters, so more quantisation damage) is exactly the risk to the project's 2 GB-versus-3 GB criterion.
- Springer et al., arXiv:2503.19206 (QNT_2025_105, level A, preprint): "catastrophic overtraining": extended pretraining makes the model more sensitive to parameter changes and harder to fine-tune (OLMo-1B: more than 2% worse on several benchmarks after 3T versus 2.3T tokens).
- Duan, arXiv:2606.03002 (QNT_2026_003 in the index; abstract verified this session: "Submitted to TMLR", so a preprint): INT7 degrades 18.7% of active SAE features on Gemma-2-2B while perplexity improves; feature survival predictable from peak activation.
- Masip et al., CoLLAs 2026 (SUP_2026_114, level A): forgetting arises when feature capacity shrinks or readout is disrupted; toy model. Only loosely related.
- Henighan et al. (SUP_2023_006, already indexed): storing data points in superposition is the overfitting regime; storing features in superposition is the generalising regime.
- Kong et al., ICML 2026 (SUP_2026_110, level A): **opposite direction**: widening a network with sparser sub-neurons at constant non-zero parameter count raises accuracy by reducing feature collisions; gain largest when interference is highest. Shows that interference is a real cost that can be traded against parameters.

**Exponent cannot be changed:** SUP_2025_002 (level L) says no for natural language; Liu and Gore 2026 (SUP_2026_102) restate that exponents are universal across data and architecture choices and push towards coefficients. Counter-evidence for the general claim that architecture cannot change exponents: Tay et al. 2022 (level S, snippet, venue citation unclear) report architecture changes the scaling slope; and a 2026 preprint (arXiv:2609.19107, listed in a search summary, not opened) tests whether architecture only changes constants. So "exponents are fixed" is a claim about *this* universality class (dense Transformers), not a theorem, but nothing found shows a training-time penalty changing the exponent.

## 3. Citing papers of SUP_2025_002 (Semantic Scholar snapshot, 37 entries)

One call to the citations endpoint returned 37 papers (it may be incomplete). Abstracts read for: 2606.25008, 2609.36455, 2602.01045, 2510.04500, 2605.23591, 2505.22506, 2606.03990. None proposes a superposition-encouraging regulariser. Titles seen but **not opened**: "Emergent Inverse-Depth Scaling From Nonlinearity In Attention" (2610.11063), "Effective Does Not Mean Useful: Conditional Functional Substitutability..." (2609.39259), "Optimizer-dependent training dynamics converge to the same one-third optimal data scaling" (2609.37745), "Emergent One-Third Scaling Law as Attention Tries to Concentrate" (2609.32100), "In-Context Binding Capacity in Language Models" (2609.30634), "Coupled Scaling: A Representational Accessibility Framework for Neural Scaling Laws" (2609.03533), "On the Smallness of the LLM Scaling Exponents" (2606.24504), "The Model Shape Behind Scaling Laws: A Gradient Superposition Perspective" (2605.27989), "Stream separation improves Bregman conditioning in transformers" (2603.21317), "Dissecting Outlier Dynamics in LLM NVFP4 Pretraining" (2602.02047; possibly relevant to quantisation and worth a glance), "Deriving Neural Scaling Laws from the statistics of natural language" (2602.07488), "Inverse Depth Scaling From Most Layers Being Similar" (2602.05970), "Universal One-third Time Scaling in Learning Peaked Distributions" (2602.03685), "On the origin of neural scaling laws: from random graphs to natural language" (2601.10684), "Shared Gradient Discovery & Superposition: Learning Dynamics of Generalization in LLMs" (no ID given). Treat these as unexamined.

## 4. Closeness table

Closeness scale: **identical** (does the same thing), **overlapping** (shares the mechanism or measurement but differs in goal or setting), **adjacent** (same area, different mechanism), **superficial** (shares a word or form only). "To" says which part of our plan it is close to.

| Work | Key | To | Closeness | Level |
|---|---|---|---|---|
| Liu, Liu, Gore, NeurIPS 2025 | SUP_2025_002 | whole idea (names it as untested) | overlapping (it is the base; the idea is an untested hypothesis there) | L |
| Gao et al., ICLR 2019 | RAT_2019_101 | 1a, spread-out penalty on tied embedding rows in LMs | overlapping (same object and direction, different goal, small gains) | P |
| Bank and Giryes, arXiv 2018 (BMVC 2020 per record) | RAT_2018_103 | 1a, coherence-to-ETF-Gram penalty | overlapping (same maths, goal is generalisation, small nets) | P |
| Minimum Hyperspherical Energy, NeurIPS 2018 / CVPR 2020 | RAT_2018_011, RAT_2020_104 | 1a | overlapping | A |
| Wang et al., ICLR 2020 | RAT_2020_102 | 1a (output embedding anisotropy) | adjacent | A |
| Miller, Draye, Schölkopf, COLM 2026 | SUP_2026_004 | 1a form; joint SAE + LM; Q-LIT-4 | overlapping (coherence penalty on SAE dictionary, opposite goal) | P (58%) |
| Lai and StefanHex (blog) | RAT_2025_106 | joint SAE + LM training | overlapping (same mechanism, toy scale, opposite goal) | P |
| Park et al., ICLR 2025 oral | RAT_2025_013 | 1b geometry in real LMs | overlapping (observes parent-plus-offset, no training use) | A |
| Prieto et al., ICLR 2026 poster | SUP_2026_101 | 1b (correlated features in trained toy autoencoders) | overlapping (setup close; no hierarchy, no regulariser) | P (81%) |
| Chanin et al. hedging / absorption; SynthSAEBench | SAE_2025_101, SAE_2024_004, SAE_2026_102 | 1b toy data | adjacent (hierarchical toy data, SAE is the subject, features orthogonal) | A |
| Kong et al., ICML 2026 | SUP_2026_110 | goal (parameters vs interference) | adjacent (reduces interference, opposite direction) | A |
| Gorton and Lewis; Bereska et al. | SUP_2025_107, SUP_2025_108 | measuring superposition; harm | adjacent | P / A |
| Liu and Gore 2026 position paper | SUP_2026_102 | coefficient-versus-exponent framing | adjacent | A |
| Kim et al. 2026 (Muon capacity) | SUP_2026_113 | optimiser as a packing lever | adjacent | A |
| Kumar et al.; Ouyang et al.; Springer et al.; Duan | QNT_2025_103/104/105, QNT_2026_003 | quantisation fragility | adjacent | A |
| QuaRot; SpinQuant | QNT_2024_101, QNT_2025_102 | outlier handling in the 2 GB versus 3 GB test | adjacent | A |
| SAFR | RAT_2025_012 | regularising superposition | superficial (different goal, classification) | A |
| Token Superposition Training; Superposition-in-Transformers MoE; Superposition of many models into one | — | — | superficial (name collision) | S |
| He et al. hierarchy encoders; HELM; HypStructure | RAT_2024_112, RAT_2025_113, RAT_2024_114 | 1b via hyperbolic geometry | adjacent | A / S |
| Hisku blog (orthogonality in a TinyStories GPT) | RAT_2026_105 | 1a in a small LM | adjacent (anti-superposition goal, single runs) | P |

**No work found that is identical to any part of the plan.**

## 5. Negative results (searched, found nothing)

- A regulariser or training method whose stated aim is to **increase** superposition or feature packing in a transformer so a smaller model can match a larger one (searches 1, 2, 5, 33, plus the citation list).
- A frame-potential, Welch-bound, ETF-seeking or squared-coherence penalty applied to the **toy model of superposition** and compared with tuned weight decay (2, 7, 34).
- **Negative weight decay or norm-pulling** on embedding/unembedding rows in a real LM (4, 32, 35). Only SUP_2025_002's toy-model use, and OLMo 2's choice to exclude embeddings from decay.
- A fixed or regularised ETF/Grassmannian unembedding in LM pretraining (15).
- MHE or hyperspherical-energy regularisation used in LM pretraining (26).
- A critique or replication of SUP_2025_002 (27).
- An SAE trained **online and jointly** with an LM from scratch beyond toy scale, with capability results (6, 17, 19, 20).
- A trained toy model of superposition with tree-structured features at n ≫ m (8, 9, 38).
- A training objective that encourages child features to share a parent direction in order to save capacity (8, 10, 38).
- A direct test of "denser packing makes quantisation worse" (22, 28 to 30).
- Any direct link between outlier features and superposition (30).

State of coverage: roughly 40 web-search queries plus abstract/HTML reads, in English, mostly arXiv-indexed material and blogs. Not covered: Google Scholar, direct OpenReview or GitHub search, non-English venues, and anything posted after about 2026-11-13. Absence here means "not found in these searches".

## 6. Honest novelty assessment

### Stage 1a: penalty on feature-direction overlap (frame potential / coherence) vs tuned weight decay

- **Already covered:**
  - the *form* of the penalty (pairwise cosine on tied LM embeddings: Gao 2019; coherence to ETF Gram: Bank and Giryes; hyperspherical energy: Liu et al.; squared off-diagonals of a dictionary Gram matrix: SUP_2026_004; and SUP_2025_002 App. D.2 uses the same squared-off-diagonal objective to draw a figure);
  - the *control knob in the toy model* (weight decay, positive and negative: SUP_2025_002);
  - the claim that spreading directions gives small generalisation gains in vision and translation.
- **Not covered / genuinely open:**
  - whether such a penalty beats **tuned** weight decay in the toy model at n = 1000, measured as the loss-versus-width coefficient;
  - whether it changes only the coefficient (as SUP_2025_002 predicts) and by how much;
  - whether it interacts with norm-pulling (negative decay) rather than substituting for it;
  - any LM-scale test, and any quantised comparison.
- **Verdict:** **not novel as a method; the open part is empirical** and it is a controlled-comparison question, not a new technique. Most likely outcome from the existing evidence: a small coefficient change (the earlier packing_ideas.md estimate of 1.5 to 2.5x headroom in mean squared overlap was read by eye off a figure, and nothing in this search refines it), and the penalty may largely duplicate what the loss already does. A clean negative result ("tuned weight decay already captures it") is plausible and would be a fair, publishable-in-a-workshop-sized finding but not a headline. Do **not** describe 1a as a new method in any write-up; cite Gao 2019, Bank and Giryes, MHE.

### Stage 1b: tree-structured toy features with a shared parent direction

- **Already covered:** hierarchical and correlated toy data for studying SAE failure (Chanin et al., absorption, Matryoshka, SynthSAEBench); real-LM geometry of hierarchy (Park et al., child minus parent orthogonal to parent); correlated-feature geometry in trained superposition autoencoders (Prieto et al.; the original Toy Models, per secondary summary); hyperbolic hierarchy encoders as an architectural route.
- **Not covered / genuinely open:**
  - a trained superposition model on a **tree of features at n ≫ m** where the question is capacity: loss versus width with and without hierarchy, and whether children reuse a parent direction;
  - whether any training objective makes sharing happen more than it already does;
  - how the SAE-measured and ground-truth feature counts behave under hierarchy (links to Q-DES-1).
- **Verdict:** **genuinely less explored than 1a**, because I found no trained toy-model study of it. But the likely finding (from Park and Prieto) is that models already organise correlated or hierarchical features compactly, so the interesting product may be a *measurement* (how much capacity hierarchy gives for free, which would also give a baseline for real LMs) rather than a new training trick. The "shared parent saves capacity" intuition has not been tested by anyone I could find; it is also not supported yet. Mark as exploratory.

### Goal as a whole (more superposition, then 2 GB quantised matches 3 GB)

Not found anywhere. The idea is named as an untested hypothesis in SUP_2025_002, and Liu and Gore's 2026 position paper steers towards coefficients. Evidence on quantisation (Kumar, Ouyang, Duan) says the more efficiently parameters are used, the more quantisation hurts, so the quantised comparison is where the idea is most likely to fail. That is a risk to plan for, not a reason to drop the toy-stage test.

## 7. Papers needing a user-provided PDF

**None required.** All sources above are open access (arXiv, ICLR/NeurIPS/COLM proceedings, blogs). Items I could only read partially or not at all, which are worth a full-text read by a later session (not paywalled):

| Item | What is missing |
|---|---|
| SUP_2026_004 (arXiv:2602.04718) | Last ≈42% of the HTML (appendices, including Llama 3.2 1B results) not read |
| Prieto et al. (arXiv:2603.09972) | Last ≈19% not read |
| Gao et al. 2019 (arXiv:1907.12009); Bank and Giryes (arXiv:1810.06049) | Read only through summarising fetch of the ar5iv HTML; formulas and numbers should be spot-checked in the original |
| Toy Models of Superposition (transformer-circuits.pub/2022/toy_model/) | The correlated-feature section was not reached (page too large for the tool); claims above come from secondary summaries |
| Turner, Wu, Batson (Transformer Circuits, 2026-08-21) | Only the first part read (page truncated) |
| Wang et al. ICLR 2020; MHE NeurIPS 2018 / CVPR 2020; Kumar et al.; Ouyang et al.; Springer et al.; QuaRot; SpinQuant; Kong et al.; HELM | Abstract level only; read methods before relying on any number |
| LessWrong shortform comment on binary-tree feature geometry | Could not open (connection refused) |

## 8. What this means for the plan (proposal only, nothing approved)

1. Frame the project's novelty as a **controlled empirical comparison** (and a measurement of how much hierarchy buys), not as a new regulariser. Cite the existing penalty forms.
2. For 1a, include both a pure overlap penalty and a norm-pulling term (negative decay) as separate arms, since reasoning suggests they push different levers; tuned weight decay stays the mandatory baseline.
3. Add a quantisation-fragility check on the toy model early (quantise W after training and compare loss increase between packed and unpacked solutions), because the real-model fragility evidence all points the unfavourable way.
4. Use the same PTQ pipeline for both models in the final comparison.
5. Q-LIT-4 can be marked answered (see the pointer in open_questions.md).
