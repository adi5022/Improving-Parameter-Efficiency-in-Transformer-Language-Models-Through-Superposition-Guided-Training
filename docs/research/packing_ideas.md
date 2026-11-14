# Ideas for Fitting More Into the Same Space

Session 2, 2026-11-13. Prompted by the user: "any suggestions from existing work that could help me fit more into the space and force my neurons to store more data? Like superposition, maybe a different way."

Status: **candidates only, nothing approved** (CLAUDE.md §9). Evidence levels follow paper_library.md: **L** = full text read, **A** = abstract or search-result summary only (not verified against the full paper).

## 0. What "fit more" can mean — from the full text of SUP_2025_002 (level L, all 37 pages read)

The PDF was supplied by the user (`Research References/Superposition Yields Robust Neural Scaling.pdf`). The paper gives four separate levers, which matter because they point at different methods:

| Lever | What the paper shows | Where |
|---|---|---|
| **A. Geometry** (lower the interference constant) | In strong superposition loss ∝ squared overlaps ∝ 1/m. Feature vectors in trained toy models are ETF-like (equal-angle tight frame: low-variance overlaps), and the Welch bound κ = sqrt((ν−m)/(m(ν−1))) ≈ 1/sqrt(m) is the best possible worst-case overlap. The measured **mean** squared overlap sits **above** the 1/m line at small α, by roughly 1.5–2.5× (read by eye off Fig. 17; approximate), while still scaling as 1/m. That gap is headroom in the coefficient, not the exponent. | Eq. 5, Fig. 5, Fig. 17 |
| **B. Error correction** (tolerate more overlap) | ReLU plus negative biases cancel interference; without the nonlinearity superposition cannot lower loss. The authors suspect the softmax is also important for LLMs. | p. 4, p. 10 |
| **C. How many features get a dedicated direction** | Strongly represented features (‖W_i‖ > 1) number about min(n, m²/2), the ETF ceiling, which is far above m. The rest are "weakly represented" and squeezed into small angles. If the weak ones dominate, loss ∝ m^(−2(α−1)). | A.1, Fig. 8c, Fig. 16 |
| **D. What the paper says is out of reach** | Changing the width exponent for natural language: "No for natural languages", "yes for domain tasks with super skewed feature frequencies". Improvements from encouraging superposition "may be related to altering coefficients ... rather than the exponents". | p. 10 |

Also stated by the authors (p. 10): encouraging superposition "could enable smaller models to match the performance of larger ones". They cite nGPT (unit-norm constraints) and optimisers that work without weight decay (their own FOCUS) as possible examples, but **do not test this**. It is a hypothesis in the paper, not a result.

Not in the paper but standard maths (not looked up this session, to verify before citing): a dimension-m space holds exponentially many (about exp(c·ε²·m)) vectors whose pairwise overlap is at most ε (Johnson–Lindenstrauss-type packing), versus about m²/2 at the minimum possible worst-case overlap. So "store far more" is possible only by **accepting more interference**; the real trade-off is features stored versus interference tolerated, which is lever B.

## 1. Candidate ideas, ranked by how directly they test the user's hypothesis

### 1. Frame-potential / coherence penalty on the feature vectors — strongest candidate for the Stage 1 oracle
- **What:** add λ · Σ_{i≠j} (Ŵ_i·Ŵ_j)² (Ŵ = row-normalised W), or a max-overlap / Riesz-energy version, to the loss. In the toy model the true features are known, so no SAE is needed.
- **Evidence:** SUP_2025_002 itself builds its Fig. 2c illustration by optimising exactly the squared off-diagonals of the normalised W (App. D.2) — level L. Minimum Hyperspherical Energy (Liu et al., NeurIPS 2018, RAT_2018_011, level F for pp. 1-3 and 6-8 as of 2026-11-14) is the general version: a Riesz-energy penalty spreading neuron weight vectors uniformly on the sphere; evidence is vision classifiers only (e.g. CIFAR-100 CNN-9 error 28.13 → 25.61, ImageNet ResNet-18 32.95 → 32.45). The penalty λ‖tril(DᵀD)‖²_F of SUP_2026_004 (level F) is the same frame-potential form, applied to an SAE decoder with the opposite aim (less overlap for cleaner interventions). Not found: either used to push *language-model* feature packing.
- **Why it might work:** targets lever A directly; cheap; our toy code already measures mean squared overlap.
- **Why it might not:** gradient descent on the task loss already optimises overlaps for that loss. The penalty can only help through optimisation (escape bad local packings) or by shifting the trade-off. The expected gain is small (a constant factor), which is exactly what Stage 1 must measure against tuned weight decay.
- **Cost:** hours on the toy model.

### 2. Negative weight decay on a real small LM's embedding and unembedding rows
- **What:** the paper's norm-to-one decay, applied to the LM head rows of a small transformer (nanoGPT-scale), versus tuned plain decay.
- **Evidence:** SUP_2025_002 shows it works in the toy model and observes that real LLMs are already in strong superposition. **The paper does not train LLMs with it** (level L).
- **Why it might work:** it is the cheapest possible "force more superposition" experiment on real language.
- **Why it might not:** LLMs are already strongly superposed (φ½ ≈ 1), so there may be nothing left to push. It also acts only on the output/embedding layer, not on activations inside the network.
- **Role:** this is also the baseline any fancier penalty must beat (S0-4 logic carried to LMs).

### 3. Hypersphere-normalised transformer (nGPT)
- **What:** all embedding, attention, MLP weights and hidden states kept at unit norm, with learnable per-dimension step sizes (Loshchilov et al., ICLR 2025, LMT_2025_006, level A).
- **Evidence:** reports 4–20× fewer training steps to reach the same accuracy (their abstract). Liu et al. name it as consistent with encouraging superposition. **The speedup is in steps, not capacity per parameter, and I saw no independent reproduction.** *Full text read at level P on 2026-11-14:* the factor is in tokens/iterations (4×/10×/20× at 1k/4k/8k context) for 0.5B and 1B models only; time per step is 60–80% higher; only the initial learning rate was tuned for both models; weight decay and warmup were removed for nGPT; there is no sub-0.5B result; and the nGPT paper itself does not discuss superposition (the link is Liu et al.'s).
- **Caveat for the user's plan:** it changes the architecture, not just the training path, which conflicts with "reuse everything, focus on training".

### 4. Data-side: change what is stored, not how
- **What:** Allen-Zhu & Li (ICLR 2025, LMT_2024_005, now level P): ~2 bits of knowledge per parameter **when each fact is seen about 1000 times** (≥ 1 bit at 100 exposures), holding under int8 but **more than halved under int4**; **prepending domain-name-like tokens reduced the capacity lost to junk data from ~20× to ~2× at 100 exposures — shown only inside a 1/8-useful, 7/8-junk mixture, not as a gain on clean data (correction to the abstract-level wording)**; GPT-2 with rotary matches or beats LLaMA/Mistral (gated MLP) at storing knowledge, especially with shorter training. Their abstract lists data signal-to-noise as a factor but I did not read its direction or size.
- **Relevance:** proof that "bits per parameter" is a movable quantity by cheap means, and the right yardstick for Stage 4 (the 2 GB vs 3 GB test). Morris et al. 2025 (LMT_2025_007, level P) give another yardstick (≈ 3.6 bits/parameter; **measured on uniformly random token sequences in ≲ 20M-parameter GPT-2-style models, a lower bound, no capacity number for text**; a measurement framework separating memorisation from generalisation).

### 5. Optimiser choice
- **What:** FOCUS (Liu, Liu & Gore, LMT_2025_008, level A) and Muon-type optimisers. Liu et al. speculate that optimisers that stabilise training without weight decay may help via superposition. Our existing side direction A2 (Muon vs AdamW packing) already covers this; nothing new to approve.

### 6. Shared parent concept plus small child offsets (user idea, 2026-11-14)
- **User's idea:** group "close" concepts (e.g. the heart of many animals) into one shared parent representation, and store each specific case only as a small difference, so the generic part is not stored again for every child. Their stated assumption: this is a best-case sketch and needs checking.
- **Prior work found (level A, summaries only):**
  - Park et al., ICLR 2025 (RAT_2025_013; full text read at level P on 2026-11-14): in Gemma-2B (LLaMA-3-8B in an appendix), parent-child concepts are represented so that the child-minus-parent difference is orthogonal to the parent direction (957 WordNet synsets). **Scope limit: this is shown for the final-layer unembedding (output-token) representation after whitening; the authors say internal-layer geometry is open.** So trained models already do some "shared part + difference" at the output, but not shown inside the network, which is where our packing question lives.
  - Feature absorption and splitting (SAE_2024_004) and Matryoshka SAEs (SAE_2025_005): the same hierarchy shows up as a nuisance for SAEs; Matryoshka SAEs are built around nested parent/child dictionaries.
  - Kavin & Goyal, EMNLP Findings 2025 (LMT_2025_009): token embeddings compressed with shared codebooks of "concept vectors" to ~0.4-0.5% of embedding parameters with >95% task performance on some encoder models. Post-hoc compression of the input embedding table, not a training-time penalty on internal features.
  - A 2026 arXiv paper (RAT_2026_014, arXiv:2604.07886, authors not recorded): parent-child relations are linearly recoverable from LM representations in a low-dimensional subspace. Analysis only.
- **Not found:** a training-time loss that pushes a model to share parent representations more in order to save capacity. One standard search, not exhaustive (Q-LIT-2).
- **Reasoning (mine, not from a paper):** in the hidden vector a child is parent + offset, so sibling arrows overlap by design and can blur into each other unless the offsets are large enough. The saving is more likely in the weights: generic knowledge about the parent is stored once and reused by all children. Both parts need testing.
- **Cheap test:** a toy model whose features form a tree (a parent is on whenever any of its children is on), unlike the independent features of Stage 0. Known tree = ground truth. WordNet could supply a real tree later.
- **Risks:** the model may already share as much as is useful; real concept hierarchies are messy; the gain may be small.

## 2. Looked for and not found, or low trust

- **No paper found** in which a superposition-encouraging regulariser lets a smaller model match a larger one (one standard search, not exhaustive; Q-LIT-2 stays open).
- SAFR (arXiv:2501.16374, level A): regularises superposition, but for interpretability: monosemantic on important tokens, polysemantic on correlated pairs. Different goal.
- A LessWrong post "Superposition and Dropout" claims dropout leaves room for more superposition. Blog post, not peer-reviewed; treat as a hint only.

## 3. Suggested use in the plan (proposal, not approved)

- Stage 1 candidate 1b (user idea 6, tree-structured toy features) was added to research_roadmap.md on 2026-11-14 with the user's go-ahead; design still to be proposed.
- Stage 1 oracle penalty candidates: (i) frame potential, (ii) max-coherence, (iii) a ground-truth "feature utilisation" bonus. Each compared with tuned γ and with seed noise from Stage 0.
- Stage 0 add-on (small): report mean squared overlap × m per γ, so the headroom in the constant (lever A) is measured on our own runs instead of read off a figure.
- Stage 4 yardstick: Allen-Zhu & Li synthetic knowledge tuples.
- New decision needed before Stage 1: which penalty family to try first (frame potential recommended).
