# Paper Library

Full bibliographic records for every source examined so far. Claims are indexed in
[citation_index.md](citation_index.md); topic syntheses are in [papers/](papers/).

Last updated: 2026-11-14 (Q-LIT-1 verification pass, see "Verification pass 2026-11-14" below).

## Verification levels (read this before trusting any entry)

| Code | Meaning |
|---|---|
| **L** | Local copy read directly from disk (user's own work, or a PDF the user supplied). |
| **F** | PDF pages rendered and read page by page by Claude (added 2026-11-14). The pages covered are stated in each entry; text, equations, tables and figures were seen directly. Stronger than P. Not a substitute for the authors' appendices where those are listed as not read. |
| **P** | Paper HTML full text fetched and read through a summarising fetch tool. Usually the first ~100k characters were covered (introduction, method, experiments, main results); appendices and some limitations sections may be missing. Numbers were extracted by the tool and **must be spot-checked against the PDF before being quoted in a paper.** The tool sometimes garbles tables and does not always show the venue. |
| **A** | Abstract and arXiv/venue metadata only. **Methodology and experimental findings NOT verified.** |
| **S** | Search-result snippets or secondary summaries only. **Treat as unverified pointers.** |

Venue notes: where an entry says "venue per arXiv comments (session 1), not visible in full text", the full text/HTML that was read did not show the venue, so it was neither confirmed nor refuted. Entries are only marked "confirmed" where the venue appears in the paper itself.

Every central paper (marked ★) still needs a full-text read before it is used to justify an approved decision (tracked in [open_questions.md](open_questions.md), Q-LIT-1). Remaining unread ones are listed in [reading_log.md](reading_log.md) under "Needs user-provided PDF" and "Not reached".

## Verification pass 2026-11-14 (Q-LIT-1) — summary

Upgraded to F (PDF pages read): SUP_2026_004, SAE_2025_002, SAE_2025_003, SAE_2025_009, RAT_2018_011 (pp. 1-3, 6-8), SUP_2022_001 (pp. 3-17 only). Upgraded to P (HTML full text via summariser): SAE_2024_010, SAE_2024_004, SAE_2025_005, SAE_2025_006, SAE_2024_001, SAE_2024_011, SAE_2025_012, RAT_2024_006, RAT_2025_013, SUP_2022_005, LMT_2025_006, LMT_2024_005, LMT_2025_007, QNT_2023_001, QNT_2025_002, QNT_2026_003, KD_2025_001 (App. A and B/C only). New records: SUP_2022_008 (Jermyn et al.), SAE_2025_014 (Chanin & Garriga-Alonso), LMT_2025_010 (Muon is Scalable). Claims corrected: see the "Correction" lines in the individual entries (SUP_2026_004 scope, SAE_2025_002 venue, QNT_2023_001 clipped softmax on OPT, QNT_2025_002 kurtosis source, QNT_2026_003 title and meaning of 18.7%, RAT_2024_006 "no degradation", LMT_2024_005/LMT_2025_007 conditions, RAT_2025_013 unembedding-space caveat).

Peer-review status is taken from the arXiv comments field or the proceedings page. "Not stated" means the arXiv page lists no venue; it does not mean the paper was rejected.

---

## Superposition and representational capacity

### SUP_2022_001 — Toy Models of Superposition
- Authors: N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, et al., C. Olah
- Year / venue: 2022, Anthropic technical report (Transformer Circuits Thread); arXiv preprint
- Peer review: **Not peer-reviewed** (technical report)
- ID: arXiv:2209.10652 — https://arxiv.org/abs/2209.10652 — open access
- Code: none official
- Level: **F, partial** (upgraded 2026-11-14 from A: arXiv PDF pages 3-17 rendered and read, i.e. summary, definitions, "Demonstrating Superposition", "Mathematical Understanding", "Superposition as a Phase Change", start of "The Geometry of Superposition"/"Uniform Superposition". **Not read:** the rest of the geometry section (polytopes, tegum products, non-uniform case), correlated/anti-correlated features, superposition in a privileged basis, computation in superposition, adversarial examples, appendices.)
- Main findings (abstract): polysemanticity can arise when sparse features are stored in superposition; phase change between superposed and non-superposed representations; geometry linked to uniform polytopes; evidence linking superposition to adversarial examples.
- Verified in the read pages: (1) model h = Wx, x' = ReLU(WᵀWx + b) ("ReLU output model") versus the linear model x' = WᵀWx + b; loss L = Σ_x Σ_i I_i (x_i − x'_i)², features zero with probability S, otherwise uniform on [0,1] (Experiment Setup / The Model / The Loss). Note this is h = **Wx**, not the Wᵀx convention of SUP_2025_002. (2) Basic results with n = 20, m = 5, I_i = 0.7^i and n = 80, m = 20, I_i = 0.9^i: the linear model learns only the top-m features at every sparsity (PCA-like); the ReLU model matches it when dense, and as sparsity rises stores more features by making them non-orthogonal, first as antipodal pairs (Basic Results figures). (3) Heuristic loss decomposition L ~ Σ_i I_i (1 − ‖W_i‖²)² [feature benefit] + Σ_{i≠j} I_j (W_j·W_i)² [interference] for the linear case; for the ReLU model, interference with a negative bias is "free" in the 1-sparse case, and uniform importance gives a generalised Thomson problem (Mathematical Understanding). (4) Phase diagrams for n = 2, m = 1 and n = 3, m = 2: the extra feature is not represented, stored in superposition, or given a dedicated dimension, depending on sparsity and relative importance, with a first-order (discontinuous) transition in the theory (Superposition as a Phase Change). (5) Uniform superposition (n = 400, m = 30): the number of features learned is measured as ‖W‖²_F, "dimensions per feature" D* = m/‖W‖²_F is "sticky" at 1 and 1/2 (antipodal pairs), and the authors say the number of input features n does not matter provided n ≫ m (Uniform Superposition).
- Authors' caveats seen in the read pages: "very unclear what to generalize to real networks"; the feature-geometry results are "at least partly idiosyncratic to the toy model".
- Relevance: the conceptual foundation of the README's §3. Practical for this project: ‖W‖²_F as a rotation-invariant "number of represented features" and D* as "dimensions per feature" are candidate answers to Q-DES-1 in the toy model (ground truth only, not for real LMs). Also: with a ReLU-free (linear) readout superposition does not pay, so any toy setup must keep the nonlinearity.

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
- Check 2026-11-14: the supplied PDF is arXiv v4 (29 Nov 2025) and prints "39th Conference on Neural Information Processing Systems (NeurIPS 2025)" on page 1, which confirms the venue; the "Best Paper Runner-up" note comes from the arXiv page and is not visible in the PDF. Fig. 1d (page 2) shows toy models in strong superposition and actual LLMs both on a slope of about −1. The other numbers (α_m = 0.91 ± 0.04, 0.88 ± 0.06) were read on 2026-11-13 and not re-read in this pass.
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
- Level: **F** (upgraded 2026-11-14 from A: arXiv v5 PDF, pages 1-10 main text and pages 14-20 appendices A-C read. Venue confirmed: "Published as a conference paper at COLM 2026" in the page header.)
- Abstract: bounds the propagation of feature interference by the self-coherence of the feature dictionary; adds orthogonality regularisation on the dictionary; enables more isolated interventions on mathematical-reasoning concepts "while preserving model performance".
- **What the "dictionary" is (answers Q-LIT-4):** the **decoder matrix D of a TopK SAE**, not the language model's own weights. Eq. 1 (Sec. 3.2): ℓ(x) = ‖x̂ − x‖₂/‖x‖₂ + λ‖tril(DᵀD)‖²_F. Pipeline (Sec. 5): (1) fine-tune a pre-trained TopK SAE (Gao et al.; width 2^16 = 65,536, inserted after layer 13 of 26 of Gemma 2 2B) with the orthogonality penalty, penalty evaluated at each step on a random subset of 2^10 = 1,024 active features for cost reasons; (2) freeze the SAE, splice it into the residual stream (all information flows through it), and LoRA-fine-tune the **LM** around it (rank 128, 1 epoch of MetaMathQA, LR 5e-5). Also Llama 3.2 1B (limitations paragraph). λ ∈ {0, 1e-6, 1e-5, 1e-4}. **So it is fine-tuning, not pretraining; the penalty never touches the LM's packing of its own features, and the LM is made to adapt to the SAE (the same direction as SAE_2025_002).**
- Theory (Sec. 4, Thm. 4.1, App. B): under a linear/exponential-family causal model, the expected mismatch between the true and the model's post-intervention output is bounded by a baseline term plus δ²(d−1)μ(D) (mean self-coherence). The authors' own numerical check (App. B.3, Fig. 6, d = 12, m = 8) shows the bound is **not tight**, by orders of magnitude.
- Results: (a) the penalty does lower the orthogonality loss monotonically in λ (Fig. 2). (b) GSM8K accuracy ≈ 0.55-0.60 for all λ, bootstrap intervals overlapping, "no significant difference"; the authors report only "a slight performance increase" which is within noise (Fig. 3); the 2B model is below the 0.665-0.777 range cited for 7B MetaMath models. (c) Name-swap interventions on 12 first-name features, 3,960 examples: reasoning accuracy unchanged across λ (Fig. 4a); the intended name appears in the output 74.9% of the time at λ = 1e-4 versus 62.2% at λ = 0 (Fig. 4b, Sec. 5.2). (d) Interference measures (Table 1): ROUGE-L recall 0.763 (λ = 0), 0.746 (1e-6), 0.730 (1e-5), 0.773 (1e-4); JSD 0.222, 0.254, 0.235, 0.214. **The two intermediate λ are worse than λ = 0; only the strongest penalty is better. The authors say the intermediate values "do not yield definitive conclusions".** (e) Auto-interp score 0.37-0.47 across λ, no significant difference (App. C.3, Fig. 8). Compute: ~8 h (SAE) + ~16 h (LoRA) on 4×80 GB GPUs (App. C.1).
- Limitations (authors, Sec. 6): two off-the-shelf SAEs, middle layers only, mathematical reasoning only; the intervention changes a feature that is not an ancestor of the final answer. Statistics are bootstrap over test examples; **no repeated training runs / seeds** are reported. The authors themselves list as an open follow-up "whether enforcing greater feature orthogonality reduces superposition", i.e. they did not test it.
- Relevance: the closest found work to "control interference by optimisation", but (i) it acts on an SAE decoder, (ii) in fine-tuning, (iii) with the goal of intervenability, and (iv) its direction (reduce interference/raise orthogonality) is **opposite** to this project's goal (more superposition). It is also another case where the LM adapts to a frozen SAE. It does not overlap with gap G1 (SAE-derived loss on the LM's own weights during pretraining), and it shows the penalty λ‖tril(DᵀD)‖² is the same frame-potential form as packing_ideas.md idea 1.

---

## Sparse autoencoders and their reliability

### SAE_2024_001 — Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning
- Authors: Dan Braun, Jordan Taylor, Nicholas Goldowsky-Dill, Lee Sharkey
- Year / venue: 2024; arXiv page lists no venue (believed NeurIPS 2024 — **unverified**)
- ID: arXiv:2405.12241 — open access
- Level: **P** (upgraded 2026-11-14 from A; HTML v2 read to ~100k of ~120k characters; appendices F-H truncated; some numbers garbled by extraction; venue not visible in the text, so "NeurIPS 2024" stays unverified)
- Abstract: train SAEs to minimise KL between original and SAE-spliced model outputs; Pareto improvement (more performance explained, fewer total and active features).
- Details (Sec. 2-3): SAE_local = MSE + L1; SAE_e2e = KL(logits) + L1 with the model frozen; SAE_e2e+ds adds a downstream-activation MSE term. GPT-2 small, residual stream before layer 6 (also layers 2, 10), dictionary 60× the residual width (46,080), 400k samples of 1024 tokens of OpenWebText; also TinyStories-1M. Claim: at equal L0 the CE-loss increase of e2e is under 45% of local's (L0 50-300, Sec. 3.1 fn. 3); Table 1 (layer 6, similar CE increase ~0.14): local L0 69.4 / 26k alive, e2e L0 27.5 / 22k alive, e2e+ds L0 36.8 / 15k alive. **Costs and caveats the library did not record:** 2-3.5× longer runtime (Sec. 3.1); e2e SAEs reconstruct activations and directions worse, with strong feature suppression (L2 norm ratio 0.31 vs 0.92 for local, App. B, Table 5); **e2e features did not replicate across seeds, local and e2e+ds did** (Sec. 3.3.2); interpretability not significantly different at similar L0 (p = 0.61/0.71/0.33) and slightly better for e2e+ds at similar CE (Sec. 3.4). Loss weights (0.5 on KL, β values) were not heavily tuned (App. D).
- Relevance: SAE objectives that use the model's output — the inverse direction of this project (SAE adapts to model). Note for the toy plan: the e2e variant that trains the SAE against the model's output was the least seed-stable of the three.

### SAE_2025_002 ★ — Low-Rank Adapting Models for Sparse Autoencoders
- Authors: Matthew Chen, Joshua Engels, Max Tegmark
- Year / venue: 2025, **ICML 2025** (the PDF footer reads "Proceedings of the 42nd International Conference on Machine Learning, Vancouver, PMLR 267, 2025") — **correction:** earlier recorded as "arXiv preprint (no venue listed)"; it is peer-reviewed.
- ID: arXiv:2501.19406 — open access; code https://github.com/matchten/LoRA-Models-for-SAEs
- Level: **F** (upgraded 2026-11-14 from A: PDF pages 1-9 read, i.e. all main text, Tables 1-5, Figs. 1-9; references/appendices not read)
- Abstract: freeze a trained SAE and LoRA-fine-tune the *language model* so that splicing the SAE costs less; CE-loss gap reduced 30–55%; reaches e2e-SAE downstream CE 3–20× faster (Gemma), 2–10× (Llama); no harm to general capabilities reported.
- Verified details: loss = KL between the original logits and the SAE-spliced logits, only LoRA adapters trained, SAE and base model frozen (Sec. 3.4, Eq. 7). Gemma Scope JumpReLU SAEs (adapt only layers after the SAE so L0 is unaffected) and own TopK SAEs (k = 64) on Gemma-2-2B and Llama-3.2-1B; **only 15M tokens of The Pile** for the Gemma Scope sweeps. CE-gap closed 30–55% across sparsity, width, rank, layer (Fig. 3); larger effects at low width/low L0 and in larger models (Fig. 4, Gemma 2B/9B/27B). Speed (Tables 1, 2): 3.05×-20.01× faster than e2e SAE on Gemma-2-2B; 2.23×-10.38× on Llama-3.2-1B. **Not uniformly better:** on Llama the e2e SAE reaches a lower final CE than TopK+LoRA (Sec. 4.2, Limitations); e2e SAEs were not run to convergence (a week). Multiple SAEs at once: compound CE for 7 SAEs 7.83 → 2.78 nats (Fig. 6). General capability (Table 5): "SAE+LoRA" beats "SAE" on MMLU/HellaSwag etc. and **the adapted model without any SAE is at least as good as the original** (Gemma-2-2B MMLU 49.3 → 50.0 ± 0.4, HellaSwag 55.0 → 56.0 ± 0.5), so no capability harm, but note this is LoRA fine-tuning on extra data. SAEBench (Table 3): LoRA model improves SCR, TPP, sparse probing; autointerp ≈ equal (0.830 vs 0.832); absorption slightly *worse* (0.210 vs 0.205, higher = worse). 37% of validation tokens get worse loss (Fig. 8). A LoRA on only the first layer after the SAE gets 88.14% of the benefit of adapting all layers (Sec. 6.3, Fig. 9).
- Authors' limitations (Sec. 7.1): interpretability work usually studies a frozen model; here the model is changed; no evidence yet that circuits found are more faithful.
- **Prior art found in this paper (relevant to Q-LIT-2):** Sec. 2 states two earlier works, Lai & Heimersheim (2024) and Lai & Huang (2024), "train SAEs and models concurrently" and find this improves the spliced loss, but "only shown to work in toy settings"; this paper claims its own method is ~10⁷× cheaper than those (authors' own estimate, footnote 2, based on 6T tokens vs 15M tokens × 3% of parameters). Neither Lai paper was located or read (see reading_log.md).
- Relevance: **the closest existing case of "modify the LM's weights with an SAE-derived loss".** It shows the model can be made more SAE-decomposable cheaply; it does not claim the model becomes more capable. It is also direct evidence for the "gaming the ruler" risk in D-09: a model can be made to fit a fixed SAE better without any claim about the model's own computation.

### SAE_2025_003 ★ — Sparse Autoencoders Trained on the Same Data Learn Different Features
- Authors: Gonçalo Paulo, Nora Belrose
- Year / venue: 2025, arXiv preprint
- ID: arXiv:2501.16615 — open access
- Level: **F** (upgraded 2026-11-14 from A: PDF pages 1-8 read, main text, Figs. 1-6, Table 1; appendix figures A1-A3 not read). Venue: the PDF is in an ICML-style template but no venue is printed; treat as preprint.
- Abstract: changing only the SAE seed, only ~30% of features shared (131k-latent SAE on Llama-3-8B MLP); ReLU+L1 SAEs more seed-stable than TopK.
- Verified: a latent counts as shared if it has the same Hungarian-matched counterpart in both the encoder and decoder matchings **and** cosine ≥ 0.7 in both (Sec. 3). Pythia-160M layer-6 MLP, 2^15 latents, 8B tokens: only **42%** shared by that definition (Sec. 3); with 9 seeds the share of latents found in only one SAE falls from ~58% (2 SAEs) to ~35% (9 SAEs) and appears to plateau (Fig. 2, Sec. 4). The "30%" is Llama-3-8B (131k latents, MLP). Overlap falls with more latents (Fig. 4 left) and more active latents k (Fig. 4 middle) and rises with training tokens (Fig. 4 right: 0.8B → 80B tokens, roughly 7% → 60% aligned for the 32k SAE, read by eye); overlap is high for small SAEs (32 latents ≈ 80-100% in Fig. 4). Gated and L1 SAEs overlap more than TopK on GPT-2 (Fig. 5, right). Orphan (unshared) latents are often as interpretable as shared ones (Sec. 5, Fig. 6, Table 1): average detection score 0.72 for the 32k SAE.
- Limitations/gaps: only Hungarian matching on cosine similarity; no analysis of whether the **number** of features (a count) is stable across seeds, only identity; SAEs not trained with the reconstruction-in-the-loop regime that this project would use. The authors' own interpretation: SAE features are "a pragmatically useful decomposition", not a canonical list; lack of universality due to nonconvexity of the SAE loss (Sec. 7).
- Relevance: an SAE-derived training signal is partly an artefact of one SAE run. Any loss computed from one SAE inherits this noise. **For a toy model with a *small* SAE (tens to hundreds of latents) the instability is much weaker (Fig. 4), and more training raises stability**, which supports using a small, long-trained SAE as the ruler. Whether a feature *count* is more stable than feature identity is still untested (R8 question).

### SAE_2024_004 — A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders
- Authors: David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, Joseph Bloom
- Year / venue: 2024 (v6), **NeurIPS 2025 oral** (arXiv comments)
- ID: arXiv:2409.14507 — open access
- Level: **P** (upgraded 2026-11-14 from A; arXiv v6 HTML, text truncated inside Appendix A.3. The venue "NeurIPS 2025 oral" comes from the arXiv comments (session 1) and is **not visible in the full text**.)
- Abstract: features that look monosemantic fail to fire when a more specific child feature absorbs them; caused by sparsity optimisation on hierarchical features; varying SAE size or sparsity does not fix it.
- Verified: definition of a hierarchy (f2 implies f1) in Sec. 2. Toy model (Sec. 3): four orthogonal 50-dim features, feature 0 fires with p = 0.25 and features 1-3 with p = 0.05, child features only when the parent is on; 4-latent SAE, L1 3e-5. Absorption lowers the L1 loss whenever p(parent and child) > 0 (App. A.2), shown for a simplified setup (orthogonal features, no biases); this shows absorption is **favoured by the sparsity penalty**, not that it is the only outcome. Real-model measurement (Sec. 4-5): first-letter task on Gemma-2-2B with Gemma Scope residual SAEs (16k, 65k widths, layers 0-17), plus own SAEs on Qwen2 0.5B and Llama 3.2 1B; **absorption rate rises with SAE width and with higher sparsity (lower L0)** (Fig. 9b caption). The paper contains no table of rates and **tests no mitigation** (Matryoshka etc. are later work). Limitations (authors): the metric is a likely underestimate, thresholds set by manual inspection, layers capped at 17, few non-JumpReLU SAEs.
- Relevance for this project: any SAE trained on features with parent-child co-occurrence (e.g. the tree-structured toy of Stage 1 candidate 1b) will mis-count features by design; absorption **under-counts** parent features (the parent fails to fire in child contexts) while splitting **over-counts**.

### SAE_2025_005 — Learning Multi-Level Features with Matryoshka Sparse Autoencoders
- Authors: Bart Bussmann, Noa Nabeshima, Adam Karvonen, Neel Nanda
- Year / venue: 2025, arXiv preprint (no venue on page)
- ID: arXiv:2503.17547 — open access
- Level: **P** (upgraded 2026-11-14 from A; arXiv v1 HTML; several results exist only in figures and could not be read; the page lists "ICML" as a keyword but acceptance is not confirmed there)
- Abstract: nested dictionaries reduce absorption; better probing and concept erasure; minor reconstruction cost. Includes TinyStories experiments.
- Verified: loss = sum of reconstruction MSEs over nested prefixes (Eq. 5; Gemma prefixes {2048, 6144, 14336, 30720, 65536}, equal weights, BatchTopK), Gemma-2-2B layer 12, 500M Pile tokens, dictionary 65,536, L0 20-320 (Sec. 4.3). At L0 = 40: variance explained ~70% vs 72% for BatchTopK; first-letter absorption 0.05 vs 0.49; about 1 vs about 3 latents per letter (splitting). Meta-SAE variance explained 42% vs 55% (App. D.1). TinyStories (4-layer, d = 768) and a 20-latent toy are qualitative only. **Costs/limits:** ~50% longer training; no clear gain on Chess-GPT (App. E), degradation at L0 ≥ 150; stop-gradients and weighted losses hurt (App. G); evaluation relies on automated metrics. Absorption/probing numbers other than those above are figure-only.
- Relevance: with SAEBench (SAE_2025_006) it is the best-supported fix for absorption; relevant to the tree-structured toy.

### SAE_2025_006 — SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability
- Authors: Adam Karvonen, Can Rager, Johnny Lin, et al., Samuel Marks, Neel Nanda
- Year / venue: 2025, **ICML 2025** (arXiv comments)
- ID: arXiv:2503.09532 — open access
- Level: **P** (upgraded 2026-11-14 from A; arXiv v4 HTML, appendices E-J partly truncated; page lists "ICML" as keyword, acceptance not shown in the text)
- Abstract: eight metrics, >200 SAEs, eight architectures; gains on unsupervised proxy metrics do not reliably translate to practical performance.
- Verified (Sec. 3-4): metrics = loss recovered, automated interpretability, sparse probing (35 tasks), feature absorption, unlearning, SCR, TPP, RAVEL, plus L0. Architectures named in Table 1 by the fetch tool: ReLU, Matryoshka BatchTopK, TopK, BatchTopK, Gated, JumpReLU, P-Annealing (seven names; the abstract-level record said eight, not resolved — spot-check Table 1), Gemma-2-2B layer 12 and Pythia-160M layer 8, widths 4k/16k/65k, L0 20-1000. The "proxy metrics do not reliably predict" claim (Sec. 4.1) rests on **rankings and examples, not reported correlation coefficients**. Matryoshka is best on 5 of 8 metrics at L0 40-200 but trails TopK on the sparsity-fidelity frontier; for non-hierarchical SAEs absorption and SCR get **worse with width** ("inverse scaling", cause hypothesised to be feature splitting); a middling L0 of 50-150 is suggested as compromise. Limitations: ground-truth metrics cover only a fraction of latents; sparse probing noisy; two models only; metrics should not be merged into one score.
- Relevance: if this project uses "SAE quality" as an outcome, SAEBench metrics are the reference, and the width/absorption inverse scaling means a *count* of SAE latents grows with width for reasons unrelated to the model.
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
- Level: **F** (upgraded 2026-11-14 from A: PDF pages 1-9 read, main text Sec. 1-6 and Figs. 1-5; appendices B-L not read; venue not printed in the PDF — preprint)
- Abstract: SAEs on randomly initialised Pythia models get auto-interp and reconstruction scores similar to SAEs on trained models.
- Verified: per-layer TopK SAEs (expansion 64, k = 32, 100M RedPajama tokens) on Pythia 70M-6.9B; variants = trained, re-randomised incl./excl. embeddings, Step-0 checkpoint, and a Gaussian-embedding control (Sec. 3). Fuzzing/detection AUROC and reconstruction metrics of the randomised variants are close to the trained model (Figs. 1-2), far from the control; the gap between trained and randomised variants is larger for **smaller** models and layers and narrower for bigger ones (Related work, Fig. 2). Explained variance/cosine similarity are **often lower for the control** than for the other variants. CE loss recovered only makes sense for the trained model. Token-distribution entropy of latent activations is lower for randomised models (latents are more token-specific) and, unlike in trained models, does not rise with layer depth — the authors' own proxy for "abstractness" (Sec. 3, Fig. 2 bottom row; preliminary). **Sec. 4 (toy models) is relevant here:** (a) in Sharkey-style toy data an SAE recovers ground-truth features only in a "Goldilocks" band of L1 coefficient, with mean-max-cosine similarity ≈ 1 inside and ≈ 0.2 outside, and the band moves with the number of features n = 512…4096 (Fig. 4); (b) passing superposed inputs through a **randomly initialised** MLP preserves or even increases SAE-measured sparsity (Fig. 5), i.e. random networks "preserve superposition" (hypothesis only; the authors defer which cause predominates).
- Limitations (Sec. 5): Pythia and RedPajama only; default EleutherAI auto-interp pipeline; the authors "do not claim that SAEs fail" on trained transformers, only that aggregate auto-interp metrics do not show that interesting features exist.
- Relevance: **"more interpretable features" measured by auto-interp is not evidence of better learned representations.** Every interpretability metric in this project needs a random-init / untrained-checkpoint control. The Fig. 4 result is a direct warning for the toy ruler: SAE feature recovery depends on the sparsity coefficient, so the SAE's L1/k must be calibrated against ground truth before any feature count is trusted (see also SAE_2025_014).

### SAE_2024_010 — Scaling and Evaluating Sparse Autoencoders
- Authors: Leo Gao, Tom Dupré la Tour, et al. (OpenAI)
- Year / venue: 2024, **ICLR 2025** (proceedings link seen in search)
- ID: arXiv:2406.04093 — open access
- Level: **P** (upgraded 2026-11-14 from S; arXiv v1 HTML read to ~100k of ~119k characters; **no venue shown in the page**, "ICLR 2025" comes from a session-1 search result and is unverified here)
- Verified details: TopK SAE z = TopK(W_enc(x − b_pre)), plain MSE loss, no L1 (Eq. 2, Sec. 2.3); inputs mean-subtracted and unit-normalised, context 64, layer 5/6 of GPT-4-family or layer 8 of GPT-2 small (Sec. 2.1); b_pre = geometric median of the data, encoder initialised parallel to the decoder (App. A.1; Sec. 2.4 says "transpose" — inconsistent in the paper); decoder columns unit-norm after every step; Adam β = (0.9, 0.999), constant LR (the largest converging LR ∝ 1/√n), batch 131,072 tokens (App. A.3-A.5). AuxK: L_aux = ‖e − ê‖² with ê built from the top k_aux (≈ 512 = power of two near d_model/2) **dead** latents, total loss L + α·L_aux with α "typically 1/32"; "dead" = no activation for ~10M tokens; ~10% extra compute; without mitigations "up to 90% dead latents", with AuxK + tied init the 16M-latent GPT-4 SAE (40B tokens) has ~7% dead (Sec. 2.4, App. A.2, B.1; no clean with/without number at 16M). Scaling (Sec. 3, Eq. 3): joint fit of MSE in (n, k) with fitted coefficients (α = −0.50, β_k = 0.26, β_n = −0.017, γ = −0.042, ζ = −1.32, η = −0.085 as extracted); tokens-to-convergence ∝ n^0.6-0.65; laws break down as k → d_model (~512 for GPT-2 small); larger subject models need more latents for the same MSE. Multi-TopK (Sec. 5.3) makes the code progressive over k. Limitations (Sec. 6): TopK forces exactly k latents per token; optimisation, evaluation and auto-interp (N2G) are incomplete; 64-token context; features "not yet adequately monosemantic".
- Relevance: standard modern SAE recipe if we train our own SAEs. Reading caveat: AuxK was needed because dead latents were a real problem at scale; for a continuously retrained online SAE both the TopK constraint and AuxK are extra moving parts that must be tested in the toy first (compare R7/R8).

### SAE_2024_011 — Tracking the Feature Dynamics in LLM Training: A Mechanistic Study (SAE-Track)
- Authors: Yang Xu, Yi Wang, Hengguan Huang, Hao Wang
- Year / venue: 2024 (v3 Jun 2025), arXiv preprint
- ID: arXiv:2412.17626 — open access
- Level: **P** (upgraded 2026-11-14 from A; arXiv v3 HTML; venue not shown in the page)
- Verified: SAE[k] is initialised from SAE[k−1] and trained on checkpoint k's activations (Sec. 2.3); ReLU-encoder L1 SAE with unit-norm decoder (App. B); Pythia-410M-deduped layer 4 (154 checkpoints, steps 0-143k), plus 160M/1.4B and Stanford CRFM GPT-2 small/medium (App. I, K). SAE[0] uses 300M tokens, later ones 5M-15M each (~1/20 of the initial). **Not reported in the text:** dictionary size, LR, optimiser, λ, and any numeric reconstruction/L0 values or compute saving (claims are qualitative, from figures). Findings: token-level features are present early, concept-level features form gradually; feature directions keep drifting after features are "formed" (Sec. 3, 5.1, Conclusion 3.2). Authors' own caveats (App. H, A): strict feature tracking is impractical (SAE training is a selection process that can diverge), semantic labelling is manual, mid-scale models only; they call it "a study tool rather than an engineering evaluation framework".
- Relevance: method for keeping an SAE current while the model trains — the core engineering problem of "SAE in the training loop". **It shows warm-starting an SAE from the previous one is workable for analysis, but not that it is stable enough to drive a gradient signal, and it contains no quantitative check of ruler quality over time.**

### SAE_2025_012 — Evolution of Concepts in Language Model Pre-Training
- Authors: Xuyang Ge, Wentao Shu, Jiaxing Wu, Yunhua Zhou, Zhengfu He, Xipeng Qiu
- Year / venue: 2025, **ICLR 2026** (arXiv comments, session 1; **not shown in the page text read 2026-11-14**)
- ID: arXiv:2509.17196 — open access
- Level: **P** (upgraded 2026-11-14 from A; arXiv v2 HTML, appendices truncated)
- Abstract: crosscoders across pretraining checkpoints; most features form around a specific point; complex features later; two stages (statistical learning, then feature learning); feature evolution causally linked to downstream performance.
- Verified: one cross-snapshot crosscoder (shared sparse code, one decoder per checkpoint), "feature evolution" read off decoder norms (Sec. 3); Pythia-160M (layer 6, 98,304 features) and Pythia-6.9B (layer 16, 32,768 features), 32 of 154 checkpoints, SlimPajama. Feature formation begins around step ~1,000 (Sec. 4.1). Two-phase claim (Sec. 6, Fig. 8): unigram/bigram KL converges early and total "feature dimensionality" first falls then rises to ~70% of available dimensions — described by the authors as a rough division and an analogy to fitting-then-compression, not a tested prediction. "Causal link" = integrated-gradient attribution + ablation on three simple tasks (subject-verb agreement, induction, IOI) with 1,000 samples each; numeric effect sizes are figure-only. Limitations (Sec. 7): Pythia only, simple tasks, cost linear in snapshots.
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
- Level: **P** (upgraded 2026-11-14 from A; arXiv v4 HTML to ~100k of ~151k characters; Appendix B and E truncated; no venue shown in the page)
- Abstract: sparse dictionary learning built into end-to-end MoE pretraining, 262,144 experts per layer; knowledge manipulation "without degrading general performance".
- Verified: 512 × 512 = 262,144 experts per layer by product-key composition with horizontal or vertical decomposition (Sec. 3); 850M/1.4B/4.1B total parameters vs dense LLaMA 770M/1.3B/3.8B ("total-parameter matched", sizes not exactly equal), 100B tokens (Table 2). **Correction to the abstract-level claim "without degrading general performance":** in Table 2 average accuracy Monet trails the dense baseline at every scale, 0-shot 0.453/0.478/0.511 (VD) vs 0.463/0.484/0.520, 5-shot 0.485/0.510/0.550 vs 0.492/0.518/0.567; the HD variant is lower still. The gaps are small (~0.006-0.017) and the authors call it "competitive", but it is not parity, and no perplexity is reported. Monosemanticity is shown by hand-picked examples (Fig. 2), not a metric; knowledge manipulation is unlearning only (domain/language/toxicity masking, Sec. 5).
- Relevance: another "interpretable by design" pretrained architecture that is slightly worse than a size-matched dense baseline; consistent with the pattern in papers/representation_learning.md §1(b).

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
- Additions 2026-11-14 (level P, arXiv HTML characters 100k-200k read: bibliography and Appendices A-D.4): **Appendix A limitations as stated by the authors:** language modelling only; English C4 only (large runs repeat data); downstream evaluation weak (tracks cross-entropy); only full-distribution (Hinton) distillation studied, not sequence-level or hard distillation ("important future work"); transformer teacher and student only; teacher and student share one data distribution; the law's coefficients are setup-specific and not derived from first principles. App. B.2-B.3: λ = 1.0 only, and the optimal temperature is 1 (contrary to earlier literature, attributed to non-repeated data). App. C: broken power law, the capacity gap appears when the teacher becomes stronger than the student can use; a kernel-regression proof of a U-shape (C.1.3) and MLP experiments (C.2.2). **Students shown in the read portion are 1B-30B; nothing in the read parts addresses students under 100M** — the paper's smallest students (143M) are above our target. Still not read: the scaling-law equation and fitted coefficients (Eq. 8, App. F), the truncation study (G.4), sensitivity analyses G.1-G.3.
- Relevance: argues against "biggest available teacher" and quantifies when KD can help a small student. Our <100M, narrow-domain regime is outside the range the law was fitted on.

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
- Level: **P** (upgraded 2026-11-14 from S; arXiv v2 HTML to ~100k of ~141k characters; Appendix C (training budgets) and low-bit appendix B.7 not read; **venue not shown in the page**, NeurIPS 2023 remains unverified)
- Content: outliers arise from heads learning "no-op" updates through softmax; clipped softmax and gated attention reduce outlier magnitude/kurtosis and improve W8A8 quantisation while keeping or improving FP performance (BERT, OPT, ViT).
- Verified (Sec. 3-6, Table 2, Table 3): evidence for the mechanism is observational (BERT outliers sit in a few hidden dimensions, >97% on delimiter tokens; ViT >99% in 10 dimensions), not a causal ablation. Clipped softmax = clip((ζ−γ)·softmax(x)+γ, 0, 1) with γ < 0 (Eq. 4); gated attention = per-head sigmoid gate (Eq. 5, <0.009% extra parameters for BERT-base). Setup: BERT-base (109M) MLM, OPT-125m CLM, ViT-S/16 (22M); 8-bit PTQ; each PTQ run repeated 3 times, each network trained twice. Table 2 (mean ± std): BERT ppl FP 4.49 vanilla / 4.39 clipped / 4.45 gated, W8A8 **1294 ± 1046** / 4.52 / 4.65; OPT-125m FP 15.84 / **16.29** / 15.55, W8A8 21.18 / **37.20** / 16.02; ViT acc. W8A8 69.24 / 79.77 / 79.82. Scaling (Table 3, gated attention only): OPT-350m W8A8 14.42 vs 37.52 vanilla, OPT-1.3B 29.95 vs 989.6.
- **Correction:** the earlier one-line claim "clipped softmax / gated attention reduce outliers and improve W8A8 with equal or better FP performance" holds for gated attention on all three models and for clipped softmax on BERT/ViT, but **clipped softmax made OPT-125m worse in both FP (16.29 vs 15.84) and W8A8 (37.2 vs 21.2)**; the authors list this failure without explanation (Sec. 6). The FP gains are small and not claimed to generalise.
- Limitations (Sec. 6): scale tested only to OPT-1.3B; mainly 8-bit PTQ; an extra hyperparameter per method.
- Relevance: **established precedent that a training-time representation intervention improves quantisation robustness** (at 8 bit, ≤1.3B). It is a precedent, not evidence for 4-bit or sub-100M behaviour.

### QNT_2025_002 ★ — Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models (OSP)
- Authors: Park et al.
- Year / venue: 2025, **ACL 2025** (aclanthology 2025.acl-long.618)
- ID: arXiv:2506.19697 — open access; code https://github.com/dmis-lab/Outlier-Safe-Pre-Training
- Level: **P** (upgraded 2026-11-14 from S; arXiv v1 HTML; **Table 2 is garbled in the HTML and its columns were realigned by row pattern — spot-check against the PDF**; venue not shown in the page, "ACL 2025" unverified here)
- Content: Muon + single-scale RMSNorm + learnable embedding projection; 1.4B model, 1T tokens; W4A4 average 35.7 vs 26.5 (Adam); excess kurtosis 0.04 vs 1818.56; ~2% overhead; Muon alone insufficient.
- Verified (Sec. 3-4, Tables 1-5): SSNorm(x) = γ·x/‖x‖₂ with a single scalar γ (Eq. 3); EmbProj = a learnable full-rank projection after the embedding and before the unembedding (orthogonal init); Muon on hidden matrices, Adam kept for embeddings. 1.4B LLaMA-style model only (no smaller size); ablations on a **100B-token** subset, main run 1T tokens; Adam LR 5e-3, Muon LR 5e-4, no LR sweep described, **no seeds/variance**. **Correction of provenance:** the kurtosis 1818.56 → 0.04 comes from the 100B-token ablation (Table 2), not from the 1T-token model. Ablation, 100B tokens, 4-bit W4A4KV4 without Hadamard (avg accuracy / PPL): Muon 29.0 / 1e4; Muon + SSNorm 36.4 / 44.2; Muon + EmbProj 30.4 / 114.6; all three 37.5 / 19.6 — **so SSNorm contributes most of the gain**, and "Muon alone is insufficient" is true but understates that. "Table 3 (1T tokens, 4-bit)": 26.5 → 35.7 average over 10 tasks; the bit-widths of weights/activations/KV are not specified in the caption. **Full-precision cost:** 100B: 41.5 vs 41.4 average (PPL 11.4 vs 11.2); 1T unquantised (Table 5): 45.9 (Adam) vs 44.4 (OSP), OSP lower on 9 of 10 tasks. Overhead: 97.9% of Adam's tokens/s on TPU-v4-512 (Table 1). Limitations (authors): only Muon (no Shampoo/SOAP), only 1.4B.
- Relevance: the strongest published recipe for 4-bit-robust pretraining; any quantisation-robustness claim from this project has to be compared with it.

### QNT_2026_003 — Perplexity Can Miss SAE Feature Damage Under Quantization (earlier recorded as "How Quantization Changes Interpretable Features: A Sparse Autoencoder Analysis of Language Models")
- Author: Evan Duan (University of Michigan), single author
- Year / venue: 2026, arXiv preprint (v2, 4 Jun 2026 per the page); no venue, not peer-reviewed as far as shown
- ID: arXiv:2606.03002 — open access
- Level: **P** (upgraded 2026-11-14 from S; arXiv v2 HTML read. **Title correction:** the v2 page is titled "Perplexity Can Miss SAE Feature Damage Under Quantization"; the title recorded in session 1 was possibly the v1 title — not checked.)
- Content: Pythia-70M-deduped (public `pythia-70m-deduped-res-sm` SAE, layer 4) and Gemma-2-2B (Gemma Scope layer-12 residual SAE, width 16,384); simulated round-to-nearest, per-output-channel, **weights only** (Sec. 3.2); feature "survives" if the Pearson correlation between FP16 and quantised activations is > 0.9, "degraded" if 0.5-0.9, "damaged" if < 0.5; "active" = firing rate above 0.001 (Sec. 3.3-3.4).
- Numbers (Table 1): Gemma INT7 81.26% survived, 18.74% degraded, **0.0% damaged**, perplexity −5.65% (chunked) / −6.49% (sliding window); Gemma INT6 51.30% survive, Pythia INT6 62.39%; INT8 98.31% (Pythia) / 99.09% (Gemma). **Clarifications:** the "18.7% degraded" are partially aligned features (correlation 0.5-0.9), none were destroyed; the perplexity *decrease* at INT7 appears only for Gemma (Pythia INT7: survival 86.32%, perplexity +2.26%); Gemma FP16 perplexities quoted in the main table (458.5) look very high for this model, so the evaluation protocol is unusual and should be checked. Survival predicted from four FP statistics (rarity, log mean, log peak, log concentration) by L2-regularised logistic regression: 5-fold AUC 0.924 ± 0.007 (Pythia), 0.971 ± 0.002 (Gemma); peak activation strongest (Sec. 4.2, Table 2); transfer across models not tested.
- Limitations: two models, public SAEs, simulated RTN, one layer per model, **single deterministic runs, no confidence intervals**; the paper's text and its Table B.3 disagree on subset variation (0.71 vs 1.41 points) and appendices C/H contain unexplained inconsistent perplexities and a Pythia INT7 change reported as −5.96% in one place and +2.26% in another. Treat the headline numbers as indicative.
- Relevance: perplexity alone does not show features survive quantisation — directly relevant to README §8 "preservation of learned representations". It supplies a measurement protocol, not a result about training-time interventions.

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
- Level: **P** (upgraded 2026-11-14 from A; arXiv v4 HTML (25 Mar 2025); venue not shown). Verified: capacity C_i = (W_i·W_i)² / Σ_j (W_i·W_j)² ∈ [0, 1], Σ_i C_i ≤ D (Sec. 2.1, App. F). The **analytic** model is a two-layer network with **quadratic** activations and ground truth y = Σ v_i x_i² (Sec. 3.1-3.2); sparsity enters through the kurtosis k = E[x⁴]. Optimal allocation C_i = clip(((k−1)/(k−3))·v_i/λ − 2/(k−3), 0, 1) (Eq. 15-18): a feature is ignored if (k−1)v_i/2 < λ, gets a full dimension if v_i > λ, is partially represented in between; higher kurtosis (sparser data) widens the partial region (Sec. 3.4-3.6). Numerical check: 6 features / 3 neurons phase diagram qualitatively matches (Fig. 4); ReLU and GeLU results compared with Anthropic's model (Fig. 7) have **no analytic backing**, and training details/seeds/error bars are not given on the page. Authors' caveats: only the quadratic-activation model with diagonal ground truth is solved; features are identified with input dimensions; some derivative claims "not checked" (App. G).
- Answer to reading-list R4's question: the theory says optimal training **already** allocates capacity by importance and sparsity; it gives no mechanism by which a penalty could create capacity that the loss does not already reward, only conditions (heavier tails/kurtosis, importance below λ) where partial allocation is optimal. It does not say gradient descent always reaches that optimum.

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
- Venue: ICLR 2025 (proceedings.iclr.cc, seen 2026-11-13). **Not shown in the arXiv v1 text read 2026-11-14.**
- Level: **P** (upgraded 2026-11-14 from A; arXiv v1 HTML read to ~100k of ~182k characters, appendices B-G not visible)
- Verified (Sec. 2-4, 8, 10; Results 1-12): data = synthetic biographies bioS(N) (six attributes per person; names from a pool of 400×400×1000 candidates), bit-complexity bounds (Prop. 2.3, Thm. 3.2), capacity ratio R(F) = bits stored / parameters, loss-based (Def. 4.3). **Result 1: peak R(F) ≥ 2 for GPT-2 (rotary, no dropout) from 1M to 0.5B parameters when every fact is seen 1000 times; Result 4: ≥ 1 bit/parameter at 100 exposures ("no more than 2× loss").** So "2 bits/parameter" is a **best case at ~1000 exposures**, now confirmed for the first time. Quantisation (Result 8, GPTQ only): int8 negligible loss of capacity, **int4 reduces capacity by more than 2× (about 0.7 bit/parameter)**. Architecture: LLaMA/Mistral ≈ GPT-2 at 1000 exposures, up to 1.3× lower at 100 exposures, attributed to gated MLP (Results 5-7); MoE with 32 experts loses 1.3-1.5× capacity relative to total parameters (Result 9). Junk-data setting (Sec. 10): with only 1/8 useful data and 7/8 junk, useful-knowledge capacity falls ~20× at 100 exposures (3×, 1.5×, 1.3× at 300/600/1000 exposures); **prepending a domain-name-like token reduces the 100-exposure loss to ~2× (Result 12)**. **Correction to packing_ideas.md and the abstract-level note:** the domain-name benefit was demonstrated **as a mitigation inside a junk-data mixture**, not as a general increase of capacity on clean data.
- Limitations (stated in text): synthetic, idealised data; only GPTQ tested; knowledge shown to be extractable by QA fine-tuning but reasoning not assessed. No separate limitations section was visible.
- Relevance: the right yardstick for the Stage 4 fact-storage test, but the headline number needs ~1000 exposures per fact. The paper notes bioS reaches that within a single pass of the data because each person has many re-sampled biographies, so the number does not transfer directly to ordinary web text where a fact is seen a few times.

### RES_ARENA — ARENA curriculum, chapter 1.3 (Superposition & SAEs exercises)
- Type: educational material (not a paper); mirrored at https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/ ; official site arena.education (current version not checked). Level S.

### Code reference — liuyz0/SuperpositionScaling (official code for SUP_2025_002)
- Files read 2026-11-11: `exp/exp-10.py`, `exp/exp-10-3.py`, `exp/adamw.py`. **No licence.** Toy settings recorded in phases/stage_0_toy_baseline.md §2. Notable: no random seeds set; negative weight decay implemented as W += lr·γ·W·(1 − 1/‖W_i‖) row-wise.

## Added 2026-11-14 (Q-LIT-1 verification pass; new records, level P)

### SUP_2022_008 — Engineering Monosemanticity in Toy Models
- Authors: Adam S. Jermyn, Nicholas Schiefer, Evan Hubinger. Year / venue: 2022, arXiv preprint, no venue shown (v1, 16 Nov 2022). arXiv:2211.09169 — https://arxiv.org/abs/2211.09169 — open access. Level: **P** (arXiv HTML).
- Setup: two-layer ReLU/GeLU network with k hidden units (default 1024), d = 64 input dimensions, N = 512 features, tasks = feature decoder, random re-projector, absolute-value calculator; a neuron is "monosemantic" if the ratio of its peak to summed activation exceeds 0.999 (an arbitrary threshold, Sec. 3).
- Findings (Sec. 4-5): models at different learning rates reach the **same loss** with very different numbers of monosemantic neurons (Fig. 3-4); a strongly negative initial bias (with slow bias-only weight decay) steers training into monosemantic basins (30-50% of neurons, ~0.6-1 per feature, Fig. 8); more hidden units give more monosemantic neurons; monosemanticity peaks near feature probability 2^-5 and falls as N·ε approaches d; for the decoder task monosemantic solutions are roughly equal in loss at k ≤ 512 and lower for k > 512, for the absolute-value task they are lower (Sec. 4.3). Compared with L1 on activations once: lower loss for similar monosemanticity (Fig. 14).
- Limitations (authors): toy only, fragile (bias decay must be slow relative to learning), single seed per configuration as far as the text shows, several interpretations "weakly held".
- Relevance: **direct evidence that, in toy models, how much superposition/polysemanticity a model ends up with is path dependent and steerable by initialisation and learning-rate/decay choices, at unchanged loss.** That supports the idea of training-time control of the degree of superposition, and also means Stage 0 results must report the initial bias and learning rate. It pushes toward monosemanticity, the opposite of this project's direction.

### SAE_2025_014 — Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders
- Authors: David Chanin, Adrià Garriga-Alonso. Year / venue: 2025 (v4 stamped Jul 2026), arXiv preprint, no venue shown. arXiv:2508.16560 — https://arxiv.org/abs/2508.16560 — open access. Level: **P** (arXiv HTML to ~100k of ~136k characters; proofs partly truncated).
- Method: toy models with known ground-truth features, correlated Bernoulli firing; "true L0" is known by construction (e.g. 5 features, d = 20, true L0 = 2; 50 features, d = 100, true L0 = 11; App. E.1 adds superposition with d = 40).
- Findings: if the SAE's L0 is **too low**, latents mix in components of correlated features and **every** latent is affected; the mixed solution reaches lower MSE than the ground-truth SAE (e.g. 2.73 vs 4.88 at L0 = 5, Sec. 3.3) and a proof shows the disentangled solution is not even a local minimum below the true L0 (Thm. F.1-F.2); too-high L0 gives degenerate mixtures, though many latents stay correct; SAEs started at too-high L0 recover when L0 is lowered, those started too low do not (Sec. 3.5). A proposed diagnostic, the mean absolute decoder cosine similarity c_dec, is minimised at the true L0 in toy models (Fig. 7, Fig. 16 with superposition) and gives a broad shallow minimum on LLM SAEs (Gemma-2-2b layer 5/12, Llama-3.2-1b layer 7; Fig. 9-10).
- Limitations (authors): linear representation assumption, idealised orthogonal ground-truth features in the main toy, few LLM layers, "cursory" Neuronpedia evidence for the claim that most deployed SAEs use too-low L0, automatic L0 tuning left as future work. The page excerpt did not discuss whether SAE feature **counts** are reliable.
- Relevance: **for the toy ruler, the SAE's sparsity level must be matched to the true L0; a mis-set L0 produces wrong features regardless of how well the SAE reconstructs.** In the toy model the true number of active features per sample is known, so the SAE can be calibrated against ground truth; with real LMs it cannot.

### LMT_2025_010 — Muon is Scalable for LLM Training
- Authors: not recorded (the page text read did not give them; attribution to Moonshot AI is from the Moonlight model name and is unverified); arXiv:2502.16982 v1 — https://arxiv.org/abs/2502.16982. Year: 2025, preprint, no venue shown. Level: **P** (arXiv HTML).
- Verified: two changes to Muon: decoupled weight decay (Eq. 3) and a per-matrix update scale 0.2·√max(A, B) that matches AdamW's update RMS (Eq. 4, Lemma 1; 0.2 and 0.4 perform similarly, App. A, Table 8). Muon is applied to matrix parameters; embeddings, LM head and RMSNorm gains stay on AdamW. Scaling experiment (Sec. 3.2, 399M-1.5B non-embedding parameters, ~8.9B-38.9B tokens): fitted loss 2.506·C^−0.052 (Muon) vs 2.608·C^−0.054 (AdamW), from which "about 52% of the FLOPs" is derived; **AdamW was grid-searched, Muon reused AdamW's hyperparameters; proprietary data; ≤ 1.5B; single runs, no variance** (Table 1 uses a 4B-token checkpoint of a 20B-token run). Moonlight (2.24B active/15.29B total MoE, 5.7T tokens) beats comparable models on most listed benchmarks but is lower on BBH at 1.2T tokens (43.2 vs 45.3) and below Qwen2.5-3B on GSM8K. Muon-SFT on Qwen2.5-7B underperformed Adam-SFT (Table 7). Max attention logits exceeded 100 early in training and AdamW controlled them better (App. D). SVD entropy of Muon-trained matrices is higher (flatter spectra) — authors' hypothesis for "more diverse update directions", one checkpoint.
- Relevance: the optimiser behind the OSP quantisation result and a baseline-tuning issue: the "~2× efficiency" is a fit over ≤ 1.5B models, not shown below 100M. Our toy and sub-100M runs cannot assume it.

## Not yet examined (to read; not cited as evidence anywhere)

Identified as relevant but not opened. Do not cite until examined. (Jermyn et al., Scherlis et al., Muon is Scalable and Allen-Zhu & Li were examined on 2026-11-14 and moved to full records above.)

| Topic | Work | Identifier (to verify) | Why |
|---|---|---|---|
| Superposition | Henighan et al. — Superposition, Memorization, and Double Descent | Transformer Circuits 2023 | Data-limited regime + superposition (page is very large; not attempted) |
| SAE prior art | Lai & Heimersheim (2024); Lai & Huang (2024) — train SAEs and models concurrently (cited by SAE_2025_002) | not located in two searches | The closest prior art to joint SAE+model training; toy settings per SAE_2025_002 |
| SAE | Olmo et al. — Features that Make a Difference: gradient SAEs (g-SAE) | arXiv:2411.10397 (NAACL 2025 per arXiv listing) | SAE objective that uses model gradients; abstract only, not read |
| Regularisation | Cogswell et al. — Reducing Overfitting by Decorrelating Representations (DeCov) | arXiv:1511.06068 (ICLR 2016) | The classical non-MI baseline for a decorrelation loss |
| SAEs | Bricken et al. — Towards Monosemanticity | Transformer Circuits 2023 (PDF in FeatureScalpel/docs) | Foundational SAE method |
| SAEs | Cunningham et al. — SAEs Find Highly Interpretable Features | arXiv:2309.08600 (ICLR 2024) | Foundational |
| SAEs | Rajamanoharan et al. — JumpReLU SAEs | arXiv:2407.14435 | SAE architecture choice |
| SAEs | Marks et al. — Sparse Feature Circuits | arXiv:2403.19647 | Attribution with SAE features |
| SAEs | Dunefsky et al. — Transcoders; Lindsey et al. — Crosscoders | arXiv:2406.11944; Transformer Circuits 2024 | Alternatives to residual SAEs |
| Training | Hoffmann et al. — Chinchilla | arXiv:2203.15556 | Token/parameter budget |
| Training | Biderman et al. — Pythia | arXiv:2304.01373 | Checkpointed reference suite |
| Training | Muon optimiser original writeup (Jordan blog) | keller jordan blog | Optimiser choice; ("Muon is Scalable" now examined, LMT_2025_010) |
| Training | Geiping & Goldstein — Cramming | arXiv:2212.14034 | Single-GPU pretraining |
| Training | Wortsman et al. — Small-scale proxies for large-scale instabilities | arXiv:2309.14322 | Stability at small scale |
| KD | Hinton et al. 2015; Kim & Rush 2016 (sequence-level KD) | arXiv:1503.02531; arXiv:1606.07947 | Foundations |
| KD | Cross-tokenizer KD (e.g. Universal Logit Distillation) | arXiv:2402.12030 | Needed if teacher vocab ≠ student vocab |
| Quantisation | LLM.int8(); GPTQ; AWQ; Massive Activations | arXiv:2208.07339; 2210.17323; 2306.00978; 2402.17762 | PTQ methods and outlier theory |
| Side directions | tmsq (toy superposition + RTN quantisation, GitHub); Muon vs Adam features; Anthropic crosscoders; distilled-reasoning crosscoders; SAE-FD | github.com/fqybaba/tmsq; arXiv:2606.09658; transformer-circuits.pub/2024/crosscoders; arXiv:2503.03730; arXiv:2605.25525 | Located at snippet level for alternative_directions.md A1–A4 |

## Added 2026-11-13 for packing ideas (level A: search-result summaries only; NOT verified against full text)

### RAT_2018_011 - Learning towards Minimum Hyperspherical Energy
- Authors: Weiyang Liu, Rongmei Lin, Zhen Liu, Lixin Liu, Zhiding Yu, Bo Dai, Le Song. Year / venue: 2018, **NeurIPS 2018 (confirmed: "32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montréal" in the PDF footer)**. arXiv:1805.09298 (v9) - https://arxiv.org/abs/1805.09298
- Level: **F, partial** (upgraded 2026-11-14 from A: PDF pages 1-3 and 6-8 read; pages 4-5 and 9+ (variants, geometry discussion, appendices) not read).
- Idea: regulariser that spreads neuron directions evenly on the unit sphere (Thomson problem); half-space and angular variants. Verified: energy E_s = Σ_{i≠j} f_s(‖ŵ_i − ŵ_j‖) with Riesz kernels f_s(z) = z^(−s) (s = 0 gives the log energy), objective = data loss + λ_h·(hidden-layer energy, averaged over N(N−1) pairs) + λ_o·(output-layer energy) (Eq. 1-4, Sec. 3); **the penalty acts on the normalised weight vectors of neurons/filters/classifier rows, not on activations or learned feature dictionaries**; asymptotic uniformity on the sphere (Thm. 1-2, Sec. 4.1) and a gradient-based generalisation argument for one-hidden-layer nets (Sec. 4.2).
- Experiments (Sec. 5): **vision only** (CNN-9, ResNet-32 on CIFAR-10/100, ResNet-18/34 on ImageNet-2012, class-imbalance MNIST/CIFAR, face recognition). Examples: CIFAR-100 CNN-9 test error 28.13 baseline → 25.61 (half-space MHE, s = 2) (Table 1/4); ImageNet top-1 error ResNet-18 32.95 → 32.50 (MHE) / 32.45 (half-space), ResNet-34 30.04 → 29.60 / 29.50 (Table 6), i.e. gains of ~0.5 points on ImageNet vs ~2.5 points on small CIFAR networks; CNN-15 does not converge without MHE but converges with it (Table 3). Hyperparameter λ is insensitive over 1e-2…1e2 (Fig. 3). Seeds/variance: not stated in the pages read.
- Relevance: general form of a "pack directions better" penalty (packing_ideas.md idea 1). Not found: used for language-model feature packing. Caveats for our use: it regularises weight vectors, and its evidence is generalisation of vision classifiers (the paper says half-space MHE cannot be applied to the output layer, where the classifier neurons must occupy the full space); the toy version for this project (penalising overlaps of the learned **feature** directions W_i) is a reasonable adaptation, not something the paper tested.

### LMT_2025_006 - nGPT: Normalized Transformer with Representation Learning on the Hypersphere
- Authors: Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, Boris Ginsburg. Year / venue: 2025, ICLR (peer-reviewed). arXiv:2410.01131 - https://arxiv.org/abs/2410.01131
- Claim (abstract): unit-norm embeddings, weights and hidden states with learnable per-dimension step sizes; 4-20x fewer training steps to the same accuracy depending on sequence length. No independent reproduction seen. The speedup is in steps, not capacity per parameter.
- Level: **P** (upgraded 2026-11-14 from A; arXiv v2 HTML, essentially complete (final 1.5k characters unread); venue not shown in the page — "ICLR" unverified here).
- Verified: embeddings and all attention/MLP matrices are unit-normalised along the embedding dimension after each step; hidden-state update h ← Norm(h + α_A(h_A − h)) with learnable per-dimension "eigen" learning rates α (init 0.05, Eq. 10-11); scaling factors s_qk, s_u, s_ν, s_z; **LayerNorm/RMSNorm, weight decay and warmup removed** (Sec. 2.6, Table 3). The 4×/10×/20× speedups are for 0.5B and 1B models at 1k/4k/8k context, **measured in tokens/iterations, not wall-clock: nGPT's time per step is about 80% higher at 4k and 60% higher at 8k** (unoptimised kernels, Sec. 3.1 footnote, App. A.5). Fairness: only the initial LR was tuned for both models, with the sweep reduced "to minimize the total compute"; GPT uses weight decay 0.1 and 2000 warmup steps; one nGPT-only tweak (α_init 0.1) for the 1B/8k run. Downstream comparison is shown only as figures; **no table of GPT vs nGPT downstream numbers in the text**. **No results below 0.5B parameters**; the authors say the longest runs saturate (attributed to near-reached capacity) and note their open-source re-implementation reproduces results only qualitatively. **The paper does not discuss superposition** — the link between nGPT and "encouraging superposition" is made by SUP_2025_002 (p. 10), not by the nGPT authors. Embeddings are observed to cluster rather than spread uniformly on the sphere (Sec. 3.2).
- Relevance: the speedup is in steps at ≥ 0.5B; no evidence for capacity per parameter or for sub-100M models.

### LMT_2025_007 - How much do language models memorize?
- Authors: John X. Morris et al. (Meta, Google DeepMind, Cornell, NVIDIA). Year: 2025, arXiv:2505.24832 (peer-review status not checked) - https://arxiv.org/abs/2505.24832
- Claim: ~3.64 bits/parameter for GPT models in half precision; capacity measured by separating unintended memorisation from generalisation; an empirical estimate, not a theoretical limit.
- Level: **P** (upgraded 2026-11-14 from A; arXiv v3 HTML; no venue shown in the page; the page's own date line looks wrong, so treat version metadata cautiously).
- Verified (Sec. 2-5, Table 1, Fig. 1, Fig. 6): memorisation in bits is estimated as code length from likelihoods with a reference model (the paper describes the reference model inconsistently: larger same-architecture model in Sec. 2.3, equal-size in Sec. 4). **The 3.6 figure comes from training small GPT-2-style models on uniformly random token sequences (vocabulary 2048, length 64, 10⁶ steps, five seeds), not from text**; Table 1 means are 3.83 bits/param (fp32) and 3.51 (bf16), the 3.64 figure being a fitted constant for half precision (Fig. 6). Model sizes quoted inconsistently (Table 1 gives 1-8 layers, widths 32-256, roughly 100K-20M parameters). The authors say it is a **lower bound**, and slightly above the ~2 bits/param of LMT_2024_005. Text experiments (FineWeb) give no capacity number; they show a double-descent onset when dataset capacity exceeds model capacity (a proposed explanation, not a tested result). The membership-inference scaling law (Sec. 5, c₁ = 1.34, c₂ = −0.034, c₃ = −33.14) appears internally inconsistent as extracted (limit and range of the sigmoid); not verified.
- Relevance: gives a ceiling-style yardstick on synthetic random data in tiny models; applying 3.6 to modern text-trained models is extrapolation.

### LMT_2025_008 - FOCUS: First Order Concentrated Updating Scheme
- Authors: Yizhou Liu, Ziming Liu, Jeff Gore. Year: 2025, arXiv:2501.12243 (preprint) - https://arxiv.org/abs/2501.12243
- Claim (abstract): Signum plus an attraction term toward a moving average; faster than Adam and more stable than Signum on GPT-2. Cited by SUP_2025_002 as an optimiser that works without weight decay.

### RAT_2025_012 - SAFR: Superposition-Aware Feature Regularization
- arXiv:2501.16374 (preprint; authors not recorded). Regularises superposition via a modified loss for interpretability (monosemantic on important tokens, polysemantic on correlated pairs). Different goal from ours. Level A.

## Added 2026-11-14 for the hierarchy idea (level A: search-result summaries only; NOT verified against full text)

### RAT_2025_013 - The Geometry of Categorical and Hierarchical Concepts in Large Language Models
- Authors: Kiho Park, Yo Joong Choe, Yibo Jiang, Victor Veitch. Year / venue: ICLR 2025 (peer-reviewed; earlier ICML 2024 workshop version). arXiv:2406.01506 - https://arxiv.org/abs/2406.01506
- Claim (summary): hierarchically related concepts are orthogonal in a precise sense; categorical concepts are polytopes; tested on Gemma and LLaMA-3 with 900+ WordNet concepts. Code: github.com/KihoPark/LLM_Categorical_Hierarchical_Representations.
- Level: **P** (upgraded 2026-11-14 from A; arXiv v3 HTML to ~100k of ~114k characters; **venue not shown in the page, "ICLR 2025" is from session-1 search results and unverified here**).
- Verified: hierarchical orthogonality (Thm. 8): ℓ̄_w ⟂ ℓ̄_z − ℓ̄_w for z ≺ w, plus three contrast-vector variants; categorical concepts = convex hulls (polytopes), simplices for "natural" concepts (Def. 7, Prop. 9). Vectors are estimated by an LDA-like estimator on **whitened unembedding vectors** with Ledoit-Wolf shrinkage (Sec. 5.1). Gemma-2B main (vocabulary 256k, d = 2048), LLaMA-3-8B in Appendix F "largely analogous"; 593 noun + 364 verb WordNet synsets with ≥ 50 words, 70/30 split. Cosines between parent and child-minus-parent are near 0 versus random-parent baselines (Fig. 5; individual values only in figures); adding a parent contrast changes parent logits (plant→animal 5.13 ± 1.17) and leaves child logits near zero (mammal→reptile −0.06 ± 1.22) (Table 1).
- **Correction / scope limit (important for packing_ideas.md idea 6):** the orthogonality is established for the **final-layer unembedding (output-token) representation**, not for the residual-stream activations inside the network; the authors say the whitening "only works for the final layer representation" and call internal-layer geometry open. Theorems assume exact representations, which they acknowledge are unlikely under gradient descent. Two models, single-token words only.

### LMT_2025_009 - Breaking Token Into Concepts: Extreme Compression in Token Representation via Compositional Shared Semantics
- Authors: Kavin R V, Pawan Goyal. Year / venue: 2025, EMNLP Findings. arXiv:2509.17737 - https://arxiv.org/abs/2509.17737
- Claim (summary): product-quantisation-style shared codebooks (Aggregate Semantic Grouping) compress embeddings to 0.4-0.5% of embedding parameters with >95% performance retained on mBERT, XLM-R, mT5 tasks. Post-hoc, embedding table only.

### RAT_2026_014 - Linear Representations of Hierarchical Concepts in Language Models
- arXiv:2604.07886 (preprint; authors not recorded). Claim (summary): parent-child relations are linearly recoverable in a low-dimensional (~150-250 dim), largely domain-specific subspace. Analysis, not a training method.
