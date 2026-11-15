# Stage 2 Sources (records for the Stage 2 design proposal)

Written 2026-11-15 as input to [stage_2_design_proposal.md](stage_2_design_proposal.md). **Not merged into [citation_index.md](../citation_index.md) or [paper_library.md](../paper_library.md); the lead merges.** Nothing here is an approved decision.

## 0. How to read this file

**Verification levels (same scheme as paper_library.md, reused exactly):**

| Code | Meaning |
|---|---|
| **L** | Local copy read directly from disk (user's own work or a user-supplied PDF). |
| **F** | PDF pages rendered and read page by page. Pages covered are stated in the record. |
| **P** | Paper HTML (or web page) full text fetched and read through a *summarising* fetch tool. Usually the first ~100k characters; appendices and tail sections may be missing. Numbers were extracted by the tool and **must be spot-checked against the PDF before being quoted in a paper.** |
| **A** | Abstract / listing / metadata only. Methodology and results NOT verified. |
| **S** | Search snippets or secondary summaries only. |

**Peer review:** "Peer-reviewed" only where the source itself or its listing shows it. A page that shows only an arXiv identifier is called a preprint. For several papers the fetched page did not show the venue; where the project's existing records name a venue from arXiv comments, that is noted as unverified.

**Citation keys.** New keys use a **201+ numbering block** (`SAE_2023_201` ...) so they cannot collide with keys added elsewhere. Categories `OPT` and `MATH` are new (optimisation theory; mathematical background). Existing keys are reused where the paper is already in the index.

**Author lists marked (†)** were not shown in the page text the fetch tool returned; they are given from memory and must be checked on the arXiv page before citing. Author lists without the mark were extracted from the page.

**Counts (this session):** 26 sources opened (14 new records in section 1, 12 existing records re-opened in section 2). Existing records cited without re-opening: 8 (section 3). Level tally of the 14 new records is at the end of section 1: P = 10, F = 1, A = 3. Of the 12 re-opened records, all 12 were read at level P this session (SAE_2025_003 keeps its index level F from the earlier PDF read; SAE_2025_101, SAE_2026_102 and SUP_2025_108 are upgraded from A to P).

---

## 1. New source records

### SAE_2023_201 — Sparse Autoencoders Find Highly Interpretable Features in Language Models
- Authors (†): Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey
- Year / venue: 2023, arXiv v3 (4 Oct 2023). Venue not shown on the page read; the project's reading list says ICLR 2024 (**unverified here**).
- Peer review: preprint as far as verified.
- ID: arXiv:2309.08600 — https://arxiv.org/abs/2309.08600 — open access
- Code: https://github.com/HoagyC/sparse_coding
- **Level: P** (arXiv HTML v3).
- Main findings: ReLU SAE with tied encoder/decoder weights, rows of the dictionary normalised, loss = squared reconstruction + alpha x L1 of the activations (Sec. 2, Eq. 4). Pythia-70M and Pythia-410M, Pile activations, Adam lr 1e-3, 5-50M activation vectors (App. B). Dictionary ratios 0.5-8 tested; smaller dictionaries reconstruct less accurately, interpretability holds across ratios (App. C). Dead features are common in MLP dictionaries; resampling of dead features **not implemented** (App. E). Substituting reconstructions at layer 2 of Pythia-70M raised perplexity 25 to 40 (Sec. 6.2).
- Methodology relevant to Stage 2: the tied-weight L1 SAE recipe that later ground-truth studies copy. **No ground-truth recovery metric is reported in this paper**; it cites Sharkey et al. and Wright and Ma for recovery.
- Limitations (authors): imperfect reconstruction, dead features, works best on residual stream, autointerpretability limits.
- Claims it supports: a dictionary trained with reconstruction + L1 can give decoder directions that are interpretable features (LM evidence only); dead features are an issue without resampling.
- Used in: proposal sections 2.1, 2.2.

### SAE_2022_202 — [Interim research report] Taking features out of superposition with sparse autoencoders (MMCS)
- Authors: Lee Sharkey, Dan Braun, beren (Conjecture)
- Year / venue: 13 Dec 2022, AI Alignment Forum post
- Peer review: **not peer-reviewed** (blog post, interim report)
- ID: https://www.alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition (greaterwrong mirror gave a connection error; the alignmentforum URL worked)
- Code: none verified
- **Level: P** (page read via summarising fetch).
- Main findings: ground truth G = 512 features uniformly on the unit sphere in h = 256 dimensions, on average 5 active per sample, activations uniform (0,1), correlated via a random covariance, probabilities decaying exponentially (lambda = 0.99). SAE: ReLU encoder, bias-free unit-norm decoder, loss (1/h)||x - Dc||^2 + (1/J) alpha |c|_1, Adam lr 1e-3, batch 256. **Defines mean max cosine similarity (MMCS) = (1/G) sum_g max_j cos(D_j, F_g)**; the post does not mention an absolute value, so the cosine appears signed; learned elements that match no ground-truth feature are ignored, so duplicates and dead neurons do not lower the score. Recovery works in a broad "Goldilocks" band of the L1 coefficient (about 0.03-0.3 on a log scale); dictionaries 1-8x the ground-truth size recover well with MMCS declining slowly; larger dictionaries need larger L1; failure modes: L1 too high gives dead neurons and a sharp reconstruction jump, too low gives good reconstruction but poor recovery, too-small dictionary has no optimum, too-large has dead neurons and redundant features. On a 31M-parameter LM (layer 1 MLP) dead neurons did not appear, MMCS between dictionaries never exceeded about 0.4, and the authors could not confidently identify hyperparameters.
- Limitations: interim report; the post's comments call dividing the L1 term by J a mistake (weakens sparsity as dictionaries grow); toy is 2x overcomplete with ~5 active features, not our n >> m case.
- Claims it supports: definition of MMCS; the Goldilocks band; robustness to 1-8x over-completeness in a toy.
- Used in: proposal sections 2.2, 2.3.

### SAE_2024_203 — Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders
- Authors (†): Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, Janos Kramar, Neel Nanda
- Year / venue: 2024, arXiv v3 (1 Aug 2024). Venue not shown on the page read; preprint as far as verified.
- ID: arXiv:2407.14435 — https://arxiv.org/abs/2407.14435 — open access
- Code: no link in the page read
- **Level: P** (arXiv HTML; truncated before end of Appendix J).
- Main findings: encoder f(x) = JumpReLU_theta(W_enc x + b_enc) with a learned per-latent threshold theta (Eq. 4, 8); loss = L2 reconstruction + lambda x L0 (Eq. 9); straight-through estimators with a rectangle kernel (Sec. 3-4). Gemma 2 9B, widths 131k, 8B tokens, Adam beta1 = 0, lr 7e-5, batch 4096 (App. I). "At least as good as, and often slightly better than" TopK and better than Gated on delta-LM-loss at a given L0 (Sec. 1, Fig. 2; shown graphically, no headline numbers in text). Few dead latents without resampling for JumpReLU and TopK; fewer than 0.06% of features fire on more than 10% of tokens (Sec. 5.2). Blinded manual interpretability study: similar rating distributions across architectures (Sec. 5.3.1).
- Limitations (Sec. 7): one model family; more high-frequency features than Gated; new hyperparameters (threshold init, bandwidth epsilon) with no principled selection; metrics may not reflect downstream usefulness.
- Claims it supports: JumpReLU has a per-latent threshold (useful as an "active set" definition); L0-penalty training avoids L1 shrinkage.
- Used in: proposal D-S2-1.

### SAE_2024_204 — BatchTopK Sparse Autoencoders
- Authors (†): Bart Bussmann, Patrick Leask, Neel Nanda
- Year / venue: 2024, arXiv v1 (9 Dec 2024). Venue not shown on the page read; preprint as far as verified.
- ID: arXiv:2412.06410 — https://arxiv.org/abs/2412.06410 — open access
- Code: https://github.com/bartbussmann/BatchTopK
- **Level: P** (arXiv HTML; the page text contains no numeric tables, so comparisons are qualitative).
- Main findings: keeps the n x k largest activations across a batch of n samples (Sec. 3, Eq. 6), so some samples use more than k latents and some fewer; at inference a single threshold theta (average of the minimum positive activation over batches, Eq. 7) replaces the batch dependency. Claimed better reconstruction at the same average L0 than TopK on GPT-2 Small and Gemma 2 2B, and better than JumpReLU on GPT-2 Small (Fig. 1-2); on Gemma 2 2B better NMSE at all k but better CE only at k = 16. AuxK-style dead-latent loss as in TopK. Adam lr 3e-4, batch 4096, 1e9 tokens (App. A.1).
- Limitations: two models, specific layers; human interpretability not assessed; global threshold approximates the batch rule; no numeric tables in the text read.
- Claims it supports: variable per-sample sparsity is possible with a TopK-style training rule; relevant because our toy has about 37% empty samples (my calculation, Poisson(1)).
- Used in: proposal D-S2-1.

### SAE_2024_205 — Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders
- Authors: Charles O'Neill, Alim Gumran, David Klindt
- Year / venue: 2024, arXiv v2 (30 Jan 2025). Venue not stated in the page read (keywords mention ICML; **acceptance not confirmed**); preprint as far as verified.
- ID: arXiv:2411.13117 — https://arxiv.org/abs/2411.13117 — open access
- Code: none in the visible content
- **Level: P** (arXiv HTML; appendices after H.2 cut off).
- Main findings: for K-sparse sources (K >= 2) projected linearly to M dimensions with K log(N/K) <= M < N and RIP, a linear-nonlinear (single-pass) SAE encoder "must have a non-zero amortisation gap" (Thm 3.1 / A.1); a worst-case rank argument, not a quantified gap; iterative or multi-layer encoders can avoid it (App. B). Toy data (N = 16, M = 8, K = 3): MLP encoder reaches MCC about 0.1 higher than the SAE at convergence (Sec. 4.1); with known dictionary MLP about 10% higher (Sec. 4.2); sparse coding over 10% higher on latent prediction and about 10% higher on dictionary learning (Sec. 4.3); larger run N = 1000, M = 200, K = 20 gap "more pronounced" (App. C). GPT-2 Small layer 9 (16,896 latents): SAE NMSE 0.061, L0 35.66, 11% dead (Sec. 5). MCC is used as the toy metric (definition not extracted).
- Limitations: constant sparsity and uncorrelated channels; recent SAE variants (TopK, JumpReLU) not tested; SAE+ITO used gradient inference, not matching pursuit; MLP uses more compute.
- Claims it supports: an SAE encoder's activations are an approximation to the true sparse code (my inference of the implication; the paper says it does not measure this on real data).
- Used in: proposal sections 2.2, 3, D-S2-1.

### SAE_2025_206 — Sparse Autoencoders Do Not Find Canonical Units of Analysis
- Authors (†): Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, Neel Nanda
- Year / venue: 2025, arXiv v1 (7 Feb 2025). No venue on the page read; appendices mention a "review period", acceptance not confirmed; preprint as far as verified.
- ID: arXiv:2502.04878 — https://arxiv.org/abs/2502.04878 — open access
- Code: no official repository linked (dashboard at metasaes.streamlit.app)
- **Level: P** (arXiv HTML).
- Main findings: SAE stitching: latents of a larger SAE with max decoder cosine below a threshold (0.7 for GPT-2, 0.4 for Gemma Scope, chosen heuristically) to the smaller SAE are "novel", the rest "reconstruction latents"; adding novel latents from a 1536-latent SAE to the 768-latent SAE cuts MSE by almost 10% (Fig. 15); among same-size SAEs 94% of latents are reconstruction latents (Fig. 16). Meta-SAE trained on decoder directions of a 49,152-latent SAE explains 55.47% of decoder variance (Sec. 5). Conclusion: no single width gives a unique, complete, atomic dictionary; choose width pragmatically (Sec. 6).
- Limitations: two LLMs, heuristic thresholds, meta-SAE explains ~55%, LLM-judged labels.
- Claims it supports: decoder directions depend on dictionary width; a toy with known ground truth is needed to say what "right" is.
- Used in: proposal section 2.2.

### SAE_2025_207 — On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy
- Authors: Jingyi Cui, Qi Zhang, Yifei Wang, Yisen Wang
- Year / venue: 2025, arXiv v2 (4 Mar 2026). No venue stated; preprint as far as verified.
- ID: arXiv:2506.15963 — https://arxiv.org/abs/2506.15963 — open access
- Code: none in the visible content
- **Level: P** (arXiv HTML; truncated before proofs end).
- Main findings: theory for a one-layer ReLU SAE on superposed inputs x_p = W_p x (i.i.d. sparse features, columns of W_p with non-positive inner products): exact recovery (up to reordering) only in the limit S -> 1 (Thm 2, 3); at general sparsity "feature shrinking" (a polysemantic dimension is shrunk, e.g. 1.0 to 0.2, Example 1) and "feature vanishing" (some features go to zero, Example 2). Proposed weighted reconstruction loss (WSAE) with weights from a per-dimension proxy (activation variance for LMs): synthetic n = 200, n_p = 20 lower ground-truth reconstruction error at low sparsity; Pythia-160M auto-interp improves about 3.8% on average.
- Limitations: ReLU, no bias or L1 in theory; i.i.d. features; proxies are not ground truth; recovery guarantee is asymptotic.
- Claims it supports: SAEs can shrink or drop features in superposition at moderate sparsity, so decoder columns can under-represent the heavily overlapped features.
- Used in: proposal section 2.2.

### SAE_2026_208 — ToyBench: Ground-Truth Evaluation of SAEs Across Feature Distributions
- Authors: Oliver Sieweke, Niclas Kuepper, Kaushik Reddy, Kola Ayonrinde
- Year / venue: 2026, NeurIPS 2026 workshop "Interpretability as a Science: Toward Rigorous Foundations for Understanding LLMs" (poster listing; **acceptance status not stated on the listing**, links to an OpenReview forum)
- Peer review: workshop listing; not verified
- ID: https://neurips.cc/virtual/2026/166038 — open access listing
- Code: releases a library "tmos" (name from the listing; no URL read)
- **Level: A** (listing page: abstract only. **Full methodology and results NOT verified.**)
- Main findings (abstract level): eight synthetic feature distributions compressed by a toy model of superposition, four SAE architectures compared, none dominates; BatchTopK best on average but ReLU SAEs first on some distributions; feature-detection F1 never exceeds 0.84 for the evaluated SAEs while idealised SAEs built with perfect knowledge exceed 0.95.
- Limitations: unknown (only abstract read).
- Claims it supports (tentatively): even in toy models with ground truth, trained SAEs recover features imperfectly; architecture ranking is distribution dependent.
- **Needs a full-text read before it is used to justify a decision** (listed in the proposal's PDF section).
- Used in: proposal sections 2.2, 2.4, D-S2-1.

### SAE_2026_209 — A Mechanistic Investigation of Supervised Fine Tuning
- Authors: Ruhaan Chopra (independent researcher)
- Year / venue: 12 May 2026, arXiv v1; no venue; preprint
- ID: arXiv:2605.11426 — https://arxiv.org/abs/2605.11426 — open access
- Code: https://github.com/ruhzi/sae-investigation
- **Level: P** (arXiv HTML).
- Main findings: a frozen Gemma Scope 2 SAE (trained on the base model) is applied to base and fine-tuned Gemma 3 1B IT activations; SAE-latent cosine similarity between base and fine-tuned falls much more than raw-activation cosine (raw stays above 0.96; on MultiNLI at the 262k width SAE-latent cosine goes 0.909 to 0.874, 0.924 to 0.830, 0.740 to 0.557 at layers 7, 13, 22 between steps 400 and 2000). The paper treats a frozen SAE as a valid fixed reference but **does not report whether the SAE still reconstructs the fine-tuned activations**, so part of the latent drift may be SAE mismatch (the paper says so).
- Limitations (authors): one suite and scale; depends on available SAEs. Single-author preprint.
- Claims it supports: a frozen SAE gives latent readings that drift when the model changes; staleness of an unrefreshed SAE is plausible but not separated from true change in this paper.
- Used in: proposal section 4.

### RAT_2024_210 — Obfuscated Activations Bypass LLM Latent-Space Defenses
- Authors (†): Luke Bailey, Alex Serrano, Abhay Sheshadri, Mikhail Seleznyov, Jordan Taylor, Erik Jenner, Jacob Hilton, Stephen Casper, Carlos Guestrin, Scott Emmons (only the first name, Luke Bailey, is corroborated by the code URL the page gave)
- Year / venue: 2024, arXiv v2 (8 Feb 2025). No venue on the page read; preprint as far as verified.
- ID: arXiv:2412.09565 — https://arxiv.org/abs/2412.09565 — open access
- Code: https://github.com/LukeBailey181/obfuscated-activations
- **Level: P** (arXiv HTML, about half of a very long page read).
- Main findings: models/attacks can be trained so latent-space monitors (probes, OOD detectors, SAE-based monitors) fail while behaviour is preserved: probe recall can fall from 100% to 0% while a 90% jailbreak rate is retained (Sec. 3.3, Fig. 4b); an SAE-latent monitor was also bypassed by a soft-prompt obfuscation attack (Sec. 5.2, Fig. 14); obfuscating against more layers reduces task performance, most strongly for SAE-based probes (Sec. 5.3-5.4).
- Limitations: attacker's objective is explicit evasion, not a side effect of another loss; specific models; adversarial training might converge with longer runs.
- Claims it supports: when a model is optimised with a latent monitor in the loss, the representation can move so that the monitor's output changes without the property it was meant to track changing. Supporting evidence for decision D-09's Goodhart worry; **not** a demonstration that our penalty would be gamed.
- Used in: proposal section 4, risks.

### RAT_2022_211 — Scaling Laws for Reward Model Overoptimization
- Authors: Leo Gao, John Schulman, Jacob Hilton
- Year / venue: 2022, arXiv (19 Oct 2022). Venue not shown on the page read (the project does not record one); preprint as far as verified.
- ID: arXiv:2210.10760 — https://arxiv.org/abs/2210.10760 — open access
- Code: none on the page
- **Level: A** (arXiv abstract page only. Full text NOT read.)
- Main findings (abstract): optimising a proxy reward model too far hurts gold-standard performance (Goodhart); the relationship has a functional form that depends on the optimisation method and whose coefficients scale smoothly with reward-model parameter count.
- Claims it supports: general Goodhart evidence for optimising against a learned proxy. Only used as background for why the penalty is evaluated against an independent ruler.
- Used in: proposal section 4.

### OPT_2017_212 — GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
- Authors: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
- Year / venue: 2017, Advances in Neural Information Processing Systems 30 (NIPS 2017), per the arXiv journal-reference field. Peer-reviewed (proceedings).
- ID: arXiv:1706.08500 — https://arxiv.org/abs/1706.08500 — open access
- Code: not on the page read
- **Level: A** (arXiv abstract page only. Method and proof NOT read.)
- Main findings (abstract): giving each of two coupled networks its own learning rate (TTUR) yields convergence to a stationary local Nash equilibrium under mild assumptions, using stochastic-approximation theory; extends to Adam.
- Claims it supports: background for running two coupled optimisers (model, SAE) at different learning rates. GAN setting, so the conclusion does not transfer automatically to "model + SAE"; this is my analogy.
- Used in: proposal section 4.

### MATH_2010_213 — Coherence-Based Performance Guarantees for Estimating a Sparse Vector Under Random Noise
- Authors: Zvika Ben-Haim, Yonina C. Eldar, Michael Elad
- Year / venue: IEEE Transactions on Signal Processing 58(10), October 2010, pp. 5030-5043. **Peer-reviewed.**
- ID: DOI 10.1109/TSP.2010.2052460 — author-hosted PDF: https://webee.technion.ac.il/Sites/People/YoninaEldar/Coherence.pdf — open access
- Code: none
- **Level: F** (PDF pages 1-5 of the journal paper rendered and read: abstract, intro, Sec. II-III up to Theorem 4; Sec. IV onwards not read). The PDF was fetched by the web-fetch tool, which saved it to its tool-results folder.
- Main findings: defines mutual coherence mu = max over i != j of |a_i^T a_j| for unit-norm atoms (Eq. 3). Lemma 1 (Cai, Xu, Zhang): the restricted-isometry constant delta_s <= (s-1) mu (Eq. 6). Coherence-based guarantees for estimating an s-sparse vector from noisy measurements b = A x + w with a **known** overcomplete dictionary A: Theorem 1 (Cai, Wang, Xu, Dantzig selector) needs s < (1/2)(1 + 1/mu) (Eq. 14); Theorem 3 (BPDN) needs s < 1/(3 mu) (Eq. 23); Theorem 4 (OMP) needs |x_min| - (2s-1) mu |x_max| >= 2 sigma sqrt(2(1+alpha) log m) (Eq. 30), i.e. the smallest nonzero entry must be large relative to noise and to the largest entry.
- Methodology: non-asymptotic bounds based only on coherence, so applicable to arbitrary dictionaries.
- Limitations: deterministic sparse vector, **known** dictionary, Gaussian noise; says nothing about learning the dictionary, so it is an analogy for SAEs, not a guarantee about them.
- Claims it supports: lower coherence of the dictionary (lower worst-case overlap) enlarges the sparsity level at which sparse recovery is guaranteed. Used only as an illustration (proposal sections 2.4, 5; my arithmetic with Welch-bound coherence is marked as inference). Entry-magnitude spread matters for greedy recovery: our features have magnitudes Uniform(0,2) (Stage 0 data code).

### MATH_1974_214 — Welch bounds and mutual coherence (two Wikipedia pages; secondary)
- Authors: Wikipedia contributors (secondary source). Original Welch bound: L. R. Welch, 1974 (**not opened**).
- Year / venue: pages read 2026-11-15. **Not peer-reviewed** (encyclopedia).
- ID: https://en.wikipedia.org/wiki/Welch_bounds ; https://en.wikipedia.org/wiki/Mutual_coherence_(linear_algebra) — open access
- **Level: P** (secondary page text via summarising fetch).
- Main findings: for m unit vectors in an n-dimensional space the sum over distinct pairs of squared inner products is at least m(m-n)/n, so the mean over ordered pairs is at least (m-n)/(n(m-1)) (equality iff the vectors form an equiangular tight frame); coherence form: max |inner product|^2 >= (m-n)/(n(m-1)). **In this project's notation (N vectors in m dimensions) the floor on mean squared overlap is (N-m)/(m(N-1)).** My arithmetic with N = 1000: 0.0657 (m = 15), 0.0390 (m = 25), 0.0149 (m = 63), which equals the "Welch minimum" values recorded in notes 2 Addendum 19.
- Limitations: Wikipedia notation swaps n and m; the Mutual coherence page did not state a recovery inequality.
- Claims it supports: any set of N unit vectors has mean squared overlap at or above the floor; this also bounds an SAE decoder's measured overlap from below, so the SAE ruler cannot read below the floor (my inference).

### Section 1 tally (new records)
Level P: SAE_2023_201, SAE_2022_202, SAE_2024_203, SAE_2024_204, SAE_2024_205, SAE_2025_206, SAE_2025_207, SAE_2026_209, RAT_2024_210, MATH_1974_214 (10). Level F: MATH_2010_213 (1). Level A: SAE_2026_208, RAT_2022_211, OPT_2017_212 (3). Total new = 14.

---

## 2. Existing records re-opened this session (level confirmed or changed; claims used by the proposal)

| Key | Paper | Level this session | What was confirmed or changed |
|---|---|---|---|
| SAE_2024_010 | Scaling and Evaluating SAEs (Gao et al., OpenAI), arXiv:2406.04093, code github.com/openai/sparse_autoencoder | **P** (HTML to ~100k of 119k chars; same as index) | Confirmed: TopK encoder z = TopK(W_enc(x - b_pre)), reconstruction-only loss, decoder columns renormalised each step, geometric-median b_pre (Sec. 2.1-2.3, App. A.1). Dead = inactive for typically 10M tokens; AuxK uses top k_aux = 512 dead latents, alpha = 1/32, ~10% extra compute (App. A.2, B.1). Without mitigation up to 90% dead; with AuxK and tied init 7% for the 16M-latent SAE (Sec. 2.4). Constant-lr Adam; largest convergent lr scales ~ 1/sqrt(n) (Sec. 3.1.2, Fig. 3). Dense latents appear from k about 256 (App. E.5). TopK forces exactly k latents per token (Sec. 6, stated limitation). Venue: page shows an arXiv preprint; the index's "ICLR 2025" is not visible in the text. |
| SAE_2025_006 | SAEBench (Karvonen et al.), arXiv:2503.09532, code github.com/adamkarvonen/SAEBench | **P** | Confirmed: eight metrics; absorption worsens with width for all architectures except Matryoshka (Sec. 4.2); sparsity-fidelity frontier "does not reliably indicate performance on downstream tasks" (Sec. 4.1); moderate L0 50-150 compromise (Sec. 4.3). Venue: page lists ICML only as a keyword, acceptance not confirmed in text (index says ICML 2025 from arXiv comments). |
| SAE_2025_009 | Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers (Heap et al.), arXiv:2501.17727 | **P** (index has F from earlier PDF read; unchanged) | Toy setup: 512 features, 256 dims, E = 5 active, frequency decay lambda = 0.99, 10K samples, 10 seeds, 100 epochs; MMCS defined as for Sharkey (Sec. 4, App. I.1-I.2); "Goldilocks zone" of L1 coefficient; no numeric band extracted. Random MLP outputs are sparse regardless of input (Fig. 5a). The page gives no venue; preprint. |
| SAE_2024_004 | A is for Absorption (Chanin et al.), arXiv:2409.14507, code github.com/lasr-spelling/sae-spelling | **P** (to ~100k of 137k chars) | Confirmed definitions: splitting = general latent becomes several specific ones in a wider SAE; absorption = specific latent fires in place of a general one on some tokens. Toy: four orthogonal 50-d features, feature 0 p = 0.25, others 0.05, 4 latents, L1 3e-5, 100M activations; absorption when feature 1 fires only together with feature 0 (Sec. 3). Simplified argument: loss p11(2 - delta) + p10, derivative -p11, so more absorption lowers the sparsity loss whenever p11 > 0 (App. A.2; the authors' own caveat: one parameterised family, not a general proof). Rate rises with width and with sparsity (Sec. 5.1, 5.3, Fig. 9; numbers only in figures). Venue: page shows arXiv v6 only (index says NeurIPS 2025 oral from arXiv comments, unverified here). |
| SAE_2025_014 | Sparse but Wrong (Chanin and Garriga-Alonso), arXiv:2508.16560, code github.com/chanind/sparse-but-wrong-paper | **P** (to ~100k of 136k chars; same as index) | Confirmed: toy features are orthogonal unit vectors (a superposition variant with 50 features in d = 40 in App. E.1), true L0 known; L0 too low mixes in correlated features, and a low-L0 SAE with mixed latents reaches MSE 2.73 against 4.88 for the ground-truth SAE at L0 = 5 (Sec. 3.3, Thm F.1-F.2); metric **c_dec = mean absolute pairwise cosine of decoder latents (Eq. 4)**, minimised near the true L0 in the toy (Fig. 7, App. E.1-E.2) and with broad minima in some LLM layers (Sec. 4, Sec. 6); decoder cosine heatmaps used as the recovery picture. No venue on page; preprint. |
| SAE_2025_101 | Feature Hedging (Chanin, Dulka, Garriga-Alonso), arXiv:2505.11756, code github.com/chanind/feature-hedging-paper | **P** (upgraded from A) | Narrow SAEs (too few latents for correlated features) mix a component of the missing feature into a latent; hedging is worse for narrower SAEs, absorption worse for wider (Table 1, Sec. 4.1); full-width toy SAEs recover the features despite correlation (Sec. 3.5.3-3.5.4); toys use orthogonal features and 15M synthetic samples. Balance-matryoshka mitigation gives modest, metric-dependent gains (Sec. 5). No venue on page; preprint (MATS). |
| SAE_2026_102 | SynthSAEBench (Chanin and Garriga-Alonso), arXiv:2602.14687, code github.com/decoderesearch/synth-sae-bench-experiments | **P** (upgraded from A; to ~100k of 118k chars) | Synthetic data: N = 16,384 features in D = 768 (rho_mm about 0.15), Zipfian firing p_max 0.4 to p_min 5e-4, hierarchy and correlation options, ground-truth L0 34 (Sec. 3, 5). **Metrics defined (Sec. 4):** R^2 (Eq. 16); **MCC** = absolute cosine between each decoder column and each true feature, optimal one-to-one matching (Hungarian), average over min(L, N) (Eq. 17); **feature uniqueness** = distinct best-match indices / L (Eq. 18); **precision, recall, F1** treating each latent as a classifier for its best-matching feature (Sec. 4.3); L0 and dead latents (Sec. 4.4). Findings: higher L0 raises recall, lowers precision; JumpReLU and BatchTopK nearly identical F1 at L0 20-25; Matryoshka best MCC and F1 but poor R^2; MP-SAE best R^2 but poor MCC and F1, with MCC and F1 falling as superposition rises (Sec. 6). The visible text reports no direct splitting or absorption measurements. Preprint; no venue. |
| SAE_2025_003 | SAEs Trained on the Same Data Learn Different Features (Paulo and Belrose), arXiv:2501.16615 | index level **F** kept; HTML re-read at **P** | Confirmed: latents matched across SAEs with the Hungarian algorithm, "shared" if cosine >= 0.7 for both encoder and decoder matchings; 30% shared (Llama-3-8B, 131K latents), 42% on Pythia-160M 32K latents (2 seeds), about 35% found in only one of nine SAEs; overlap falls with SAE size and k, rises with training tokens; ReLU/L1 SAEs overlap more than TopK; mean-max-cosine matching barely changes the result. No formal limitations section; 0.7 threshold is a chosen definition. Venue: "ICML" keyword and an anonymous submission; preprint. |
| SAE_2024_001 | End-to-End SAE (Braun et al.), arXiv:2405.12241, code github.com/ApolloResearch/e2e_sae | **P** (same as index) | Confirmed: loss = KL(SAE-spliced vs original output) + L1; base model frozen, gradients pass through it to the SAE; e2e takes about 2-3.5x the wall-clock time of local training; e2e features differ substantially across seeds, local and e2e+downstream are more stable; e2e SAEs reconstruct activations worse; authors raise Goodhart's law as a concern; main results GPT-2 small layers 2, 6, 10. No venue on page; index says NeurIPS 2024 (unverified). |
| SAE_2024_011 | SAE-Track (Xu et al.), arXiv:2412.17626, code github.com/Superposition09m/SAE-Track | **P** | Confirmed: each checkpoint's SAE is initialised from the previous one; later SAEs need at most 1/20 of the initial tokens (App. I); feature directions keep drifting after features form and stabilise only late (Conclusions 3.1-3.2). **The paper does not evaluate an SAE on checkpoints it was not trained on, so it gives no staleness number.** Dictionary size, optimiser, lr not stated in the text read. No venue on page; preprint. |
| RAT_2025_106 | SAE regularization produces more interpretable models (Lai and StefanHex), LessWrong, 28 Jan 2025; code github.com/peterlai/gpt-circuits | **P** (blog; **not peer-reviewed**) | Confirmed: SAE losses (reconstruction scaled by activation magnitude, plus sparsity) added to the LM loss, LM and SAEs trained simultaneously, char-level GPT-2, 4 layers, width 64, Shakespeare; fresh SAEs trained afterwards for evaluation; per-layer coefficients not given, chosen to amplify the effect; "reliably better" on most layers with layer 3 inconsistent; too-large coefficients hurt cross-entropy; results only in graphs (no numbers); the post does **not** say whether SAE-loss gradients flow into the LM or inputs are detached; authors call joint training computationally impractical unless decomposition is paramount. |
| SUP_2025_108 | Superposition as Lossy Compression (Bereska et al.), arXiv:2512.13568 | **P** (upgraded from A) | Defines superposition psi = F/N with F = exp(entropy of per-latent L1 activation shares) from an SAE (N = neurons). On 100 Elhage-style toy models (20 features, 5 neurons) the measure correlates r = 0.99 +/- 0.01 with the toy weights' interference structure (the authors' reference), 0.94 +/- 0.02 when applied to SAE activations; sensitive to L1 strength, dictionary size, ghost features and feature splitting; "comparative rather than absolute". **Venue correction:** the page says "Reviewed on OpenReview" and thanks "anonymous TMLR reviewers" but does not confirm acceptance; the index's "TMLR listing" is not confirmed by this page. |

## 3. Existing records cited by the proposal but NOT re-opened this session

SUP_2025_002 (L, project PDF; Stage 0 reference), SUP_2026_004 (F), SAE_2025_002 (F), SAE_2025_005 (P), RAT_2019_101 (P), RAT_2018_011 (F pp. 1-3, 6-8), RAT_2025_009 (P), SUP_2026_101 (P). Claims from them are quoted as the index states them.

## 4. Searched for and not found

- Lai and Heimersheim (2024) and Lai and Huang (2024), named by SAE_2025_002 as concurrent joint SAE + model training: the Lai/StefanHex LessWrong post (RAT_2025_106) appears to be the same line of work (StefanHex is Stefan Heimersheim), but I could not confirm that identification.
- Any measurement of "SAE decoder-column squared overlap versus true squared overlap" in a toy model of superposition: not found. Closest are c_dec (SAE_2025_014), the entropy-based measure (SUP_2025_108) and MCC-style recovery metrics (SAE_2026_102, SAE_2022_202).
- Any retraining-frequency recommendation for an SAE that follows a moving model: none found.
- Donoho and Elad (2003, PNAS) exact coherence theorem: the PMC page was blocked by a browser check; the exact inequality for the general-dictionary case was **not verified** (Ben-Haim et al. gives the related bounds above).
- The Gated SAE paper (arXiv:2404.16014): not opened; Gated SAEs are only discussed here through JumpReLU, SAEBench and Paulo and Belrose.
