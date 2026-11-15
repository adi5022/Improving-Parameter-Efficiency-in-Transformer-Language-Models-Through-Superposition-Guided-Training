# Stage 2 Design Proposal: Make the Penalty Work Without Knowing the True Features

Written 2026-11-15 (research-agent draft for the user to review tomorrow). **Status: PROPOSED, NOT APPROVED.** Every decision D-S2-1 ... D-S2-10 below is a recommendation, not an approval (CLAUDE.md section 9). Nothing here has been run or implemented. Source records, with verification levels, are in [stage_2_sources.md](stage_2_sources.md); keys in square brackets, e.g. [SAE_2025_014], refer to those records or to the existing [citation_index.md](../citation_index.md). Where I say "(my inference)" the statement is reasoning of mine, not a finding of a paper. Numbers labelled "my calculation" were computed by me from formulas in the project's own docs; numbers labelled "estimate" are unmeasured guesses.

Reading order for the user: section 0 (words), section 1 (what Stage 2 must show), section 6 (the ten decisions, each with a one-line recommendation in the table at its start), section 8 (what a failure looks like), section 10 (what I could not verify).

---

## 0. Words used in this document (plain language)

| Term | Plain meaning |
|---|---|
| **Feature / concept** | One of the 1000 things the toy model has to remember. Feature *i* is "on" in a sample with probability p_i (a few features are common, most are rare). |
| **Activation (h)** | The model's internal vector for a sample: h = xW, a list of *m* numbers (the "width"). |
| **Direction W_i** | Row *i* of W: the arrow in *m*-dimensional space where feature *i* is written. |
| **Overlap** | How much two arrows point the same way: the squared cosine between W_i and W_j. Lower overlap means less crosstalk. **Mean squared overlap** is the average over all pairs. |
| **Welch floor** | The smallest mean squared overlap that *any* 1000 arrows in *m* dimensions can have: (N - m) / (m (N - 1)) with N = 1000 [MATH_1974_214, secondary source; my calculation gives 0.0657 / 0.0390 / 0.0149 at m = 15 / 25 / 63, equal to the "Welch minimum" in notes 2 Addendum 19]. Stage 1's penalty reaches this floor. |
| **Oracle** | A method that is allowed to look at the true directions W_i (Stage 1). Stage 2 removes that permission. |
| **SAE (sparse autoencoder)** | A small second network that learns to write each activation as a sum of a few "latent" directions from a dictionary. Each dictionary direction is a **decoder column** d_j. If it works, the decoder columns line up with the true directions. |
| **Latent** | One dictionary entry of the SAE (one decoder column plus its on/off detector). |
| **L0** | How many latents are on for one sample, on average. In our toy the true number of features on per sample is about 1 (by construction). |
| **ReLU+L1, TopK, BatchTopK, JumpReLU** | Four ways to make an SAE sparse. ReLU+L1 adds a penalty that pushes latent values down. TopK keeps exactly the k largest latents per sample. BatchTopK keeps the k x batch-size largest across the whole batch, so some samples get more and some fewer. JumpReLU lets a latent fire only above its own learned threshold. |
| **Dead latent** | A dictionary entry that never fires, so it contributes nothing. |
| **Feature splitting** | One true feature gets spread over several latents (or one latent covers several features). |
| **Feature absorption** | A specific latent fires *instead of* a general one on some inputs, so the general latent has holes. It needs one feature to occur only together with another (hierarchy) [SAE_2024_004]. |
| **Ground truth** | In the toy, the true list of features and directions, known because we built the data. Not available in a real language model. |
| **Stop-gradient (detach)** | Using a number in a loss but blocking learning signals from flowing back through it. Used so the SAE is a *ruler* and the model cannot train the ruler. |
| **Goodhart effect** | When a measurement becomes the training target, the thing being trained finds ways to improve the measurement without improving the thing it was meant to measure. |
| **Seeds** | Independent repeats with different random numbers. Pilot seeds 0-2, final seeds 3-9, as in D-S1-3. |

---

## 1. What Stage 2 must show

Stage 1 asks: if we *know* every concept's arrow (the oracle), does pushing the arrows apart make a small model more accurate than the best tuned weight decay? The 3-seed pilot says it helps by about 5% / 15% / 23% at widths 15 / 25 / 63 (notes 2, Addendum 19, local evidence; final 10-seed grid running, not yet analysed). Stage 2 only starts if Stage 1 passes its gate.

Stage 2 asks the harder question: **can the same push be applied when we are not told the arrows, by watching the model's own activations through an SAE that is trained at the same time?** That is what any real language model would need, since nobody hands us the arrows there.

Stage 2 must show four separate things. Each has its own test and gate (D-S2-8):

1. **The ruler works (G2.0).** On models whose true arrows we know, the overlap read through the SAE tracks the true overlap, including after the penalty has pushed the arrows near the Welch floor, and it ranks a packed model above an unpacked one correctly.
2. **The penalty works (G2).** A penalty built only from what the SAE and the activations reveal lowers the test error below the best tuned weight decay by the same margin Stage 1 uses, and keeps a fixed share of the oracle's gain.
3. **The SAE is doing the work.** It beats equal-cost controls: a version with scrambled SAE directions, a version with no joint training, and a version with no SAE at all.
4. **It is not gamed.** The true overlap (known in the toy) goes down together with the SAE's reading, and fresh SAEs trained afterwards agree.

What Stage 2 does **not** show: anything about real language models. In this toy the features are independent, about one is on per sample, and the SAE's job is much easier than in a language model (section 8, risk R1). A pass here means "the machinery works in the easiest setting", not "it works in an LLM".

---

## 2. Measuring overlap through an SAE when the true features are known (validation setup)

### 2.1 What is measured, and where the SAE reads

The toy model is h = xW, y = ReLU(hW^T + b) (toy/model.py). The only internal vector is h (width *m*), so that is where the SAE reads (D-S2-2). Data: feature *i* is on with probability p_i proportional to 1/i (alpha = 1), value Uniform(0, 2); the expected number on per sample is 1 (toy/data.py, Stage 0 reference table). My calculation: p_1 = 0.134, p_1000 = 1.34e-4; the number on per sample is close to Poisson(1), so about 37% of samples are empty (h = 0), 37% have one feature, 18% two, 8% three or more.

The SAE has *N* latents with unit-length decoder columns d_j. Two overlap readings from it:

- **Matched overlap (validation only, uses ground truth):** pair each true feature with its best latent (Hungarian matching as in [SAE_2026_102] Eq. 17), then compute the mean squared overlap of the matched decoder columns versus the mean squared overlap of the true rows. Tells us whether the SAE got the geometry right.
- **Unmatched overlap M_dec (usable in a loss, no ground truth):** mean squared cosine over all pairs of alive decoder columns. Same spirit as Chanin and Garriga-Alonso's c_dec, which is the *mean absolute* cosine between decoder latents [SAE_2025_014, Eq. 4].

Any set of N unit vectors in *m* dimensions has mean squared overlap at or above the Welch floor [MATH_1974_214], so M_dec cannot read below the floor either (my inference). That is useful: the SAE cannot "report" a better-than-possible geometry.

### 2.2 Known failure modes and whether they bite in this toy

| Failure | What it is | Source | In our toy? (my inference unless cited) |
|---|---|---|---|
| **Wrong sparsity setting (L0 too low or too high)** | Too low: latents mix in correlated features and a trained low-L0 SAE can reach *lower* reconstruction error than the ground-truth SAE (MSE 2.73 vs 4.88 at L0 = 5 in their toy); too high: degenerate mixed latents. Reconstruction error cannot detect it. | [SAE_2025_014] Sec. 3.2-3.3, Thm F.1-F.2 | Applies. The true L0 is known here (about 1), which is a **best case that real language models do not offer**. Their features are orthogonal; their superposition variant (50 features, d = 40) still has c_dec minimised near the true L0 (App. E.1). |
| **L1 coefficient "Goldilocks" band** | Recovery works only in a band of the L1 strength; too high gives dead latents, too low gives good reconstruction with poor recovery; larger dictionaries need larger L1 | [SAE_2022_202]; [SAE_2025_009] Fig. 4 | Applies to ReLU+L1. Their toys (512 features in 256 dims, about 5 on per sample) are far less crowded than ours (1000 in 15-63 dims). |
| **Dictionary width** | 1-8x the true number of features recovers well with MMCS declining slowly; too small has no optimum; too large gives dead and redundant latents. Real SAEs have no single right width. | [SAE_2022_202]; [SAE_2025_206] Sec. 6 | Applies. N = n = 1000 uses ground-truth knowledge (best case); N = 500 and 2000 are sensitivity checks (D-S2-1). |
| **Feature splitting / duplicates** | One feature gets two latents with nearly equal decoder columns. | [SAE_2024_004] App. A.1, Sec. 5.3; [SAE_2025_206] | Possible with N > n. A duplicate pair has overlap near 1 and inflates M_dec; it also creates a trap for the penalty (section 5.3). |
| **Absorption** | A specific latent fires in place of a general one; needs a parent-child relation between features; rate rises with width and with lower L0. | [SAE_2024_004] App. A.2, Sec. 5.1; [SAE_2025_006] Sec. 4.2 | Should be **absent** in Stage 0 data because features are independent (my inference from the mechanism; the authors' toy needs a feature that fires only with another). It returns in the tree toy (Stage 1b) and in language models. So this toy under-states the difficulty. |
| **Hedging** | A SAE that is too narrow for correlated features mixes a part of the missing feature into a latent; worse for narrower SAEs. | [SAE_2025_101] Sec. 3-4 | Not expected with independent features and N >= n (my inference); relevant if N < n. |
| **Dead latents** | Latents that never fire: up to 90% without mitigation in TopK ablations, about 7% with AuxK at 16M latents; few for JumpReLU and TopK according to its authors. | [SAE_2024_010] Sec. 2.4; [SAE_2024_203] Sec. 5.2 | Applies. A feature whose latent is dead is invisible to M_dec. Rare features are *not* automatically dead here: even feature 1000 appears about every 3.7 optimisation steps at batch 2048 (my calculation). |
| **Encoder error ("amortisation gap")** | A one-pass SAE encoder must make errors in some regimes; it is an approximation to the true sparse code. In a toy, a better encoder or plain sparse coding beat the SAE by about 10 MCC points. | [SAE_2024_205] Sec. 3.1, 4.1-4.3 (N = 16, M = 8, K = 3) | Applies to the "which latents are on" part of the in-loop penalty. With about 1 feature on per sample it should be milder (my inference). |
| **Shrinking and vanishing** | In superposition, a polysemantic direction can be shrunk or dropped by the SAE at moderate sparsity. | [SAE_2025_207] Examples 1-2, Thm 2-3 | Applies to the most overlapped features, exactly the ones the penalty is about. |
| **Crowding limits recovery** | For a *known* dictionary, the sparsity level at which recovery is guaranteed falls as the largest pairwise cosine (coherence mu) rises; guarantees also need the smallest active entry not tiny relative to the largest. | [MATH_2010_213] Lemma 1, Thm 1, 3, 4 (known dictionary, noise; analogy only) | Illustration (my arithmetic, not a guarantee): at the Welch-floor coherence mu = 0.256 (m = 15) the condition s < (1/2)(1 + 1/mu) allows s up to 2; at 0.198 (m = 25) up to 3; at 0.122 (m = 63) up to 4. Features have magnitudes Uniform(0, 2), so some active features are tiny, which greedy recovery handles badly. |
| **Seed instability** | Only 30-42% of latents are shared across seeds in language-model SAEs; overlap falls with width and k, ReLU+L1 more stable than TopK. | [SAE_2025_003] Sec. 3, Fig. 1-5 | Matters less here: we compare to ground truth, not to another SAE. Matters for the fresh-SAE check (D-S2-9). |
| **Metrics that mislead** | SAE scores on random networks resemble those on trained ones; sparsity-fidelity frontiers do not predict downstream use. | [SAE_2025_009]; [SAE_2025_006] Sec. 4.1 | This is why the validation uses ground truth and a loss-based outcome, not SAE scores alone. |

### 2.3 Ground-truth recovery metrics used in the literature (names and definitions checked in the fetched text)

| Name | Definition | Source |
|---|---|---|
| **MMCS (mean max cosine similarity)** | (1/G) sum over true features g of max over latents j of cos(D_j, F_g). For each true feature take the best-matching latent; average. Latents matching nothing are ignored, so duplicates and dead latents do not lower it. The post does not mention an absolute value (appears signed). | [SAE_2022_202]; used in [SAE_2025_009] Sec. 4 |
| **MCC (mean correlation coefficient)** | Absolute cosine between each decoder column and each true feature, optimal one-to-one matching (Hungarian), average of matched similarities over min(latents, features). | [SAE_2026_102] Eq. 17; MCC also used as the toy metric in [SAE_2024_205] (definition not extracted) |
| **Feature uniqueness** | Number of distinct best-match true features divided by number of latents; 1.0 means no two latents share a feature. | [SAE_2026_102] Eq. 18 |
| **Per-latent precision / recall / F1** | Each latent is treated as a detector for its best-matching feature: fires when latent > 0 versus feature is on. | [SAE_2026_102] Sec. 4.3 |
| **c_dec** | Mean absolute cosine between all pairs of decoder columns; minimised near the correct L0 in toys. A gauge of mixing, needs no ground truth. | [SAE_2025_014] Eq. 4 |
| **Shared-latent fraction between seeds** | Latents matched across two SAEs by Hungarian matching; "shared" if cosine >= 0.7 for both encoder and decoder matchings. | [SAE_2025_003] Sec. 3 |
| **Reported scale** | In an eight-distribution toy benchmark, evaluated SAEs never exceeded F1 of 0.84 while idealised SAEs with perfect knowledge exceeded 0.95. **Level A (workshop listing only; not verified).** | [SAE_2026_208] |
| **Our own tracking metrics (new, not from a paper)** | (a) ratio M_dec / M_true of unmatched SAE overlap to true overlap; (b) Spearman rank correlation between each feature's true total overlap and the SAE's; (c) *direction test*: does M_dec fall when the true overlap falls (baseline model versus penalised model, same seed)? | this proposal, D-S2-8 |

### 2.4 How well does SAE-measured overlap track the true overlap? What the evidence says

**No source I found measures this exact quantity** (decoder-column squared overlap versus true squared overlap in a toy model of superposition). What exists is related:

- c_dec is lowest near the true L0 in toy models, including a superposed variant [SAE_2025_014].
- An entropy-based superposition measure from an SAE correlated r = 0.94 +/- 0.02 with the toy weights' interference structure on 100 toy models of 20 features in 5 neurons, and r = 0.99 when computed from the toy weights; it is comparative, not absolute, and depends on L1 strength and dictionary size [SUP_2025_108].
- Recovery metrics of trained SAEs in toys stay below ideal (F1 <= 0.84 versus > 0.95 ideal, level A [SAE_2026_208]; MCC gaps of about 0.1 for encoder variants [SAE_2024_205]).

So S2-A (D-S2-10) is a genuine experiment, not a formality, and the answer could be negative.

**What I expect, with reasons (inference):**

1. When the true arrows are well spread (after the Stage 1 penalty), decoder columns should be easy to match and M_dec should sit near the floor, because well-spread dictionaries are the easiest ones to recover from [MATH_2010_213, analogy].
2. In the unpenalised baseline (overlap about 3.2-3.3x the floor at m = 15-25 for gamma = -1, notes 2 Addendum 17) the worst pairs are close together. An SAE can merge two nearly parallel features into one latent; then that pair's overlap is **invisible** to M_dec, so M_dec is biased low for exactly the pairs that matter most.
3. Therefore the direction test (c) and the sign test across seeds are the most important checks: the ruler must at least notice that a penalised model is better packed than a baseline model.

**What this implies for using it in a loss.** A reading with errors of the kind above should not be minimised directly. If the loss were "minimise M_dec", the model would need to change the decoder columns, which it cannot do (stop-gradient), and the SAE would simply follow the model (D-09). The penalty therefore must be a function of the **model's activations**, using the SAE only to say *which directions to look at* and *which are on* (D-S2-3). An error then perturbs which pairs are pushed apart rather than creating a phantom target.

---

## 3. Estimating feature frequencies from SAE activations (for the weighted penalty)

Stage 1's weighted arm uses pair weights w_i w_j with w_i = sqrt(p_i) / mean(sqrt(p)) (notes 2 Addendum 18). In Stage 1 the true p_i is used. In Stage 2 it would be estimated from how often each latent fires.

**Estimator (proposed):** p_hat_j = exponential moving average of the indicator "latent j is on for this sample", with the SAE's own on/off rule (a TopK-style threshold or a JumpReLU threshold [SAE_2024_204] Eq. 7; [SAE_2024_203]). Assign each true feature the p_hat of its matched latent (validation) or use latents directly (in-loop).

**Random error (my calculation, Poisson counting).** If the window holds N samples, the relative error of p_hat_j is 1 / sqrt(N p_j). For the rarest feature (p_1000 = 1.34e-4): N = 1e5 samples gives 27%, N = 1e6 gives 8.7%, N = 2e6 gives 6.1%, N = 1e7 gives 2.7%. One step is 2048 samples, so a window of about 500 steps (1e6 samples) gives about 9% error on the rarest feature and negligible error on common ones. Because w is proportional to sqrt(p), the error in the weight is half the error in p_hat.

**Systematic errors (my inference from the failure modes above):** (i) a latent that fires only above a threshold misses small-magnitude activations: with magnitudes Uniform(0, 2), a cut at value t drops a fraction t/2 of the occurrences, and the cut acts harder on weakly stored features; (ii) splitting shares one feature's frequency across latents; (iii) absorption (absent here) would lower a parent's frequency; (iv) dead latents give p_hat = 0.

**Sensitivity bound (my reasoning):** if p_hat carries no information, the weights become flat and the weighted penalty turns into the plain penalty. So the loss from bad frequencies is bounded by the plain-versus-weighted difference, **unless** the estimates are rank-inverted (rare features weighted as common). The Stage 1 pilot difference between the two is small and noisy (best at m = 25: -10% plain versus -15% weighted; at m = 63: -15% versus -23%; 3 seeds, notes 2 Addendum 19). Experiment S2-B tests this directly in the oracle setting by perturbing the true weights with log-normal noise of three sizes plus a rank-shuffled control (D-S2-5). No source I found studies this sensitivity.

---

## 4. Joint SAE + model training: prior art, instabilities, cost, staleness

### 4.1 Prior art (what exists, what it shows)

| Work | What was done | Outcome | Read at |
|---|---|---|---|
| Lai and StefanHex (blog) [RAT_2025_106] | SAE losses added to the LM loss; LM and SAEs trained together; character-level GPT-2, 4 layers, width 64 | Fresh SAEs on the regularised LM were better on most layers, layer 3 inconsistent; large coefficients hurt cross-entropy; coefficients unreported; **post does not say whether SAE gradients reach the LM**; authors call it computationally impractical unless decomposition matters | P (blog, not peer-reviewed) |
| End-to-end SAEs [SAE_2024_001] | SAE trained against the LM's output (KL); LM frozen | 2-3.5x wall-clock time of local SAEs; e2e features differ across seeds; authors cite Goodhart as a concern | P |
| LoRA on the LM around a frozen SAE [SAE_2025_002] (index record, not re-opened) | LM adapted to a fixed SAE | closes 30-55% of the spliced CE gap in 15M tokens; adapted LM without the SAE not worse (index, level F) | F (index) |
| SAE-Track [SAE_2024_011] | Each checkpoint's SAE warm-started from the previous | later SAEs need at most 1/20 of the initial tokens; directions keep drifting after features form; **no staleness test** | P |
| Frozen SAE under fine-tuning [SAE_2026_209] | Base-model SAE applied to base and fine-tuned activations | SAE-latent cosine falls much more than raw-activation cosine (e.g. 0.740 to 0.557 at layer 22, steps 400-2000); cannot separate real change from SAE mismatch | P (single-author preprint) |
| Training against a monitor [RAT_2024_210] | Models/attacks trained to make latent monitors fail | probe recall 100% to 0% with 90% jailbreak rate kept; an SAE-based monitor also bypassed; more layers obfuscated means worse task performance | P |
| Two time-scale updates [OPT_2017_212]; reward over-optimisation [RAT_2022_211] | background theory and Goodhart evidence, GAN and RL settings | only abstracts read | A |

**Not found:** any from-scratch joint SAE + model training beyond toy scale with capability results; any quantified staleness curve for a lagging SAE; any retraining schedule recommendation (prior_art_search.md section 2, Q4, plus this session's two extra searches).

### 4.2 Instabilities to plan for

1. **Goodhart / co-adaptation.** If penalty gradients could reach the SAE, the SAE would learn to *report* low overlap rather than *measure* it (D-09 option c, recommended against). Stop-gradient closes that. Even with it, a model optimised against a fixed monitor can move activations so the monitor's reading changes without the intended property changing [RAT_2024_210, RAT_2022_211]; the defence is an independent ruler (D-S2-9).
2. **Staleness.** The model changes every step; an SAE updated less often reads an old geometry. [SAE_2026_209] shows frozen-SAE latents drifting under fine-tuning but cannot separate that from SAE error; [SAE_2024_011] shows warm starts are cheap but never tests a stale SAE. So the lag effect must be measured here (S2-C, D-S2-4).
3. **Dead latents and shrinkage.** Fixes exist: AuxK for TopK-style SAEs [SAE_2024_010] (alpha about 1/32, k_aux about 512, about 10% extra compute in their setting); resampling for ReLU+L1 (not described in the sources I could open; see section 10); L0-penalty SAEs shrink less than L1 [SAE_2024_203].
4. **Sparsity-level error.** An SAE with the wrong L0 mixes features without any warning from the reconstruction loss [SAE_2025_014]. Here the true value is about 1; the sensitivity check uses k in {0.5, 1, 2}.
5. **Two optimisers, two time-scales.** The standard theory for two coupled learners with separate learning rates is stochastic approximation (GAN setting, abstract level only) [OPT_2017_212]. My argument by analogy: the SAE is a follower that should adapt faster than the model it measures, so its learning rate should be at least as high as the model's *effective* rate late in the cosine schedule. This is untested here.

### 4.3 Cost (estimates, not measured)

- Toy step now: about 6 large matrix products (forward and backward of x W and h W^T). One SAE step with N = 1000 latents has a comparable count (encoder and decoder, forward and backward), and the leakage penalty adds about 2 products. So I estimate a joint step at about 2.5x the baseline toy step. Stage 1's penalty cost 1.17x when measured (notes 2 Addendum 18).
- Stage 1 pilot ran at 0.31 min per run on the GTX 1660 Ti (34 min / 108 runs, notes 2 Addendum 19); the baseline on a T4 ran at 0.35 min per run (G-C). So I estimate about 0.8 min per run for the SAE arm on either machine (1660 Ti: 6 GB, no tensor cores, so stay in fp32; fp16 measured 3x slower there, notes 2 Addendum 10).
- Memory: SAE parameters for N = 2000, m = 63 are about 250k numbers per run; activation tensors of 2048 x 2000 floats are 16 MB each; with 12 runs per group I estimate under 2 GB total, against 0.5-0.6 GB measured today. Fits a 6 GB card.
- For reference only: a 4096-latent SAE on a 768-dimensional synthetic dataset took about 15-20 minutes on one H100 for 200M samples [SAE_2026_102]. Our SAEs are far smaller.

---

## 5. How the penalty is computed from the SAE (design that the decisions rest on)

### 5.1 Why "penalise the decoder" does not work as a training signal
If the penalty is computed from the SAE's decoder columns alone with stop-gradient, **no gradient reaches the model**: the model's weights do not appear in the formula. The model must enter through its activations h (or through weight rows matched to latents, which does not exist in a general network).

### 5.2 The proposed form: "leakage" of each activation onto the dictionary directions that are off (my design, an adaptation of the Stage 1 pair penalty)

For one sample with activation h, unit direction h_hat = h / ||h|| (skip samples with ||h|| below a small floor), SAE active set S(h) (latents that are on), unit decoder columns d_j, and optional weights q_j (q_j = 1 for the plain version):

  leak(h) = sum over latents j NOT in S(h) of q_j * (d_j . h_hat)^2

Loss term: lambda times the batch mean of leak(h). All SAE quantities (d_j, S, thresholds, q_j) are detached; the gradient reaches the model only through h_hat. Closed form (my derivation): grad_h leak = (2 / ||h||) (I - h_hat h_hat^T) D_off diag(q) D_off^T h_hat, where D_off holds the off columns. It needs about two extra B x N x m products; I would write it by hand, as the Stage 1 penalty was, and check it against autograd.

**Why this is the Stage 1 penalty in disguise (my derivation).** Take a sample where only feature i is on and let the SAE be perfect: h_hat = unit(W_i), the on-latent is i, and leak = sum over j != i of q_j cos^2(W_i, W_j). Averaged over samples, row i is weighted by how often it is the active feature, about p_i. So the expected gradient is the oracle pair-overlap gradient with row weights proportional to p_i. Consequences:
- The activation-based penalty is naturally **frequency-weighted** (weights about p_i on the sample side; partner weights q_j chosen by us). To reproduce Stage 1's *plain* (unweighted) penalty we would need sample weights 1 / p_hat, which has large variance for rare features. The natural comparison is therefore the oracle with the same weighting (rung R1 below), not Stage 1's plain arm. This is a reason to expect a smaller gain than Stage 1 plain, since rare features' rows are touched rarely (a batch of 2048 contains the rarest feature 0.27 times on average, my calculation).
- It uses *scale-free* directions (cosines), so the model cannot reduce the penalty by shrinking activations.

### 5.3 Ways this form can fail (all inference; each gets a check)

1. **Twin latents.** If the SAE splits feature i over two latents with near-equal columns and only one is on, leak counts the twin (cos near 1) and pushes h away from it: the penalty fights the SAE's own splitting. Mitigation: drop from the sum any off-latent whose cosine to an on-latent exceeds 0.9 (threshold a placeholder); log how many pairs this removes.
2. **Merged latents.** A latent that covers two near-parallel features hides that pair (section 2.4, point 2).
3. **Wrong active set** (encoder error, section 2.2) turns an own-feature latent into "off" and pushes h away from its own direction.
4. **Rare features barely trained** by the penalty (section 5.2); the oracle with the same sampling (R1) shows how much that costs by itself.
5. **Two or more features on at once** (about 26% of samples): h is a mixture; leak treats the mixture direction as one direction. The oracle R1 has the same property, so R1 isolates this from SAE error.

### 5.4 A ladder that locates where any loss of gain comes from

| Rung | What it uses | What it answers |
|---|---|---|
| **R0** | Stage 1 oracle pair penalty (plain and weighted); already exists | The ceiling |
| **R1 (ORA-L)** | Leakage form with the *true* unit rows as dictionary and the *true* active set; no SAE | How much is lost just by moving from a weight-row penalty to an activation-sampled penalty |
| **R2 (SAE-L)** | Leakage with the jointly trained SAE (stop-gradient) | How much is lost by replacing the truth with an SAE |
| **Controls** | see D-S2-7 | Whether the SAE is what helps |

---

## 6. Decisions (CLAUDE.md section 9.1 format). All PROPOSED, NOT APPROVED.

| ID | Decision | My recommendation in one line |
|---|---|---|
| D-S2-1 | SAE type, width, sparsity | Choose among ReLU+L1, BatchTopK, JumpReLU by a pre-registered bake-off on frozen models (S2-A); prior pick BatchTopK, k = 1, N = 1000; N = 500 and 2000 as sensitivity |
| D-S2-2 | Where the SAE reads | The bottleneck activation h = xW only |
| D-S2-3 | What the penalty is computed from | Activation "leakage" with detached SAE directions and active set (section 5.2); not the decoder Gram matrix, not co-activation products |
| D-S2-4 | SAE training schedule | SAE updated every model step on detached activations, constant learning rate (tuned), penalty ramped in only after a reconstruction gate; compare refresh gaps of 1 / 10 / 100 / never |
| D-S2-5 | Frequency estimation and weights | Moving-average firing rates; first arm needs no frequency weights; weighted arm only if Stage 1 final shows weighted beats plain; S2-B noise test |
| D-S2-6 | Which Stage 1 arm carries over | Plain penalty + best weight decay (gamma*) as primary; weighted as secondary |
| D-S2-7 | Controls | Oracle leakage (R1), scrambled-direction SAE, frozen-SAE, SAE-free batch-coherence, and the tuned weight-decay baseline |
| D-S2-8 | Success criteria with numbers | G2.0 (ruler) and G2 (method) as in the table in D-S2-8; fixed before joint runs |
| D-S2-9 | Independent ruler and logging | True overlap plus fresh SAEs on saved checkpoints; claims use test error and true overlap only |
| D-S2-10 | Compute, order, seeds | S2-0 build, S2-A, S2-B, S2-C pilot, user check, S2-D final; about 6-7 GPU hours (estimate) |

Common context for all ten: Stage 2 is the second half of the central test. Stage 1 asks whether the oracle push helps; Stage 2 asks whether the push survives without the oracle. Prior art for the penalty forms is in prior_art_search.md section 6 (the contribution is the controlled comparison, not a new regulariser). D-09 (joint SAE with stop-gradient and an independent ruler) is still unapproved; Stage 2 uses its option (b) and needs its approval.

---

### D-S2-1: SAE architecture, width and sparsity

1. **Decision:** Which SAE type, how many latents, and how its sparsity is set.
2. **Context:** Every later Stage 2 result depends on the ruler. Each SAE type has a different failure profile, and the toy is the only place where we can check which is least wrong.
3. **Research evidence:**
   - TopK: exactly k latents per sample, no L1 term, AuxK handles dead latents [SAE_2024_010]; limitation: "forces exactly k latents per token".
   - BatchTopK: variable per-sample sparsity via a batch-level rule and an inference threshold [SAE_2024_204]; our toy has about 37% empty samples and 18% with two on (my calculation).
   - JumpReLU: learned thresholds, L0 penalty, similar to TopK in quality; extra hyperparameters (bandwidth, threshold init) [SAE_2024_203].
   - ReLU+L1: classic; recovery in a "Goldilocks" band [SAE_2022_202, SAE_2025_009]; more seed-stable than TopK in language models [SAE_2025_003]; L1 shrinks values [SAE_2024_203].
   - Rankings are not stable: BatchTopK best on average but ReLU first on some distributions (level A) [SAE_2026_208]; standard L1 had the best MCC when firing magnitudes vary more, while JumpReLU and BatchTopK were nearly identical on F1 [SAE_2026_102] Sec. 6.2, App. M.5 (as extracted).
   - Sparsity level: must be near the truth or features mix [SAE_2025_014].
   - Encoder limitation: iterative inference beat single-pass encoders in toys [SAE_2024_205].
4. **Available approaches:** (A) ReLU+L1 with resampling of dead latents; (B) BatchTopK with an AuxK-style dead-latent loss; (C) JumpReLU with an L0 penalty; (D) fallback only: keep the learned decoder but infer codes iteratively (sparse coding) if encoder error is the limiting factor. Width N in {500, 1000, 2000}; sparsity: L1 sweep (A), k in {0.5, 1, 2} (B), target L0 in {0.5, 1, 2} (C).
5. **Advantages:** A is the classic recipe and the most stable across seeds; B matches variable sparsity and has no shrinkage; C has learned thresholds that give a natural active-set rule; D removes encoder error.
6. **Disadvantages:** A shrinks values, needs an L1 search and resampling; B needs the AuxK term and a threshold for inference, and its k must be set; C has the most knobs; D is slower and further from "SAE".
7. **Recommendation:** run a pre-registered bake-off (S2-A) on frozen Stage 0/1 models, selection on seeds 0-2, rule fixed in advance: pick the configuration with the highest score on (MCC, direction test, ratio M_dec / M_true) averaged over three widths and both model types; ties go to the simpler one. My unvalidated prior is BatchTopK with k = 1 and N = 1000 (matches the known mean L0 and variable sparsity); I would be surprised only mildly if ReLU+L1 wins. D only if all of A-C fail G2.0 for encoder reasons.
8. **Consequences:** fixes the ruler for all later runs; N = n and k = true L0 are **best-case choices that a language model will not allow** (section 8, R1); if nothing passes G2.0 Stage 2 stops here with a negative result.
9. **Approval:** PROPOSED, NOT APPROVED.

### D-S2-2: Where the SAE reads from

1. **Decision:** Which internal vector the SAE decomposes.
2. **Context:** In a language model this is a residual-stream or MLP site; in the toy there is one internal vector.
3. **Research evidence:** The toy's only internal vector is h = xW (toy/model.py). Language-model SAEs in the sources read the residual stream or MLP outputs [SAE_2023_201, SAE_2024_010, SAE_2024_203]; Cunningham et al. found MLP dictionaries had many dead features and the method worked best on the residual stream [SAE_2023_201] App. C.3, E.
4. **Approaches:** (a) h only; (b) h plus the pre-ReLU output logits (1000-dimensional).
5. **Advantages:** (a) is exactly the "m dimensions" where packing is defined; (b) adds nothing about packing.
6. **Disadvantages:** (a) none for the toy; (b) the logit space is 1000-dimensional and is not where the squeeze happens.
7. **Recommendation:** (a). For language models the site is a later decision.
8. **Consequences:** M_dec is measured in the same space as the Stage 1 overlap.
9. **Approval:** PROPOSED, NOT APPROVED.

### D-S2-3: What exactly the penalty is computed from

1. **Decision:** The formula whose gradient reaches the model.
2. **Context:** Section 5.1: a penalty on decoder columns alone gives the model no gradient.
3. **Research evidence:** Form of the oracle penalty: Stage 1 and prior art in prior_art_search.md section 6; coherence of an SAE decoder as an interference measure [SUP_2026_004] (index; acts on the SAE with the opposite aim); c_dec [SAE_2025_014]. No source uses an SAE-derived loss to raise packing (prior_art_search.md section 5).
4. **Approaches:** (A) **leakage** (section 5.2): activation direction versus off-latent decoder columns, SAE quantities detached; (B) **matched-row** penalty: match each model weight row to its nearest decoder column and apply the Stage 1 pair penalty to the rows, with SAE-estimated weights (toy-only, because a general network has no rows); (C) **co-activation** product: sum over latent pairs of z_i z_j (d_i . d_j)^2, with gradient through the encoder's z.
5. **Advantages:** A works in any network and in the toy, and has a clean link to the oracle (section 5.2); B isolates "estimated directions and weights" from "activation sampling" and is the closest to what Stage 1 already trained, but only exists in the toy; C is cheap to write.
6. **Disadvantages:** A is frequency-weighted by construction and may under-train rare features; B does not carry to LMs; C has no clean path to packing: the model can lower it by changing *which latents co-fire* (and the encoder is part of the path), which is a Goodhart route of the kind in [RAT_2024_210] (my inference).
7. **Recommendation:** A as the Stage 2 method, B as an optional diagnostic only if A under-performs R1 and we need to know why, C not tried.
8. **Consequences:** needs a hand-written gradient (about two extra products), a "twin" filter (section 5.3), and a per-run lambda scaled so lambda x |grad leak| / |grad task| matches the Stage 1 scale check (0.001-0.9 range, notes 2 Addendum 18).
9. **Approval:** PROPOSED, NOT APPROVED.

### D-S2-4: SAE training schedule (joint training)

1. **Decision:** How and how often the SAE is updated while the model trains, and when the penalty is switched on.
2. **Context:** D-09 option (b): the SAE takes its own step every model step on detached activations; the penalty never trains the SAE; the penalty is switched on only while the SAE reconstructs well; the true overlap is the independent ruler.
3. **Research evidence:** Prior art and instabilities in section 4: toy-scale joint training exists only as a blog [RAT_2025_106]; warm starts are cheap [SAE_2024_011]; stale frozen SAEs drift but are confounded [SAE_2026_209]; two-timescale theory in GANs [OPT_2017_212] (abstract level); constant-learning-rate Adam is the standard recipe, with the best learning rate falling roughly as 1 / sqrt(width) [SAE_2024_010] Sec. 3.1.2 (language-model widths; not transferable to our size).
4. **Approaches:** (a) SAE updated every model step (1:1); (b) every R steps, R in {10, 100}; (c) fit once on a baseline model and freeze (no joint training); (d) fully coupled (penalty gradient reaches the SAE: not considered).
5. **Advantages:** (a) always current; (b) cheaper; (c) simplest and a needed control.
6. **Disadvantages:** (a) costs the most per step; (b) lags; (c) stale by design.
7. **Recommendation:** (a) as the method; (b) with R = 10, 100 and (c) as the staleness measurement in S2-C (and (c) as the FRZ-L control). Schedule: SAE trains from step 0 while lambda = 0; the penalty is ramped in linearly once the SAE's fraction of variance unexplained is within 25% of the same measure for an SAE whose decoder is set to the true unit rows (placeholder gate, to be reset after S2-A, and recorded). The SAE learning rate is tuned in S2-A (3e-4 to 3e-3 are the values used by the toy studies I read: 1e-3 [SAE_2022_202, SAE_2023_201], 3e-4 [SAE_2024_204]); the model keeps its Stage 0 schedule unchanged. Order in a step: model forward; SAE step on detached h; penalty with the updated SAE (all detached); model step. Dead latents: AuxK or resampling, per D-S2-1's winner.
8. **Consequences:** two optimisers; about 2.5x step cost (estimate); the lag curve (section 4.2, point 2) becomes a reported result.
9. **Approval:** PROPOSED, NOT APPROVED (D-09 approval needed first).

### D-S2-5: Frequency estimation and the weighted arm

1. **Decision:** Whether Stage 2 uses estimated frequencies, and how they are estimated.
2. **Context:** Stage 1's weighted arm needs p_i. Section 3 gives the errors.
3. **Research evidence:** Section 3 (counting error, my calculation); sampling-weight argument in section 5.2 (my derivation). No paper tests sensitivity to frequency errors.
4. **Approaches:** (a) no frequency weights beyond the sampling weight that leakage already has; (b) explicit q_j = sqrt(p_hat_j) / mean from a moving average; (c) both.
5. **Advantages:** (a) one fewer source of error; (b) mimics Stage 1's weighted arm; (c) shows which matters.
6. **Disadvantages:** (b) doubles the tuned variants and adds a window-length choice; error propagation untested.
7. **Recommendation:** run (a) first. Run S2-B (cheap, oracle setting, perturbed true weights at log-normal noise 0.1 / 0.3 / 1.0 plus a rank-shuffled control) *before* deciding on (b). Add (b) to the final grid only if Stage 1's final analysis shows the weighted arm beats the plain arm by more than 3 combined standard errors at two or more widths.
8. **Consequences:** keeps the final grid smaller; the weighted arm stays an upper bound unless S2-B says otherwise.
9. **Approval:** PROPOSED, NOT APPROVED.

### D-S2-6: Which Stage 1 arm carries over

1. **Decision:** Which Stage 1 configuration is the starting point (weight decay, penalty form, strength).
2. **Context:** Stage 1 pilot: plain + gamma* gave -2 to -5% (m = 15), -1 to -10% (m = 25), mixed and noisy at m = 63; weighted + gamma* gave similar or somewhat larger gains at lambda = 0.1; penalty alone (gamma = 0) was worse than gamma* at every width (notes 2 Addendum 19, local evidence, 3 seeds). The pilot's best lambda was the largest tried at two widths.
3. **Research evidence:** local pilot only; the final Stage 1 grid (seeds 3-9) is not yet analysed.
4. **Approaches:** (a) plain + gamma*; (b) weighted + gamma*; (c) penalty alone.
5. **Advantages:** (a) fewest moving parts; (b) larger pilot gains; (c) cleanest but already shown inferior.
6. **Disadvantages:** (b) needs frequencies; (c) loses against gamma*.
7. **Recommendation:** (a) as primary with gamma* per width as in the Stage 1 amendment (-0.55 at m = 15, -1.0 at 25, -2.0 at 63); (b) secondary only per D-S2-5; never (c). Lambda for the leakage form is retuned on seeds 0-2 around the scale-matched value (four values, log-spaced) and the final value is chosen on seeds 0-2 before judging on seeds 3-9, as in D-S1-3.
8. **Consequences:** the Stage 2 verdict is a comparison against L*(m) from the 10-seed baseline and against the Stage 1 result at the same widths 15, 25, 63; if Stage 1 final changes the carried arm, this decision changes with it.
9. **Approval:** PROPOSED, NOT APPROVED.

### D-S2-7: Controls

1. **Decision:** Which controls are mandatory for any Stage 2 claim.
2. **Context:** The project's biggest worry is that the SAE adds nothing (feasibility_assessment.md, risk "SAE adds nothing"), and SAE metrics can look good on models where they mean nothing [SAE_2025_009].
3. **Research evidence:** [SAE_2025_009] (random-network SAEs resemble trained ones); CAFT's use of random and PCA directions as controls [RAT_2025_009] (index).
4. **Approaches (all measured at the same widths, tuning effort and seeds):**
   - **ORA-L** (rung R1): leakage with the true unit rows and true active set. The ceiling for this penalty form.
   - **RND-L**: SAE-L with the decoder columns multiplied by one fixed random rotation Q (m x m, orthogonal). Keeps the SAE's own Gram matrix and active sets, destroys alignment with the real activations. If it helps as much, the geometry of the SAE does not matter.
   - **FRZ-L**: SAE fitted once on a baseline model's activations and frozen (no joint training). Measures the staleness cost.
   - **BCOH (SAE-free)**: penalty on the mean squared cosine between the activation directions of *different samples* in the batch (non-empty samples only). In this toy, two samples with the same feature have identical directions (cosine 1, zero gradient) and samples with different features contribute their true overlap, so BCOH is an SAE-free estimator of the p_i p_j-weighted oracle penalty (my derivation). **If BCOH matches SAE-L, the SAE adds nothing in this toy; but BCOH cannot work in a real network where activations are dense mixtures, so toy parity would not show the SAE is useless in an LLM** (section 8, R2).
   - **Baseline:** tuned weight decay gamma* (Stage 0/1), already run.
5. **Advantages:** together they separate estimator form (ORA-L), SAE alignment (RND-L), joint training (FRZ-L) and the need for an SAE (BCOH).
6. **Disadvantages:** five extra arms; BCOH's strength is a special property of the toy.
7. **Recommendation:** all five in the final grid at one lambda each (chosen on seeds 0-2); RND-L and FRZ-L can be dropped if S2-C shows the SAE arm already fails the pilot gate.
8. **Consequences:** about 80 extra final runs (D-S2-10).
9. **Approval:** PROPOSED, NOT APPROVED.

### D-S2-8: Success criteria and gates (numbers proposed to be fixed before the joint runs)

1. **Decision:** What counts as pass, inconclusive or fail.
2. **Context:** Stage 1's pass rule (D-S1-4): at 2 of 3 widths, mean test error on seeds 3-9 more than 10% below L*(m) and more than 3 combined standard errors; 3-10% or one width is inconclusive. Pilot effects are 5% / 15% / 23% (3 seeds), so the 10% bar is already demanding for a weaker estimator.
3. **Research evidence:** local; no external source fixes such thresholds. The values below are mine.
4. **Approaches:** (a) the table below; (b) lower bars; (c) report effect sizes only, no gate.
5. **Advantages:** (a) fixed in advance protects against rationalising; (c) avoids arbitrary bars.
6. **Disadvantages:** (a) the numbers are guesses and might be too strict (MCC 0.9) or too loose; (c) allows a story to be chosen afterwards.
7. **Recommendation:** (a), with the rule that thresholds V1-V4 are *confirmed or lowered by the user after seeing S2-A pilot results (seeds 0-2) and before any joint-training run*, and any change is recorded with its reason. After that they are frozen.

| Gate | Test | Pass | Notes |
|---|---|---|---|
| **G2.0 ruler valid** (frozen models, seeds 3-9, each of m = 15, 25, 63, both baseline-gamma* models and Stage 1 penalised models) | V1: MCC (Hungarian, absolute cosine, [SAE_2026_102] Eq. 17) | >= 0.90 mean | placeholder; ToyBench saw F1 <= 0.84 on harder toys (level A) |
| | V2: M_dec / M_true | within 0.85-1.15 | M_dec unmatched, alive latents only |
| | V3: direction test: M_dec(baseline) - M_dec(penalised) | right sign in all 7 seeds and >= 70% of the true difference | the most important single check |
| | V4: sparsity robustness: M_dec at k in {0.5, 2} versus k = 1 | within 25% (reported, not gating) | tells how far to trust the in-loop reading |
| **G2 method** (judged on seeds 3-9, lambda chosen on seeds 0-2) | SAE-L (plain + gamma*) test error | below L*(m) by > 10% and > 3 combined standard errors at >= 2 of 3 widths | same as Stage 1 |
| | share of ORA-L's gain | >= 50% (with its confidence interval) | gain = (L*(m) - L) / L*(m); the 50% is a placeholder |
| | against controls | better than BCOH, RND-L and FRZ-L by > 3 combined standard errors at those widths | otherwise "the SAE is not what helps" |
| **G2.1 not gamed** | final-checkpoint in-loop M_dec and fresh-SAE M (other seed) versus M_true | both within 15% of M_true | if M_dec fell more than M_true, flag "ruler drift" and do not count the result |
| **Inconclusive** | 3-10% gain, or one width only, or G2 met but a control matches it | reported as inconclusive | same wording as Stage 1 |
| **Fail** | none of the above | negative result (section 8) | |

8. **Consequences:** if Stage 1's final gain is below about 20% at every width, a Stage 2 that must keep half of it cannot clear the 10% bar; I would then recommend reviewing whether to run Stage 2 at all (my inference, from the placeholders above).
9. **Approval:** PROPOSED, NOT APPROVED.

### D-S2-9: Independent ruler and what is logged

1. **Decision:** How the result is checked against something the model cannot influence.
2. **Context:** D-09 safeguard; section 4.2, point 1.
3. **Research evidence:** [RAT_2024_210], [RAT_2022_211] (Goodhart; the second at abstract level), [SAE_2024_001] (authors raise it too).
4. **Approaches:** (a) true overlap of the learned rows (available only in the toy); (b) fresh SAEs (other seeds, S2-A recipe) trained on saved checkpoints; (c) in-loop reading only.
5. **Advantages:** (a) exact; (b) the only ruler available in an LLM, so it must be rehearsed in the toy; (c) free.
6. **Disadvantages:** (b) costs an SAE per checkpoint and is itself an SAE [SAE_2025_003].
7. **Recommendation:** (a) and (b) at checkpoints (every 2,500 steps and the end). Claims use test error against L*(m) and the true overlap only; in-loop M_dec and fresh-SAE M are reported next to the true overlap. Log for every run: seed, commit, SAE type, N, k or L1, SAE and model learning rates, lambda, gate step, SAE reconstruction error (FVU), fraction of dead latents, number of twin pairs removed, M_dec, M_true, M_fresh, estimated and true frequencies for the 10 rarest and 10 commonest features.
8. **Consequences:** adds a few percent to run time; gives the "reading versus truth" curve over training.
9. **Approval:** PROPOSED, NOT APPROVED.

### D-S2-10: Compute plan, order and seeds

1. **Decision:** Which machines, in what order, with how many runs.
2. **Context:** GTX 1660 Ti (6 GB, no tensor cores, fp32 only) locally, Colab/Kaggle T4s for bigger sweeps; Stage 1 timings in section 4.3.
3. **Research evidence:** local timings only; estimates marked.
4. **Approaches:** all local; local pilots plus T4 final; all T4.
5. **Advantages / disadvantages:** local is free but serial; T4 pair is faster but needs notebook setup (already exists for Stage 0).
6. **Recommendation:** pilots local, final on the Kaggle T4 pair. See section 7 for the table. Order: S2-0 build and unit tests (gradient check against autograd; leakage reproduces the Stage 1 overlap in the oracle case); S2-A; S2-B; S2-C; **user checkpoint** (confirm V1-V4, SAE choice); S2-D final; analysis and write-up.
7. **Consequences:** about 6-7 GPU hours in total (estimate), a few days of wall time including reviews.
8. **Approval:** PROPOSED, NOT APPROVED.

---

## 7. Experiment spec (all estimates are unmeasured; per-run times scale from section 4.3)

Notation: widths m = 15, 25, 63 (D-S1-3); pilot seeds 0-2; final seeds 3-9. "Frozen models" means saved weight matrices from Stage 0/1 runs; I did not check that W is stored for the needed runs (stage doc section 5 says it is planned); if not, retrain the 18 needed models (about 6 min).

| ID | Purpose | Runs | Variables | Metrics | Estimated time |
|---|---|---|---|---|---|
| **S2-0** | Build: SAE module (the chosen types), leakage penalty with hand-written gradient, tests | none | | gradient equals autograd (float64, 1e-8, as in Stage 1); leakage with true rows equals the Stage 1 overlap in the oracle case | coding, no GPU |
| **S2-A1** screen | Bake-off on ONE frozen model per cell | 3 widths x 2 model types (gamma* baseline, Stage 1 best-lambda penalised) x 1 seed = 6 models; per model 18 SAE configs (3 types x 3 sparsity values x 2 widths N = 1000, 2000), 3 learning rates for the winner later | | MCC, uniqueness, F1, M_dec / M_true, Spearman, direction test, FVU, dead fraction | about 1 h |
| **S2-A2** confirm | Winner(s) plus runner-up on seeds 0-2, then judged on seeds 3-9 for G2.0 | 18 frozen models (3 widths x 2 types x 3 seeds) x up to 18 configs = up to 324 SAEs; 7-seed judging set of 42 models for 1-2 chosen configs | N = 500 / 1000 / 2000; k or L1 | same as A1; G2.0 verdict | about 2 h (batched 12-18 SAEs per group; at about 6 ms per step x 40k steps x 18 groups, estimate) |
| **S2-B** | Frequency-weight sensitivity in the oracle setting | Stage 1 weighted arm with perturbed weights: 3 widths x 4 noise settings (0.1 / 0.3 / 1.0 log-normal, rank-shuffle) x 3 seeds = 36 runs | | test error versus unperturbed weighted arm and plain arm | about 15 min |
| **S2-C** | Joint training pilot | SAE-L: 3 widths x 4 lambda x 3 seeds = 36; ORA-L 36; BCOH 36; staleness at m = 25, one lambda, SAE refresh gap in {1, 10, 100, never} x 3 seeds = 12; learning-rate check for the SAE at m = 25 (3 values x 3 seeds = 9) | lambda (scale-matched, four log-spaced values); SAE lr; refresh gap | test error versus gamma* baseline (same seeds), M_true, M_dec, FVU, lag curve | about 1.5 h (SAE-L 0.8, ORA-L 0.5, BCOH 0.45 min per run, estimate) |
| **User checkpoint** | confirm V1-V4, SAE type, lambda ranges; decide on D-S2-5 (b) | | | | none |
| **S2-D final** | The verdict | 3 widths x 7 seeds x arms: SAE-L at 2 lambda (42), ORA-L (21), BCOH (21), RND-L (21), FRZ-L (21) = 126 runs; weighted SAE arm (+21 to +42) only if D-S2-5 says so | lambda chosen on seeds 0-2 | G2 and G2.1 table; true overlap; fresh-SAE overlap | about 1.4 h on a 1660 Ti, about 45 min on two T4s (estimate) |
| Reused, no new runs | baseline L*(m) (10 seeds), Stage 1 oracle arms | | | | |

Total new GPU time: about 6-7 hours (estimate). Memory per group under 2 GB (estimate).

---

## 8. Risks, failure modes and a plan for a clean negative result

| ID | Risk | Why it matters | Plan |
|---|---|---|---|
| **R1** | **The toy is too easy for the SAE.** Independent features, about 1 on per sample, N = n, true L0 known | A pass says little about LMs, where absorption, hedging, unknown L0 and dense mixtures exist [SAE_2024_004, SAE_2025_101, SAE_2025_014] | State it in every result. A later Stage 2b with the tree toy (Stage 1b) or more features on per sample (needs a new Stage 0 baseline) would raise the difficulty |
| **R2** | **An SAE-free estimator works equally well in the toy (BCOH).** Sparse samples let one compare samples directly | Then the SAE adds nothing here; but it would not transfer to dense LMs | Report BCOH next to SAE-L. If BCOH = SAE-L, say "in this toy the SAE is not needed; whether it is needed for dense activations is untested" |
| **R3** | **The ruler fails G2.0** (misreads crowded geometries; merges worst pairs) | Every downstream step depends on it | Stop Stage 2; this is a result (N1 below) |
| **R4** | **Gaming / ruler drift** | in-loop reading improves, true overlap does not [RAT_2024_210] | G2.1; result not counted |
| **R5** | **Stage 1 gain is smaller than the pilot** (final grid uses seeds 3-9 and lambda chosen on 0-2; pilot showed non-monotone cells) | Stage 2 cannot clear the 10% bar if Stage 1 barely does | D-S2-8 consequence; reconsider before running |
| **R6** | **Penalty is a weaker estimator** (rare features under-trained, section 5.2) | gain shrinks even with a perfect SAE | ORA-L isolates it |
| **R7** | **Cost larger than estimated** | budget | measure the real step time in S2-0 before S2-A and S2-C; adjust the grid, not the criteria |
| **R8** | **Twin / merged latents** distort the penalty | section 5.3 | log twin counts; S2-C shows effect |
| **R9** | **Tuning asymmetry**: the SAE arm has more hyperparameters than weight decay | flatters or handicaps | tune on seeds 0-2, judge on 3-9 for every arm; tune weight decay with the full 10-seed baseline already in hand |
| **R10** | **Packed solutions may be more quantisation-fragile** (prior_art_search.md section 8) | the final 2 GB versus 3 GB test | separate cheap check recommended before Stage 3; not part of Stage 2 |

**Clean negative result plan (written so it can be published as is):**

- **N1, ruler invalid (G2.0 fails).** Title the finding as: "SAE-measured overlap is not reliable enough to steer a penalty even in the easiest toy." Report V1-V4 for every SAE type, the pattern of errors (merged pairs, bias at baseline geometry), and what would be needed (iterative inference, other dictionary types). Stop; the project's "interpretability-guided" part ends here and the Stage 1 oracle penalty stands as a weight-level result.
- **N2, ruler valid but no method gain.** Report SAE-L versus ORA-L versus the baseline at each width; the ladder says where the gain went (estimator, SAE, frequency, staleness). Stop or move to alternative_directions.md.
- **N3, method matches only its controls (BCOH or RND-L).** Report that the SAE added nothing in this toy; keep the oracle and SAE-free results; carry BCOH-type estimators only as a conceptual baseline.
- **N4, ruler drift.** Report the divergence curve between in-loop, fresh-SAE and true overlap; it is evidence for the Goodhart risk.
- In all cases: record failed settings, the lambda ranges tried, the number of runs, and what was not tried, in a numbered journal entry (CLAUDE.md section 12), and keep the saved configs so the result can be reproduced.

---

## 9. What this proposal adds to the project's list of open questions (for open_questions.md; not edited here)

- Q-DES-2 (how the penalty reaches the model's weights given a held-fixed SAE) gets a concrete candidate: section 5.2 (leakage).
- Q-DES-1 (which quantity measures superposition) gets two candidates that need no ground truth: M_dec and c_dec; both bounded below by the Welch floor.
- New: does an SAE-free batch estimator (BCOH) explain everything in the toy (R2)?

---

## 10. Honest limits: what I could not verify

1. **No experiment was run.** Every statement about how the SAE, leakage penalty or controls will behave is reasoning (marked "my inference") or an estimate. The central empirical question (does SAE-measured overlap track true overlap near the Welch floor?) has no direct answer in anything I read.
2. **Reading depth.** Most sources were read through a summarising fetch tool (level P): numbers it extracted must be spot-checked before they go in a write-up. Only Ben-Haim et al. was read page by page (level F, pages 1-5). Level A only: ToyBench (workshop listing), Gao-Schulman-Hilton, Heusel et al. No source used for a decision rests on an abstract alone, except that ToyBench motivates how much recovery to expect and is flagged.
3. **Author lists and venues.** Five author lists are from memory (marked in the sources file). Several papers did not show a venue on the fetched page; I treat those as preprints and note any venue the project's records give as unverified.
4. **Dead-latent resampling (ReLU+L1).** The Anthropic pages that describe it were too long for the fetch tool; I describe it only as "exists" and give no recipe. If ReLU+L1 wins the bake-off, I need that text (or a replacement such as AuxK-style loss).
5. **Welch floor.** I checked it against a Wikipedia page (secondary) and against the project's measured Welch minimum values (they agree), but did not open Welch's 1974 paper.
6. **Coherence recovery bounds.** Donoho and Elad (2003) could not be opened (browser check). I used Ben-Haim et al.'s bounds only as an illustration for a known dictionary with noise; they are not guarantees for SAEs.
7. **Not found:** the original Lai and Heimersheim (2024) / Lai and Huang (2024) joint-training papers named in SAE_2025_002 (the Lai and StefanHex blog may be the same work, unconfirmed); any staleness curve for a lagging SAE; any direct test of frequency-estimation sensitivity; any measure of decoder overlap against true overlap in a crowded toy.
8. **Project facts I did not re-derive:** Stage 1 numbers, the 10-seed baseline and the pilot overlaps come from the project's notes and docs (notes 2 Addenda 15-19, stage_0_toy_baseline.md section 13). The final Stage 1 grid is not yet analysed; D-S2-6 and D-S2-8 may need updating when it is. I did not open `results/`, so I could not check whether the frozen weight matrices needed by S2-A are saved.
9. **Time and memory figures** are estimates scaled from Stage 0/1 timings and counting matrix products; they must be measured in S2-0 before the grids are fixed.
10. **Thresholds** in D-S2-8 (0.90 MCC, 0.85-1.15 ratio, 70%, 50%, 15%, 25%, twin threshold 0.9, the 25% reconstruction gate) are my proposals without outside support; they are placeholders to be confirmed after S2-A, then frozen.
11. **Search coverage** repeats the limits in prior_art_search.md (web search only; no Scholar, OpenReview or GitHub code search). This session added about nine searches: joint SAE + model training (none beyond the known blog), SAE staleness and warm starts, ground-truth metrics, coherence theory.

---

## 11. PDFs I need from the user

None of the sources is paywalled; every one I used is open access. Nothing blocks a decision. Worth a full-text read when convenient (in order of usefulness to Stage 2):

1. **ToyBench (Sieweke et al., NeurIPS 2026 workshop)**: only the listing abstract was seen. It evaluates SAEs on toy-model-of-superposition data across eight feature distributions, the closest published setting to S2-A. PDF should be on the OpenReview forum linked from https://neurips.cc/virtual/2026/166038 (my tool could not open OpenReview directly). Needed before relying on the "F1 <= 0.84" figure.
2. **Dead-latent resampling description (Anthropic, "Towards Monosemanticity" and the January 2024 circuits update)**: only needed if ReLU+L1 is chosen; the pages exceed my fetch tool's limit.
3. **Donoho and Elad (2003), PNAS, "Optimally sparse representation in general (nonorthogonal) dictionaries via l1 minimization"**: open on PubMed Central; my tool was blocked by a browser check. Only needed if the coherence argument in sections 2.2 and 2.4 is to be turned into a stated bound.
4. **Sparse-but-wrong (arXiv:2508.16560), SynthSAEBench (arXiv:2602.14687) and SAEBench (arXiv:2503.09532) appendices**: the last 14-37k characters of each were not read; they hold the L0-selection pseudocode and extra ablations.
