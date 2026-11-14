# Research Gap Analysis

Rewritten 2026-11-14 for the goal as the user clarified it (research_question.md §5). The first version (2026-11-11) assumed the goal was to *reduce* interference with an SAE-derived loss; that framing is obsolete and is kept only where noted in §8. Sources: [literature_review.md](literature_review.md), [papers/](papers/), [packing_ideas.md](packing_ideas.md), [citation_index.md](citation_index.md).

**Verification level next to every claim** (definitions in [paper_library.md](paper_library.md)): **L** = full text read by us (best); **P** = full HTML text read through a summarising tool, numbers need spot-checking; **A** = abstract/metadata or secondary summary only, methodology not verified; **S** = search snippet only, treat as a pointer. Claims at A or S must not justify an approved decision. "My reasoning" means it is not from a paper.

Searches were not exhaustive. Every "not found" means "not found in the searches so far". The systematic prior-art search is a separate document (§7).

## 1. The goal being analysed

Push a transformer, with a penalty/reward inside the training loop, to pack **more concepts per dimension** (more superposition) than standard training does, accepting worse interpretability. SAEs are used only to **measure** how much superposition there is (jointly trained, held with a stop-gradient; anything inserted into the forward pass uses delta patching only). Test: a ≈ 2 GB model trained this way (A) should match a ≈ 3 GB standard model (B), also after quantisation. Toy model first, small language models later. Interpretability of the final model is not a goal.

In plain words: the standard model is a suitcase packed by an optimiser that already tries to fit as much as it can. The project asks whether an extra rule ("pack tighter") makes the same suitcase hold what a suitcase 1.5 times bigger holds.

## 2. Where the idea comes from, and what that implies

The idea is not new to this project. SUP_2025_002 (Liu, Liu & Gore, NeurIPS 2025, **L**, p. 10) says encouraging superposition "could enable smaller models to match the performance of larger ones", cites nGPT and its own FOCUS optimiser as possible examples, and **does not test it**. The same paper says the loss exponent cannot change for natural language ("No for natural languages", **L**); only the coefficient can. Also **L**: no LLM was trained with negative weight decay in that paper; its LLM analysis uses pretrained models' unembedding rows only.

So the project is a test of an untested suggestion from a strong paper. That is a legitimate reason to run it. It is not novelty by itself: a hypothesis stated in a paper and then tested is an empirical contribution only if the test is clean and the result informative either way.

## 3. What already exists, mapped to the new goal

| Element of the goal | Already done by others | Level | Not found (provisional) |
|---|---|---|---|
| Superposition controllable by training | Weight decay, including negative weight decay, switches the toy model between weak and strong superposition (SUP_2025_002) | L | Doing this on a real LM; any other in-loop knob compared against it |
| LMs already use superposition | LMs (OPT, GPT-2, Qwen, Pythia) are in the strong regime, width exponent 0.91 ± 0.04 (SUP_2025_002) | L (unembedding analysis only) | Evidence that more can be extracted |
| Loss law | Strong superposition: loss ∝ 1/width because squared overlaps of feature vectors scale ~1/width (SUP_2025_002) | L | Anything on cross-entropy; the paper itself lists squared error as a limitation |
| Packing penalty form | Frame-potential style penalty (squared off-diagonals of normalised W) is used by the authors to draw Fig. 2c (SUP_2025_002 App. D.2); minimum hyperspherical energy spreads neuron directions and reports better generalisation in vision (RAT_2018_011) | L / A | Either used to push LM feature packing for capacity |
| Hierarchy / shared parent | Parent-child concepts already have "parent + orthogonal difference" geometry in Gemma and LLaMA-3 (RAT_2025_013); hierarchy appears as absorption in SAEs (SAE_2024_004; Matryoshka SAE_2025_005); shared codebooks compress token embeddings post hoc (LMT_2025_009) | A | A training-time loss that makes a model share parents to save capacity |
| Capacity per parameter | ≈ 2 bits/parameter, holds at int8, domain-name prefixes raise it (LMT_2024_005); ≈ 3.64 bits/parameter for GPT models (LMT_2025_007) | A | Whether any training intervention moves these numbers and survives int4 |
| Training signal from interpretability | In fine-tuning only: CAFT ablates SAE/PCA directions (RAT_2025_009, **P**); preventative steering (RAT_2025_010, A); LoRA-to-fit-SAE (SAE_2025_002, A); gradient routing in pretraining (RAT_2024_008, **P**) | mixed | An online SAE measurement driving a capacity-oriented loss in pretraining from random init |
| Encourage vs suppress monosemanticity | Both signs reported to help (RAT_2024_001 decorrelation, fine-tuning, **P**; RAT_2025_003 inhibiting monosemantic neurons in Pythia pretraining, **P**, single runs, no seeds, no loss) | P | A seeded, compute-matched resolution |
| SAE as a ruler | ~30% of features shared across seeds (SAE_2025_003, A); metrics look similar on random-init and trained models (SAE_2025_009, A); proxy metrics do not predict usefulness (SAE_2025_006, A); SAEs often lose to simple baselines (SAE_2025_007/008, A) | A | A validated SAE-based measure of "features stored per dimension" |
| SAE trained along with the model | Series of SAEs along checkpoints (SAE_2024_011, A); SAE trained to match model output via KL (SAE_2024_001, A) | A | Joint training with the SAE feeding a loss back (stop-gradient) |
| Quantisation and features | Quantisation damages SAE features without changing perplexity (QNT_2026_003, **S**); training-time outlier suppression helps PTQ (QNT_2023_001 **S**, QNT_2025_002 **S**) | S | Whether *denser packing* makes features more quantisation-fragile (alternative direction A1: no systematic study found, S-level pointers only) |
| Interpretable-by-design costs capability | SoLU ≈ equal loss (SUP_2022_003, **P**); weight-sparse transformers need 100–1000× compute (RAT_2025_004, **P**); MoE-X mixed vs matched baseline (RAT_2025_005, **P**) | P | These push the opposite direction (more interpretable); they say nothing about less interpretable = more capable |

## 4. Why the core idea is hard (blunt)

These are the reasons I expect the main line to be difficult. They are my reasoning from the sources above, not published results.

**4.1 At the optimum an extra penalty cannot lower the training loss.** If the optimiser has found the minimum of the LM/toy loss, adding a regulariser moves you away from it (papers/superposition.md §2.3, formal and elementary). A gain can only come through (O) reaching a better point than standard training reaches in finite steps, (G) generalisation, or (R) something the loss does not measure. In the toy model fresh data arrives every step, so (G) is almost absent there. Stage 1 therefore tests (O) only: "does a push help the optimiser find a better packing?" If standard training with tuned weight decay already finds the best packing, there is nothing to win.

**4.2 Only the coefficient can move, and the target size is demanding.** In the strong regime loss ≈ C/m. Matching a model 1.5× wider means lowering C by a factor of 1.5 (my arithmetic from the 1/m law; valid only in the strong regime of the toy model). SUP_2025_002 (**L**) reports the measured mean squared overlap sitting above the best-possible (Welch-bound) 1/m line by roughly 1.5–2.5× at small α (read by eye off Fig. 17, approximate). That is the same order of size as the target, which is the only reason the toy test is worth running. It is a reading of a figure about overlaps, not a measured loss headroom, and the trained optimum need not be able to reach the bound, because the loss also weights features by frequency. Treat it as "not obviously impossible", not as encouragement.

**4.3 "More features stored" is already saturated in the toy model's strong regime.** Negative weight decay already gives nearly all features a dedicated direction (draft gate G0.1 expects φ½ ≥ 0.9 at γ = −1; not yet measured by us; regime description from SUP_2025_002, **L**). A penalty aimed at the *count* of represented features has nothing left to buy there. What remains is the *geometry* (how small the overlaps are, lever A in packing_ideas.md §0) and how well the nonlinearity cancels interference (lever B). So the realistic penalty targets overlap quality, not feature count, and a penalty that merely counts live SAE features measures the wrong thing.

**4.4 The mandatory competitor is weight decay, and it is free.** Weight decay, positive and negative, already moves the model between regimes (SUP_2025_002, **L**). Any penalty must beat the *best-tuned* weight decay at the same width (approved rule S0-4). Other cheap "superposition knobs" are untested competitors, not yet ruled out: optimiser choice (alternative direction A2; OSP says Muon removes privileged bases, QNT_2025_002, **S**), unit-norm architecture (LMT_2025_006, **A**, reports fewer steps, not more capacity per parameter). A penalty that works only because it acts like a finer weight-decay knob is not a finding.

**4.5 The ruler can be gamed and is unreliable outside the toy.** In the toy model ground-truth features are known, so the oracle can bypass the SAE entirely. In an LM there is no ground truth; the SAE is the only ruler, and SAE feature sets differ across seeds (~30% shared, SAE_2025_003, **A**) and are hard to tell from random-model SAEs on common metrics (SAE_2025_009, **A**). If the model is rewarded by an SAE that adapts to the model, the cheapest way to lower the penalty is to change what the SAE reports, not to store more (decision D-09 safeguards: stop-gradient, reconstruction-quality gate, independent ruler). This is the weakest link of the whole "interpretability-guided" claim.

**4.6 If an oracle or a non-SAE penalty works, the SAE is unnecessary, and the project is no longer "interpretability-guided".** The frame-potential penalty (candidate 1a) needs no SAE. That is a fine scientific result, but the SAE-based variants must then show something the non-SAE penalty does not (Stage 2 controls), or the SAE machinery is dead weight.

**4.7 Denser packing may make quantisation worse, which attacks the headline test.** More overlap means quantisation noise has less slack (hypothesis; alternative direction A1; related evidence is S-level: QNT_2026_003). The A-vs-B success criterion is stated *after* quantisation, so this single risk can erase a real pre-quantisation gain.

**4.8 Toy-to-LM transfer is unproven in both directions.** The toy has independent, atomic, known features with an average of one active feature per sample (stage_0_toy_baseline.md §2). Real LMs have correlated, hierarchical, unnamed features, and the authors list "tokens treated as atomic features" as a limitation (SUP_2025_002, **L**). A positive toy result would justify trying small LMs; it would not predict a result there.

## 5. What would NOT be a contribution

- A penalised model with a higher SAE-measured superposition score but no better held-out loss (a score changed, nothing stored; Goodhart).
- Beating an untuned baseline (S0-4: success only against best-tuned γ per width), or an under-trained 3 GB-class model.
- Wins at one α, one width or one seed only.
- "More interpretable features" by auto-interp without random-init controls (SAE_2025_009, **A**). Interpretability is not the goal here anyway.
- Comparing A at 2 GB with B at 3 GB at different token counts or compute; a data change dressed as a method change.
- A from-scratch small GPT (ubiquitous).
- A negative toy result presented as a negative result for the whole idea. It rules out the toy-scale mechanism and nothing more.

## 6. Candidate gaps for the new goal (ranked; provisional until §7 is answered)

**N1 — Does any in-loop pressure beat best-tuned weight decay at fixed width in the toy model? (strongest, cheapest, go/no-go)**
Oracle penalty with ground-truth features (candidate 1a overlap penalty; candidate 1b tree-structured features) against the tuned-γ yardstick L*(m) from Stage 0. Contribution type: empirical, either sign. A "no" is informative: "even a perfect guide cannot beat tuned weight decay in the toy model".

**N2 — Can an SAE stand in for ground truth as the ruler inside the loop? (methodological; the project's actual "interpretability-guided" part)**
Hypothesis HS5. Needs a rotation-invariant measure of features per dimension (Q-DES-1) and a differentiable route into the model's weights (Q-DES-2, unresolved). Only meaningful if N1 passes.

**N3 — Does denser packing cost quantisation robustness? (alternative direction A1; stands alone)**
Toy models first. Best-grounded link in the original README (training affects quantisation: QNT_2023_001, QNT_2025_002, both **S**), and it directly tests the A-vs-B criterion. A positive or negative answer is publishable either way and survives a failed N1.

**N4 — Shared parent plus small child offsets (user idea, candidate 1b)**
Tree-structured toy features. First measure whether standard training already shares parents (RAT_2025_013 says real LMs already do something like it, **A**), then whether a penalty adds anything.

**N5 — Bytes-matched A vs B in small LMs (HS3/HS4)**
Only if N1 and N2 pass. High cost, high noise (see feasibility_assessment.md §3).

**N6 — Cheap LM baselines: negative weight decay / unit-norm output rows on a real small LM (packing_ideas.md idea 2)**
Not tested in SUP_2025_002 (**L**). It is also the baseline any fancier LM method must beat. Low novelty if done alone, but it is the honest first LM experiment, and the LM analysis shows LMs are already in the strong regime, so I expect little room (reasoning, not a result).

## 7. Prior-art results (pending: see prior_art_search.md)

Not yet filled in. The prior-art search agent writes `docs/research/prior_art_search.md`; its findings must be summarised here when it exists. The questions it must answer, so that no claim of novelty is made before they are:

1. Has anyone trained a model (toy or real) with a regulariser or reward explicitly meant to **increase superposition, feature density or packing** so that fewer parameters match a larger model? Include negative weight decay on LMs, hyperspherical-energy / frame-potential / coherence penalties in LM pretraining, nGPT-type follow-ups that test the capacity claim, and any paper that tested the p. 10 suggestion of SUP_2025_002.
2. Has anyone used an **SAE-derived signal** (reconstruction error, L0, live-feature count, dictionary overlap) as a **training loss during pretraining from random init**? Any **joint** SAE+LM training with a stop-gradient or alternating scheme? (Q-LIT-2 search terms: dictionary-learning regulariser, sparse-coding auxiliary loss, SAE-regularised pretraining.)
3. What exactly does SUP_2026_004 (COLM 2026, orthogonality regulariser, currently **A**) regularise: model weights during training, or an SAE? Does it push the opposite way (less interference)? (Q-LIT-4.)
4. Is there a validated, rotation-invariant measure of "features stored per dimension" or "amount of superposition" for LMs, with or without SAEs? (Q-DES-1.)
5. Any systematic study of **superposition level versus quantisation robustness** (alternative direction A1), beyond the `tmsq` repository and QNT_2026_003 (both S-level)?
6. Training-time losses that make a model **share parent concepts** or exploit hierarchy to save capacity; toy models of superposition with tree-structured or correlated features.
7. Evidence on **knowledge capacity under int4/int8** and whether any training intervention raises bits per parameter in a way that survives quantisation (LMT_2024_005, LMT_2025_007 are A only).
8. Negative results: papers reporting that denser/more superposed representations hurt robustness, generalisation or quantisation (SUP_2022_001 links superposition to adversarial vulnerability, **A**).
9. Equal-bytes methodology: how others compare a smaller model to a larger one after quantisation (fixed-size Pareto frontiers, bits-per-parameter), so our A-vs-B design follows accepted practice.
10. Any report that dropout, optimiser choice (Muon, FOCUS) or architecture changes alter measured superposition (the blog-level hint "Superposition and Dropout" is not peer reviewed).

## 8. Disposition of the 2026-11-11 content

| Old item | Status |
|---|---|
| Evidence table "has this been investigated" (old §1) | Kept in substance in §3; reframed for the new goal |
| Old "What would NOT be a contribution" (old §2) | Updated into §5 |
| Old G1: SAE-derived loss vs non-MI regulariser, random/PCA dictionary | **Obsolete as the primary question** (it assumed interference reduction). Its controls are kept as Stage 2 controls (non-SAE penalty of equal cost, random dictionary) |
| Old G2: data-limited regime, regularisers as anti-overfitting | Obsolete for the main line (the goal is capacity, not generalisation). Remains a possible secondary mechanism (G) in LMs; H1–H7 in hypotheses.md are secondary |
| Old G3: feature-level objectives and quantisation | Kept, now central: N3 and the A-vs-B test |
| Old G4: do superposition metrics predict capability | Kept as a diagnostic by-product |
| Old G5: KD × representation interaction | Still postponed |
| Old recommended question (old §4, D-01 option A) | Superseded by the restated question in research_question.md §5 (wording not yet approved) |
| Old risk "prior art may exist" | Kept, now §7 |

## 9. Risks to the gap itself

- **Q-LIT-2 / §7:** a paper may already test the "encourage superposition" idea or use SAE signals in pretraining. Until §7 is answered, nothing here is a claim of novelty.
- The gap may be unexplored because it does not work (§4.1–4.3). The design must make a null result a clean answer: pre-registered gates, seeds, equivalence margins, yardstick from tuned weight decay.
- **Verification debt:** the arguments in §4.5 rest on level-A sources (SAE_2025_003, SAE_2025_009, SAE_2025_006); §4.7 rests on S-level quantisation sources; §4.2 rests on a figure read by eye. These should be upgraded (Q-LIT-1) before any of them is used to justify an approved decision.

## 10. Proposed decisions (not approved)

None of these is approved. Full §9.1-format text lives, or will live, in [architectural_decisions.md](architectural_decisions.md); this list only points to them.

- Restated research question wording (research_question.md §5; D-01).
- Which gaps to pursue and in what order: my lean is N1 first as a go/no-go, N3 as the hedge that survives a failed N1; nothing beyond N1 before G1 is decided.
- D-09 (joint SAE training with safeguards) and the unresolved measurement/penalty design (Q-DES-1, Q-DES-2).
