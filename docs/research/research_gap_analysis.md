# Research Gap Analysis

> **Banner (2026-11-11, later in session 1):** this document was written before the user clarified that the goal is to push the model toward **more** superposition (research_question.md §5). Its evidence tables remain valid; its recommended question (§4) and gap G1 framing are superseded. The current plan is in research_roadmap.md (v2). A full revision is pending.

Session 1 (2026-11-11). Based on [literature_review.md](literature_review.md). Searches were not exhaustive; every "not found" is provisional until Q-LIT-2 is closed.

## 1. Has this exact idea been investigated?

**Not in its exact form, but every component has close precedent.**

| Question (CLAUDE.md §7.1) | Answer |
|---|---|
| Has MI been integrated into model training? | **Yes** — CAFT (RAT_2025_009), persona-vector preventative steering (RAT_2025_010), LoRA-to-fit-SAE (SAE_2025_002), gradient routing (RAT_2024_008), interpretable-by-design pretraining (RAT_2025_004, RAT_2025_005, RAT_2024_006, SUP_2022_003). |
| Have SAE-derived features modified optimisation? | **Yes, in fine-tuning**: CAFT ablates SAE latents (human-selected) during fine-tuning; SAE_2025_002 trains the LM to fit a frozen SAE. **Not found: an online SAE driving an automatic auxiliary loss during pretraining from random init.** |
| Do existing methods achieve similar representational goals without MI? | **Yes** — activation decorrelation (DecPO; classical DeCov), activation sparsity (TopK/ReLU), weight decay as a superposition knob, outlier-safe pretraining for activation statistics. |
| What is genuinely unexplored? | See §3. |
| Kind of contribution available | **Empirical** (controlled study with negative results allowed), possibly **methodological** (a cheap, stable online-SAE training signal), not architectural. |
| Defensible gap? | Yes, if narrowed (§4). Combining two research areas is not by itself a gap. |

## 2. What would NOT be a contribution

- A from-scratch small GPT (ubiquitous).
- Showing that an SAE-regularised model gets better SAE metrics (expected; SAE_2025_002 shows the model can be fitted to an SAE cheaply). This becomes a contribution only if accompanied by capability, generalisation or robustness outcomes against fair controls.
- Showing "more interpretable features" via auto-interp without random-init controls (SAE_2025_009).
- Beating an untuned baseline, or an external model trained on different data.
- Sequence-level distillation from a big teacher (data change, not a method).

## 3. Candidate gaps, ranked by defensibility

**G1 — Does an SAE-derived training signal do anything a cheap non-SAE signal does not? (strongest)**
Pretrain small LMs from random init with an auxiliary loss computed from an online SAE (e.g. reconstruction-friendliness, feature co-activation/interference penalties), and compare at equal FLOPs against: a tuned baseline; equal-cost non-MI regularisers (raw-activation decorrelation, activation L1/TopK, weight decay); and a *random or PCA dictionary* in place of the SAE. Outcomes: held-out loss, generalisation gap, SAE quality, quantisation robustness. This is the precise version of the README's question and directly reuses the user's "SAE vs random basis" control logic from FeatureScalpel.
- Prior expectation from the literature: little or no capability gain in the compute-limited regime (C1, C2). A well-controlled null result is still informative because the field currently has only small, unseeded positive claims of both signs (C3).

**G2 — Representation regularisers in the data-limited regime.**
Regularisers fight overfitting; a multi-epoch, fixed-data setting (BabyLM 10M/100M words) is where an auxiliary loss can plausibly *lower held-out loss*. Not found: monosemanticity/interference regularisers evaluated there.

**G3 — Feature-level objectives and quantisation.**
Known: activation-statistics training improves quantisation (QNT_2023_001, QNT_2025_002). Known: quantisation degrades SAE features invisibly to perplexity (QNT_2026_003). Not found: whether feature-level training objectives make *features* survive PTQ better than outlier-suppression baselines.

**G4 — Do superposition/interference metrics during training predict final capability? (diagnostic, cheap)**
README §7's correlation question. Can be answered as a by-product of G1/G2 across all runs and seeds; existing feature-dynamics work (SAE_2024_011, SAE_2025_012) describes dynamics but was not found to test predictive power across differently-regularised runs.

**G5 — KD × representation regularisation interaction.** Real but third-order; postpone.

## 4. Recommended narrowed research question (proposal — needs approval, decision D-01)

> **Can auxiliary training objectives derived from a sparse autoencoder trained online on a small Transformer's own activations improve its held-out language-modelling performance, generalisation, or quantisation robustness, relative to compute-matched, hyper-parameter-tuned baselines and to equal-cost regularisers that do not use interpretability tools?**

Secondary: do representation metrics (interference, feature stability, SAE fidelity) measured during training predict those outcomes across runs?

This keeps the README's core idea (interpretability as a training signal), removes the unsupported "substantially better" target, and makes a negative result an answer rather than a failure.

## 5. Risks to the gap itself

- **Q-LIT-2:** a 2025–2026 paper may already do G1. Three more targeted searches are scheduled before Phase 2 is closed; SUP_2026_004 (COLM 2026) must be read in full first because it is the most likely overlap.
- The gap may be "unexplored because it doesn't work". The design must make a null result publishable: pre-registered gates, equivalence margins, seeds.
