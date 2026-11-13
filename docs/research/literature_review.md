# Literature Review — Cross-Area Synthesis

First pass, session 1 (2026-11-11). Per-area detail: [papers/](papers/). Records: [paper_library.md](paper_library.md). Claims: [citation_index.md](citation_index.md).

**Scope and honesty statement.** 38 external sources are recorded (plus 17 identified but not yet examined). 30 were opened: 10 at level P (HTML full text through a summarising fetch) and 20 at abstract level (A); 8 are snippet-level (S) only. No PDF was read page by page. One extraction error by the fetch tool was caught (MoE-X table direction; corrected in paper_library.md), so level-P numbers must be spot-checked before use in a paper. Searches were broad but **not exhaustive**: "not found" below means "not found in these searches".

## 1. Seven conclusions

**C1. Trained LMs already exploit superposition; "more capacity per parameter by optimising superposition" has no positive evidence and a strong theoretical counter-argument.** LMs sit in the strong-superposition regime that produces loss ∝ 1/width (SUP_2025_002, NeurIPS 2025). The authors expect coefficients, not exponents, to be movable. Weight decay already controls the degree of superposition. → papers/superposition.md

**C2. Interpretable-by-design training is well explored, and it buys interpretability at equal or lower capability, never higher at matched compute.** SoLU (≈ equal loss, hidden features), weight-sparse transformers (100–1,000× compute for equal capability), MoE-X (better than non-matched GPT-2, mixed vs matched MoE), Monet and codebook features (≈ no or modest loss). → papers/representation_learning.md

**C3. The sign of the "monosemanticity ↔ performance" effect is not settled.** One line claims encouraging it (decorrelation, DecPO, EMNLP 2024, fine-tuning) helps; another claims inhibiting it (MEmeL; L2E during Pythia pretraining) helps. Both report small effects without seeds or variance in what was read. A careful study could settle this — a real, if modest, gap.

**C4. Interpretability signals already influence training — in fine-tuning, to control *what* is learned.** CAFT (ablate interpreted directions during fine-tuning; 10× less emergent misalignment, capabilities unchanged), preventative steering with persona vectors, LoRA-to-fit-SAE, and gradient routing during pretraining. The README's final aspiration (§9) is already partially realised; the unexplored part is *automatic, online, from-scratch* use aimed at capability or robustness.

**C5. SAE features are a noisy signal to put in a gradient.** ~30% feature overlap across seeds; absorption; proxy metrics not predictive; SAEs lose to simple baselines on probing and steering; auto-interp cannot tell trained from random models; features drift and form at specific times in training. The user's own FeatureScalpel results agree (the SAE helps but is not necessary). Any SAE-derived loss must be compared with a non-SAE loss of equal cost.

**C6. Distillation helps small students under specific conditions, and the proposed teacher is a poor fit.** Capacity gap and compute thresholds (KD_2025_001, ICML 2025); gpt-oss-120b's 201k vocabulary blocks practical logit KD into a small student; sequence-level KD confounds method with data.

**C7. Training-time representation changes are known to improve quantisation, and quantisation is known to damage features invisibly to perplexity.** Clipped softmax / gated attention and outlier-safe pretraining (ACL 2025) reduce outliers and improve low-bit quantisation; a 2026 preprint shows SAE features degrade under INT6–INT7 even when perplexity does not. This is the best-grounded link in the README.

## 2. What exists vs what the README proposes

| README element | Already done by others | Not found (candidate gap) |
|---|---|---|
| From-scratch small GPT | Ubiquitous (nanoGPT, modded-nanogpt, TinyStories, BabyLM) | — (engineering, not research) |
| Superposition-aware objectives | Toy-model control via weight decay (SUP_2025_002); orthogonality regulariser (SUP_2026_004, stage unclear) | Pretraining a real LM with an interference objective and measuring held-out loss vs tuned weight decay |
| Monosemanticity regularisers | DecPO (fine-tuning), L2E (pretraining, opposite sign) | Seeded, compute-matched test of either sign, against DeCov-style non-MI controls |
| SAE in the training loop | Frozen SAE + LoRA (SAE_2025_002); SAE/PCA directions ablated in fine-tuning (CAFT) | **Online SAE on the model's own activations driving an auxiliary loss from random init** |
| Feature dynamics during training | SAE-Track; crosscoders through time (ICLR 2026) | Using those dynamics as a *training signal* |
| KD + representation shaping | KD widely used; SAE-anchored continual learning (2026 snippet) | Interaction of KD with representation regularisers |
| Quantisation robustness via training | Clipped softmax, OSP | Whether *feature-level* objectives preserve features under PTQ better than outlier-suppression baselines |

## 3. Implications carried into the assessment

1. The research question must be **narrowed** and stated as a test, with "no improvement" as an acceptable, publishable outcome.
2. Controls that the literature shows are necessary: tuned baselines (weight decay, dropout, LR); non-MI regularisers of equal cost; random/PCA dictionaries vs SAE dictionaries; random-init controls for interpretability metrics; multiple seeds.
3. The most promising outcomes to measure are **generalisation in a data-limited regime** and **quantisation robustness / feature survival**, not raw capability at compute-limited scale.
4. KD should be postponed.

See [research_gap_analysis.md](research_gap_analysis.md), [feasibility_assessment.md](feasibility_assessment.md).
