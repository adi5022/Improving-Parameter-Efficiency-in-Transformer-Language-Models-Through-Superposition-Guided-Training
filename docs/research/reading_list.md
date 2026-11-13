# Reading List (for the user)

Created 2026-11-11. Ordered by when each item is needed in the roadmap. All items are open access. For each item: what to read, what to extract, and questions to answer in your own words (write the answers in [reading_log.md](reading_log.md) — they become the basis for design decisions and the paper's related-work section).

Verification levels refer to how far *Claude* has checked each source (paper_library.md); your own careful reading upgrades them.

---

## Tier 1 — before Stage 0 code (≈ 1 week)

### R1. Toy Models of Superposition — Elhage et al., Anthropic, 2022 (technical report)
- Link: https://transformer-circuits.pub/2022/toy_model/index.html (arXiv:2209.10652)
- Read: introduction and key results; "Demonstrating Superposition"; "Superposition as a Phase Change"; "The Geometry of Superposition"; skim "Superposition in a Privileged Basis" and "Computation in Superposition"; skim the adversarial-examples section.
- Extract: the model h = Wx, x' = ReLU(Wᵀh + b); what feature *sparsity* and *importance* do; why the ReLU and bias let the model "filter out" interference; what makes a feature get dropped vs superposed.
- Questions:
  1. Why does superposition only pay off when features are sparse?
  2. What is the "interference" cost in the loss, and how does the ReLU/bias reduce it?
  3. In the phase diagram, which knobs move a feature from "not represented" to "superposed"? Which of those knobs could a training penalty act on?
  4. Why is the geometry (antipodal pairs, pentagons, polytopes) what it is — what is being optimised?

### R2. Superposition Yields Robust Neural Scaling — Liu, Liu & Gore, NeurIPS 2025
- Link: https://arxiv.org/abs/2505.10465 · code https://github.com/liuyz0/SuperpositionScaling
- Read: all of §1–§5 and Appendix B (toy training details); skim the LLM appendix.
- Extract: their toy setup (stage_0_toy_baseline.md §2 lists it — check it against the paper yourself); how weight decay, including the *negative* version, controls superposition; weak vs strong regime definitions; ϕ_{1/2}; the 1/m argument via overlaps.
- Questions:
  1. In your own words: why does strong superposition give loss ∝ 1/m almost regardless of the frequency distribution?
  2. What exactly does negative weight decay do to the rows of W, and why does it produce strong superposition?
  3. If LLMs are already in the strong regime, where could *our* penalty still gain? (Coefficient vs exponent; features that training fails to represent; optimisation speed.)
  4. What does the paper say could break or improve the scaling law? Do you agree?

### R3. ARENA exercises: Superposition & SAEs (chapter 1.3, hands-on)
- Link: ARENA curriculum, chapter 1, section 1.3 — 1.3.1 toy models of superposition, 1.3.2 SAEs (arena.education; mirrored on LessWrong: https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab). Check the official site for the current version.
- Do: the TMS exercises (non-privileged basis, privileged basis, feature geometry) and "SAEs in toy models".
- Why: you have used pretrained SAEs but not trained one or trained a toy model; this is the fastest route to both, and Stage 2 needs SAEs trained on toy-model activations.

---

## Tier 2 — before Stages 1–2 (≈ 2–3 weeks, can overlap with Stage 0 runs)

### R4. Polysemanticity and Capacity in Neural Networks — Scherlis et al., 2022 (arXiv preprint)
- Link: https://arxiv.org/abs/2210.01892
- Extract: "capacity" = the fraction of a dimension each feature consumes; how optimal capacity allocation depends on feature importance (important → own dimension, middle → shared, least → dropped).
- Questions: Is "push the model to give *more* features *some* capacity" compatible with their optimal-allocation result, or does their theory say training already allocates optimally? Under what conditions could it fail to?

### R5. Superposition, Memorization, and Double Descent — Henighan et al., Anthropic, 2023 (technical report)
- Link: https://transformer-circuits.pub/2023/toy-double-descent/index.html
- Extract: with limited data, models store *datapoints* rather than features in superposition; double descent between regimes.
- Questions: Could our penalty push a model to superpose memorised datapoints instead of general features? How would we detect that? (Relevant to the fact-storage test in Stage 4.)

### R6. Towards Monosemanticity — Bricken et al., Anthropic, 2023 (technical report)
- Link: https://transformer-circuits.pub/2023/monosemantic-features (a PDF copy is already in `FeatureScalpel/docs/`)
- Read: the SAE architecture and training sections; dead neurons and resampling; how they evaluate features.
- Questions: Which SAE training problems (dead latents, shrinkage, reconstruction error) would distort a *superposition measurement*?

### R7. Scaling and Evaluating Sparse Autoencoders — Gao et al., OpenAI, ICLR 2025
- Link: https://arxiv.org/abs/2406.04093
- Extract: TopK SAEs, the AuxK dead-latent loss, metrics (downstream loss, probe loss).
- Question: TopK vs ReLU+L1 for an SAE that must be retrained continuously inside the loop — which is more stable? (Compare with R8.)

### R8. Sparse Autoencoders Trained on the Same Data Learn Different Features — Paulo & Belrose, 2025 (preprint)
- Link: https://arxiv.org/abs/2501.16615
- Question: If two SAE seeds share ~30% of features, how stable is a *count* of features (our superposition measure) across seeds — more or less stable than the identity of features?

### R9. Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning — Braun et al., 2024
- Link: https://arxiv.org/abs/2405.12241
- Why: this is the precedent for your idea of training the SAE so it "suits the model's final output" (KL objective). Read the method and the cost section.
- Question: What does the KL term cost per step, and do we need it in the toy model where reconstruction is the model's own task?

### R10. A is for Absorption — Chanin et al., NeurIPS 2025
- Link: https://arxiv.org/abs/2409.14507
- Question: Would absorption make an SAE *under-count* or *over-count* features in superposition?

---

## Tier 3 — before Stages 3–5

### R11. Mathematical Models of Computation in Superposition — Hänni, Mendel, Vaintrob, Chan, 2024 (ICML 2024 Mechanistic Interpretability Workshop)
- Link: https://arxiv.org/abs/2408.05451
- Why: Stage 3 needs features that are *used*, not just stored. Extract their AND-computation construction and error-correction idea.

### R12. Physics of Language Models 3.3: Knowledge Capacity Scaling Laws — Allen-Zhu & Li, 2024 (preprint)
- Link: https://arxiv.org/abs/2404.05405
- Why: the ≈ 2 bits/parameter knowledge-capacity result and their synthetic-biography method — the template for the Stage 4 fact-storage test ("stores as much as a bigger model"). Note their claim that capacity holds under int8 quantisation — directly relevant to Stage 5.

### R13. Learning Towards Emergence (L2E) — Wang et al., 2025 (preprint)
- Link: https://arxiv.org/abs/2503.23298
- Why: the closest prior attempt to push models *away* from monosemantic neurons during pretraining. Read critically: no seeds, no loss reported (paper_library.md RAT_2025_003). What would you need to see to believe it?

### R14. How Quantization Changes Interpretable Features — Duan, 2026 (preprint)
- Link: https://arxiv.org/abs/2606.03002
- Why: protocol for measuring feature survival under quantisation (Stage 5, side direction A1).

### R15. Low-Rank Adapting Models for Sparse Autoencoders — Chen, Engels & Tegmark, 2025 (preprint)
- Link: https://arxiv.org/abs/2501.19406
- Why: an existing case of changing a model's weights to suit an SAE — the "gaming the ruler" risk in our D-09 design, seen from the other side.

---

## Suggested pace
Week 1: R1, R2 (with the questions answered) and start R3. Then we finalise Stage 0 decisions and you write the toy model. Tier 2 alongside Stage 0 runs.
