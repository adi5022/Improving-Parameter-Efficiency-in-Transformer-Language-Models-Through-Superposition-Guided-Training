# Superposition and Representational Capacity

Session 1 (2026-11-11). Records: [../paper_library.md](../paper_library.md). Verification levels apply.

## 1. What is established

**Superposition exists and is useful to the network.** Toy models show that when features are sparse, a network with m hidden dimensions stores more than m features along non-orthogonal directions, accepting some interference in exchange for representing more of the data (SUP_2022_001, abstract-level). The trade-off is set by feature sparsity and importance; there is a phase change between "drop the feature" and "superpose it".

**Superposition is what makes width scaling work (strongest relevant result).** Liu, Liu & Gore (SUP_2025_002, NeurIPS 2025, read at level P):

- Toy model: features x_i = u_i·v_i, u_i ~ Bernoulli(p_i), v_i ~ U(0,2), p_i ∝ i^(−α); hidden h = Wᵀx ∈ ℝ^m; output y = ReLU(Wh + b); loss ⟨‖y − x‖²⟩.
- Weight decay γ is the control knob: positive γ → weak superposition; small or *negative* γ (descending on (‖W_i‖ − 1)²) → strong superposition.
- **Weak superposition:** only the top ≈ m features are represented; loss ≈ sum of frequencies of the dropped features → power law in m with exponent α − 1 only if frequencies are power-law.
- **Strong superposition:** many more features represented, row norms bimodal (≈ 0 or ≈ 1); mean squared overlap of normalised rows ∝ 1/m, so loss ∝ 1/m (α_m ≈ 1) almost regardless of the frequency distribution. Conjecture for skewed frequencies: α_m ≈ 2(α − 1).
- LLMs: unembedding rows of OPT, GPT-2, Qwen, Pythia show the strong-superposition signature (overlaps ∝ 1/m); width exponent fitted α_m = 0.91 ± 0.04, and 0.88 ± 0.06 from Chinchilla sizes.
- Authors' implications: the width exponent is probably not changeable for natural language; encouraging superposition might move coefficients, not exponents; it may make interpretability harder.

**Architecture changes polysemanticity but not capability.** SoLU (SUP_2022_003, level P) raised the share of quickly-explainable neurons from ~35% to ~60% at approximately equal loss, but partly by shrinking non-neuron-aligned features and recovering them through the following LayerNorm. The features did not disappear; they became harder to see in the neuron basis.

## 2. What this means for the project

1. **The README treats superposition as an inefficiency to optimise. The best current evidence says trained LMs are already in the strong-superposition regime and that this regime is the source of their favourable width scaling.** "Higher representational capacity per parameter" by *adding* superposition is what gradient descent already does; reducing it trades capacity away.
2. **Interference is the price of capacity, not a separable defect.** In the toy model, loss in the strong regime *is* the interference term (overlap ∝ 1/m). An objective that reduces interference at fixed width must either drop features or spread them more evenly (a geometry change). The second is the only way it could help, and it would need the network's own optimiser to be failing at it.
3. **A clean consequence (formal):** at a minimiser of the training LM loss L, adding λR (any auxiliary regulariser) cannot lower L. A regulariser can only improve the *reported* metric through (a) optimisation dynamics (finite steps, non-convexity — reaching a better point faster), (b) generalisation (held-out loss in a data-limited, multi-epoch regime), or (c) properties L does not measure (quantisation robustness, editability, OOD behaviour). Every hypothesis in this project must name which of the three it relies on. See [../hypotheses.md](../hypotheses.md).
4. **Weight decay is already a superposition knob.** Any "superposition-control" intervention must be compared against simply tuning weight decay (SUP_2025_002). This is a cheap, mandatory baseline.
5. **SoLU warns about measurement:** an intervention can improve a neuron-basis interpretability metric without changing the underlying representation. Measurements must be basis-independent where possible (e.g. SAE-based or probe-based, plus random-init controls — SAE_2025_009).

## 3. Closest work on controlling interference by optimisation

- SUP_2026_004 (COLM 2026, **read in full text, level F, 2026-11-14**): the orthogonality penalty λ‖tril(DᵀD)‖²_F (Eq. 1) is applied to the **decoder of a TopK SAE**, not to the language model's weights. The SAE is fine-tuned with the penalty (Gemma 2 2B layer 13, 65,536 latents), frozen, spliced in, and the LM is LoRA-fine-tuned around it on MetaMathQA (1 epoch). So it is **fine-tuning, not pretraining**, and the LM is made to fit the SAE, as in SAE_2025_002. Evidence: orthogonality loss falls with λ (Fig. 2); GSM8K accuracy unchanged within bootstrap noise (Fig. 3); on a 12-name swap intervention the intended name appears 74.9% of the time at λ = 1e-4 vs 62.2% at λ = 0 (Fig. 4b); the two intermediate λ values are *worse* than λ = 0 on ROUGE-L recall and JSD (Table 1). The theoretical bound (Thm. 4.1) is shown by the authors to be loose (Fig. 6). No repeated training runs. Direction: it **reduces** overlap to get isolated interventions, the opposite of this project's goal; it lists "does more orthogonality reduce superposition?" as an untested follow-up. It does not overlap with gap G1.
- A non-peer-reviewed blog post found in search (Substack, "Training for Interpretability: Lessons from Anti-Superposition and Sparsity Constraints") reports that an orthogonality penalty decorrelated neurons without reducing polysemanticity, and TopK activation gating reduced polysemanticity ~30% at < 4% capability cost. Snippet-level only; not cited as evidence.

## 4. Open items

- SUP_2022_001: pages 3-17 of the arXiv PDF were read on 2026-11-14 (definitions, basic results, mathematical understanding, phase change, start of uniform superposition). **Still unread:** the rest of the geometry section, correlated/anti-correlated features, privileged basis, computation in superposition, adversarial examples. Verified there: a linear readout never superposes; the ReLU readout does; ‖W‖²_F counts features learned and D* = m/‖W‖²_F is "sticky" at 1 and 1/2; the toy-model authors say it is "very unclear what to generalize to real networks".
- SUP_2026_004: done (above). Jermyn et al. 2022 (SUP_2022_008) and Scherlis et al. 2022 (SUP_2022_005) examined at level P: in toy models the same loss can be reached with very different degrees of (mono/poly)semanticity depending on initial bias and learning rate (path dependence, Sec. 4.1); capacity theory says the loss already allocates capacity by importance and kurtosis, with partial capacity predicted only for sparse enough features (Sec. 3.4-3.6; analytic model uses quadratic activations). Henighan et al. 2023 still not examined.
- Still open: whether *more* superposition than gradient descent finds is reachable in the toy model at unchanged loss — the Jermyn et al. result (same loss, different basins) is the closest evidence that the degree of superposition is not fully determined by the loss.
- Decide which superposition/interference *measurements* are meaningful for a trained small LM (Phase 4): unembedding/embedding overlap statistics as in SUP_2025_002; SAE decoder overlap; activation decorrelation as in RAT_2024_001.
