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

- SUP_2026_004 (COLM 2026, **abstract only**): orthogonality regularisation on a feature dictionary, bounded interference propagation, "preserving model performance". Not yet known whether this is applied to the model's weights during training or to an SAE. **Must be read in full before the gap analysis is final** (Q-LIT-1).
- A non-peer-reviewed blog post found in search (Substack, "Training for Interpretability: Lessons from Anti-Superposition and Sparsity Constraints") reports that an orthogonality penalty decorrelated neurons without reducing polysemanticity, and TopK activation gating reduced polysemanticity ~30% at < 4% capability cost. Snippet-level only; not cited as evidence.

## 4. Open items

- Read SUP_2022_001 sections on computation in superposition and on the effect of feature importance.
- Read SUP_2026_004 in full.
- Read Jermyn et al. 2022, Scherlis et al. 2022, Henighan et al. 2023 (not yet examined).
- Decide which superposition/interference *measurements* are meaningful for a trained small LM (Phase 4): unembedding/embedding overlap statistics as in SUP_2025_002; SAE decoder overlap; activation decorrelation as in RAT_2024_001.
