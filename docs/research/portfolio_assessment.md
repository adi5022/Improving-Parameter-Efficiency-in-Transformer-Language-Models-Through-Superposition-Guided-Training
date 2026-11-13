# Portfolio Assessment

Session 1 (2026-11-11). CLAUDE.md §7.4.

## 1. Value by target role

| Role | Value of the narrowed project | Why |
|---|---|---|
| Mechanistic interpretability research | **High** | Training-time use of SAEs, own SAE training, random-init and random-dictionary controls, honest negatives — exactly the questions the field is debating in 2025–2026 (SAE reliability, training-time interventions). Builds directly on FeatureScalpel without repeating it. |
| ML research / research engineering | **High** | From-scratch pretraining with tuned baselines, seeded comparisons, compute matching, pre-registered gates. |
| Small language models / model optimisation | Medium–high | Pipeline, scaling of small models, quantisation measurements. |
| LLM engineering / applied AI | Medium | Less product-facing; quantisation and a clean training stack are relevant. Instruction tuning and deployment were cut, so this angle is weaker than in the original proposal. |
| Training & inference systems | Medium | Mixed precision on constrained hardware, throughput benchmarking, PTQ. Not large-scale systems work. |

## 2. What a reviewer could verify in the final repository

- A from-scratch decoder implementation and training loop with configs, seeds and logs for every run.
- A baseline tuned with a documented sweep (not a single default run).
- Own SAE training code and SAE evaluation (fidelity, L0, dead latents, seed stability, random-init control).
- The auxiliary-loss implementations and their equal-cost non-MI and random-dictionary controls.
- Pre-registered gates committed before runs (the user's established FeatureScalpel practice), results tables generated from stored outputs, and negative results reported.
- PTQ results with feature-survival measurements.

## 3. What would weaken it

- Results only against untuned or external baselines.
- Interpretability claims without controls.
- A large KD/instruction/deployment surface with thin experiments underneath ("assembling libraries").
- Overclaiming a small, unseeded improvement.

## 4. Assessment

The narrowed project demonstrates substantially more than assembling libraries: it requires implementing training, SAEs, auxiliary objectives through custom hooks, and statistically careful evaluation. Its portfolio value does **not** depend on a positive result; it depends on rigour and clarity. The original broad proposal would likely have produced a shallower version of each part.
