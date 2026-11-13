# Alternative and Follow-up Directions

Session 1 (2026-11-11). Requested by the user: under-researched directions at the same level, worth time **alongside** the main plan or **after** it (including if the main idea is disproved). None is approved; novelty is based on a handful of searches (non-exhaustive) and all new sources below are snippet-level (S).

Main plan for reference: push a model to store more features per dimension via an in-loop penalty measured by a jointly trained SAE; toy model first (research_question.md §5).

## Ranked suggestions

### A1 — Does more superposition make a model more fragile to quantisation? (recommended side branch)
- **Question:** As the degree of superposition rises (controlled by width, weight decay, or our penalty), does post-training quantisation hurt more? Theory suggests it might: quantisation noise adds to the interference between overlapping feature directions.
- **What exists:** QNT_2026_003 (feature survival under RTN quantisation in Pythia-70M / Gemma-2-2B; does not vary superposition). A GitHub replication project, `fqybaba/tmsq` (not peer-reviewed), quantised toy models and found which superposed features break first; the author calls the effect borderline. No systematic study found.
- **Why it fits:** reuses the toy-model infrastructure of the main plan; directly tests the risk to the user's A-vs-B quantised success criterion; stands on its own if the main idea fails.
- **Cost:** low (toy models on CPU/GPU; small LMs later).

### A2 — Does the optimiser shape superposition? (Muon vs AdamW)
- **Question:** Do matched models trained with Muon vs AdamW differ in superposition statistics (SUP_2025_002 overlap/row-norm measures), SAE quality (with random-init controls), and quantisation robustness?
- **What exists:** OSP (QNT_2025_002) says Muon removes privileged bases and reduces outliers; a 2026 preprint ("Muon Learns More Robust and Transferable Features than Adam", arXiv:2606.09658, S) reports higher effective rank and robustness in classification; SUP_2025_002 names weight-decay-free optimisers as possibly relevant. **No study found that measures superposition or SAE features under Muon vs Adam.**
- **Why it fits:** if the optimiser already changes packing, it is a cheap, non-interpretability "superposition knob" — a baseline the main method must beat — and a clean, publishable comparison either way.
- **Cost:** low–moderate (pairs of small training runs).

### A3 — Feature-level distillation through cross-model translators (builds on FeatureScalpel)
- **Question:** Instead of distilling outputs, give the student an auxiliary loss to represent the *teacher's* SAE features, mapped into the student's residual stream by a learned translator (the user's cross-model maps).
- **What exists:** SAE-FD (2026 preprint, S) uses SAE features as anchors in continual learning, not teacher→student; vision has sparse-representation matching (arXiv:2103.17012, S). **No LM teacher→student SAE-feature distillation found** (two searches).
- **Why it fits:** reuses the user's unique Export/Import experience (FeatureScalpel's softer Import raised Small's held-out accuracy 8.5% → 9.5%); also answers the main plan's hardest question — *which* features to pack — by taking them from a bigger model. Natural replacement for plain KD in the late optional phase.
- **Cost:** moderate–high; higher risk.

### A4 — What does scale buy at the feature level? (crosscoder diffing, small vs larger model)
- **Question:** Which features does a larger model of the same family represent that the smaller one lacks, merges, or stores with more interference? Are the small model's errors due to *missing* features (capacity) or *crowded* ones (superposition)?
- **What exists:** Anthropic's crosscoder paper (Transformer Circuits 2024, S) includes a "model diffing over layer and scale" section; a distilled-reasoning-model study (arXiv:2503.03730, S) suggests larger distilled models have more unique features. Partly explored; no clean answer found.
- **Why it fits:** diagnostic evidence for the main hypothesis; reuses GPT-2 Small/Medium and the 780 facts Medium knows but Small does not (FeatureScalpel Import set).
- **Cost:** moderate (training crosscoders).

### Not recommended as a primary direction
- **How superposition develops over pretraining checkpoints:** already covered by Ge et al. (SAE_2025_012, ICLR 2026: crosscoders over Pythia checkpoints, two-phase dynamics, compression then expansion of feature dimensionality). Useful as context, low novelty.

## How they fit together

| If the main plan… | Best follow-up |
|---|---|
| Is running | A1 and A2 as cheap side experiments on the same infrastructure (A2 also supplies a baseline) |
| Succeeds in the toy model | A3 (tell the student which features to pack), A1 at LM scale |
| Fails the oracle gate | A1 or A2 as the main project; the failed gate is itself a reportable negative result |

## Sources added this step (all level S, unverified)
- `github.com/fqybaba/tmsq` — toy-model superposition + RTN quantisation replication (code repository, not peer-reviewed)
- arXiv:2606.09658 — Muon Learns More Robust and Transferable Features than Adam (preprint)
- transformer-circuits.pub/2024/crosscoders — Sparse Crosscoders for Cross-Layer Features and Model Diffing (Anthropic technical report)
- arXiv:2503.03730 — Towards Understanding Distilled Reasoning Models: A Representational Approach (preprint)
- arXiv:2605.25525 — SAE-FD: Sparse Autoencoder Feature Distillation for Continual Learning (preprint)
