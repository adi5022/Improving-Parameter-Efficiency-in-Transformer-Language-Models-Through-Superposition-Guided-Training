# Feasibility Assessment

Rewritten 2026-11-14 for the goal as the user clarified it (research_question.md §5): push a transformer, with an in-loop penalty measured by a jointly trained SAE, to pack more concepts per dimension, then test whether a ≈ 2 GB model trained this way (A) matches a ≈ 3 GB standard model (B), before and after quantisation. Toy model first. The 2026-11-11 version assumed a different goal; what survives from it is listed in §10.

Companion: [research_gap_analysis.md](research_gap_analysis.md) (why the idea is hard, prior-art questions). Plan: [research_roadmap.md](research_roadmap.md). Stage 0 details and measured speeds: [phases/stage_0_toy_baseline.md](phases/stage_0_toy_baseline.md).

**Verification level** next to each source claim: **L** full text read by us; **P** full HTML text read via a summarising tool (numbers need spot-checking); **A** abstract or secondary summary only; **S** snippet only. Definitions in [paper_library.md](paper_library.md). Claims at A or S are pointers, not grounds for an approved decision.

**Cost numbers come in three kinds and are labelled:** *measured* (in stage_0_toy_baseline.md §8–10 on this GTX 1660 Ti), *arithmetic* (derived by me from stated formulas; the formula is shown), *assumption* (not measured, to be benchmarked). Nothing below is a result of a project experiment: the Stage 0 grid has not been analysed and no Stage 0 outcome is used here.

## 0. Bottom line (blunt)

- **Stages 0 and 1 are cheap and well worth doing.** Measured speeds put the whole Stage 0 grid around an hour of GPU time, and Stage 1 at a few GPU-hours. They answer the core question ("does any pressure beat tuned weight decay at fixed width?") for almost no money.
- **I think the main line is more likely to fail or give a small effect than to deliver "model A matches a model 1.5× larger".** Reasons (details in gap analysis §4): an extra penalty cannot lower the loss at the optimum, only help the optimiser reach it; the toy model's feature count is already saturated by negative weight decay; only the loss coefficient can move, never the exponent; weight decay is a free competitor; denser packing may be more fragile under quantisation. This is judgement, not a measured result, and I give no probability.
- **The "interpretability-guided" part is the weakest link.** In the toy model ground truth makes the SAE optional. In language models, SAEs are unstable (≈ 30% of features shared across seeds, SAE_2025_003, **A**) and give no ground truth. If a non-SAE penalty works equally well, the project is real science but not "interpretability-guided".
- **The literal 2 GB vs 3 GB test cannot be run on this machine and probably not on a student budget.** It must be scaled down and kept as a ratio (§3).
- **Do not commit to Stages 3–5 or to cloud spending before the Stage 1 go/no-go (G1).**

## 1. Mechanism-by-mechanism feasibility

| Mechanism | Implementable with existing tools? | Main obstacle | Verdict |
|---|---|---|---|
| Toy model of superposition (Stage 0) | Yes, code written, 40 tests pass (stage doc §10.1) | Grid run in progress; reproduction gates not yet checked | **Feasible; results and gates pending** |
| Oracle penalty from ground-truth features (Stage 1; frame potential / coherence, candidate 1a) | Yes; a few lines on W; SUP_2025_002 builds Fig. 2c with such a penalty (**L**) | Can only help via the optimiser; must beat best-tuned γ | **Feasible and cheap; likely small effect (judgement)** |
| Tree-structured toy features (candidate 1b) | Yes (new data sampler) | Tree design open; may already be shared by standard training | **Feasible; design needed** |
| SAE trained in the same loop on detached activations, penalty with SAE held fixed (D-09 option b) | Yes in the toy (small); custom loop in LMs | Co-adaptation / Goodhart; two optimisers; dead latents; no ground truth in LMs | **Feasible in toy; risky in LMs** |
| Fully coupled gradient into the SAE (D-09 option c) | Yes | The SAE would learn to report more superposition instead of measuring it | **Not recommended** |
| Penalty as a differentiable function of activations that reaches the model's weights | Open design problem (Q-DES-2) | "More features per token" (denser activation) is not "more features stored"; counting live features measures the wrong thing | **Unresolved; the core design risk of Stage 2** |
| Delta patching of SAE output | Yes (user's FeatureScalpel experience) | Only needed if an SAE output is inserted into the forward pass; a measure-only penalty does not insert anything | Constraint, not an extra cost |
| Computation in superposition (Stage 3: tiny transformer, synthetic task with known features) | Possible | Building a task where feature use, not storage, matters is a research design problem in itself | **Feasible but under-specified** |
| Small LM from scratch (Stage 4) | Yes (PyTorch, nanoGPT-style) | Fair baseline tuning; first pretraining run for the user | **Feasible at ≤ 30M locally** |
| Logit distillation from the original gpt-oss-120b proposal | No in practice (201,088-token vocabulary, MXFP4 needs an 80 GB GPU: KD_2025_004, **S**) | Vocabulary and cost | **Dropped** |
| Distillation from a smaller, same-tokenizer teacher (U-06) | Yes | Third-order factor; doubles the grid | **Postponed** (§2.6) |
| PTQ (INT8, INT4 round-to-nearest) and feature survival | Yes, cheap | Protocol from QNT_2026_003 is S-level | **Feasible, minutes per model** |
| Quantisation-aware training | Possible | Extra training, extra confound | Not part of the core plan; decision needed |
| Attribution / causal-intervention-derived losses | Possible but one forward per feature | Cost and noise | **Postponed** |
| Instruction tuning, deployment-as-research | Yes | No research value | **Removed** |

## 2. Stage-by-stage assessment

### 2.1 Stage 0 — toy baseline and yardstick

- **Cost (measured):** the batched fused trainer takes about 17.4 / 30.2 / 36.4 ms per step for a group of 21 runs at m = 10 / 100 / 150, peak GPU memory 0.5 GB (stage doc §10.1). One-at-a-time it was about 3.7 ms per step per run, i.e. 1.2–1.4 min per 20,000-step run, about 4.5 h for both grids. Estimated G-A with batching: about 50–60 min (estimate from the measurements, not a measured full run). G-B (12-run groups) was not benchmarked; I expect the same order (guess).
- **Human cost:** reading R1–R2, reviewing the code written for the user, fixing gate thresholds before the run (draft gates G0.1–G0.4), analysis. The roadmap says 1–2 weeks.
- **Risks:** the reproduction gates may fail (G0.1–G0.3). If they do, Stage 1 does not start on an unverified baseline. The reference code has no seeds and no separate test loss, so our variance numbers are new. Gate thresholds are still a draft.
- **What it gives:** the yardstick L*(m) (best loss over weight decay γ at width m) and the seed spread, which sets the margin for every later claim.
- **Honest assessment:** low cost, low risk, and it already produces a defensible small result (a verified reproduction with variance).

### 2.2 Stage 1 — oracle penalty, go/no-go (G1)

- **Task:** with ground-truth features known, test whether a "pack more, keep them distinguishable" penalty (candidate 1a overlap penalty; 1b tree variant) beats best-tuned weight decay at fixed width m, then whether the width-m model reaches L*(1.5·m).
- **Cost (arithmetic, illustrative grid, not approved):** 7 widths × 7 penalty strengths × 5 seeds = 245 penalty runs, plus topping the weight-decay baseline from 3 to 5 seeds (the roadmap's G1 asks for ≥ 5 seeds and G-A has 3): 7 × 7 × 2 = 98 runs. Total ≈ 343 runs ≈ 15 groups of ≤ 24. At 17–36 ms per step a group takes 20,000 × 17–36 ms ≈ 6–12 min. The overlap penalty (Gram matrix of the normalised rows, n² · m ≈ 10⁸ multiply-adds per run per step at m = 100, forward) adds roughly 25% FLOPs against the stage doc's ≈ 2.5 GFLOP step estimate (arithmetic; wall-clock unmeasured). Result: about **2–4 GPU-hours** per full grid. A tree variant or a second α roughly doubles it. Cheap enough to repeat.
- **Hidden trap — the γ grid is coarse.** G-A has 7 γ values. If the best γ lies between grid points, a penalty that effectively adds a finer knob "wins" without being a new mechanism. The baseline must be refined around its best γ before any claim.
- **Another trap — penalty as weight decay in disguise.** An overlap penalty also changes row norms. A norm-matched control (an extra weight-decay variant of equal strength) is needed to show the effect is not merely that.
- **Risk of the idea (judgement):** most likely outcomes are "no gain beyond tuned weight decay" or "a small gain that does not close the 1.5× gap". Both are clean, reportable answers.
- **Human cost:** penalty design in §9.1 format (not yet proposed), gate margins from Stage 0 variance. 1–2 weeks by the roadmap.

### 2.3 Stage 2 — SAE-measured penalty with joint SAE training

- **Cost (arithmetic, unmeasured):** a toy SAE on the m-dimensional hidden vector with d latents costs about 2·B·m·d multiply-adds per step forward (B = 2,048). For m = 100 and d = 2,000 that is ≈ 8 × 10⁸ MACs forward, ≈ 5 GFLOP with backward, against ≈ 2.5 GFLOP for the toy step itself. So a step is likely **2–3× slower** (memory traffic may change this). With a 24-run group at ≈ 40–90 ms per step, a group takes 14–30 min; a grid with several penalty strengths, SAE settings and the controls (random dictionary, non-SAE penalty of equal cost, SAE-free oracle) is plausibly **of order 5–20 GPU-hours**. SAE hyper-parameter iteration will cost more human time than GPU time.
- **Main risks:** (1) the SAE must keep up with a moving model (higher SAE learning rate, dead latents, refresh); (2) the model can lower the penalty by changing what the SAE reports rather than what it stores (D-09: stop-gradient, reconstruction gate, independent ruler); (3) the penalty design is unresolved (Q-DES-1, Q-DES-2); (4) with ≈ 1 active feature per sample (stage doc §2) the toy SAE problem is much easier than in LMs, so success here says little about LMs.
- **Validity check that cannot be skipped (HS5):** the SAE's superposition estimate must track ground truth in the toy model.
- **Likely outcome (judgement):** if G1 passes, a working SAE-driven penalty recovering part of the oracle's gain is plausible but not assured; if it only matches a non-SAE penalty, the SAE adds nothing.

### 2.4 Stage 3 — toy model that computes

- Cost: not estimated; the compute is small, the unknown is the design of a task where packed features must be *used* downstream. The reading list's Tier 3 covers computation in superposition (not yet read). I would treat the design as 3–4 weeks of human effort (the roadmap's figure) with real risk of overrun.
- Risk: the gain may vanish once features must be read out, because overlap that costs nothing in a pure reconstruction task now costs accuracy downstream (hypothesis).

### 2.5 Stage 4 — small language model

**Budget on this machine.** GTX 1660 Ti, 6 GB, Turing: no tensor cores, no bf16, no FlashAttention-2 (stage doc §10.1 notes the lack of tensor cores). Throughput is an **assumption** of about 3 TFLOP/s effective, to be benchmarked. Training compute C ≈ 6·N·D (parameters N, tokens D).

| Model / tokens | 6·N·D (arithmetic) | Time per run at 3 TFLOP/s (assumption) | Verdict |
|---|---|---|---|
| 5M params, 100M tokens | 3 × 10¹⁵ | ≈ 0.3–0.5 h | Local, comfortable |
| 10–15M params, 200M tokens (illustrative) | 1.2–1.8 × 10¹⁶ | ≈ 1–2 h | Local; a 40-run grid ≈ 40–80 GPU-h |
| 20–30M params, 0.5B tokens | ≈ 6–9 × 10¹⁶ | ≈ 6–10 h | Reduced grid only |
| 124M params, 2.5B tokens | ≈ 1.9 × 10¹⁸ | ≈ 170 h | Not local; rented |
| ≈ 1B params (what "2 GB" means at 2 bytes per weight) | any realistic token count | — | **Impossible here**: fp32 weights + gradients + Adam states ≈ 16–20 bytes per parameter (rule of thumb) ≈ 16–20 GB before activations, against 6 GB |

Memory at 124M is ≈ 2–2.5 GB for states, so it probably fits with a small micro-batch (untested).

**Run count for an honest A-vs-B test (illustrative).** Arms: B (standard, 1.5·P), C (standard, P), A (penalty, P), D (non-SAE penalty, P), E (random-dictionary penalty, P), plus standard "ladder" sizes between 0.75·P and 2·P to calibrate equivalent size. With ≥ 3 seeds that is about 24 runs, plus baseline tuning (≈ 6–10) and penalty-strength tuning (≈ 6): **≈ 40 runs.** At the 10–15M scale above that is 40–80 GPU-hours locally (arithmetic from 6·N·D, unbenchmarked); at 124M-class on a rented A100/H100 the earlier estimate was 1.5–4 h per run, i.e. about 60–160 GPU-hours (unverified; no price check done; ask before spending, Q-HW-2).

**The measurement problem is as large as the compute problem.** The loss gap between a model and one 1.5× larger is small compared with the variance of single runs: the strongest baseline recipes require p < 0.01 across seeds to count as records because of seed variance (LMT_2024_004, **S**). The size of this gap and of the seed spread at our scale is unknown and must be measured (a width ladder with seeds) before the test is designed. If the gap is not several times the seed spread, "A matches B" is untestable at that scale.

**Tokens per parameter** are an open choice (D-02/D-03). Do not set it from memory; Chinchilla is on the "not yet examined" list (Q-LIT-3).

**Risks:** SAE measurement unreliable without ground truth; penalty effect confounded with norm or weight-decay effects; the model that is 1.5× larger is also easier to quantise (more redundancy) than the packed one (hypothesis).

### 2.6 Distillation

- Evidence: KD beats plain pretraining only below a student-compute threshold and when a teacher already exists; a too-strong teacher hurts (capacity gap); logit KD needs a shared vocabulary (KD_2025_001, ICML 2025, **P**). Gemma 2 used KD in pretraining (KD_2024_003, **A**).
- The user's direction (U-06) is a smaller teacher. Feasible in principle: a same-tokenizer teacher costs one extra forward pass per batch, roughly 2·N_teacher·D FLOPs (arithmetic). A teacher 3× the student's size would about double the training cost (6·N_s·D student plus 2·3·N_s·D teacher). A teacher must exist at that scale with a compatible tokenizer; training one costs more than the experiment. Pre-computing top-k logits needs storage.
- **Confound:** distillation must be applied identically to A and B, or the A-vs-B comparison tests distillation, not packing. That doubles the grid. How teacher supervision interacts with packing is not something I examined; no source read.
- **Verdict:** postpone. Third-order, doubles runs, not in the minimum or realistic scope.

### 2.7 Quantisation

- Cost: round-to-nearest INT8/INT4 PTQ on a ≤ 100M model takes minutes and little GPU. Measuring SAE feature survival after quantisation (QNT_2026_003 protocol, **S**: INT7 degraded 18.7% of active features on Gemma-2-2B while perplexity was unchanged) needs frozen SAEs; modest.
- The user's illustrative sizes (2 GB → 1 GB, 3 GB → 1.5 GB) imply the same 2× compression for both models. That is a *same-ratio* comparison. A stricter comparison is *equal bytes*: A quantised to size S against any standard model reaching size S, including a larger standard model quantised more aggressively. Without the equal-bytes arm, A could "match B" while both lose to a bigger-but-more-compressed standard model.
- Known positive links: training-time activation-statistics control helps PTQ (QNT_2023_001 **S**, QNT_2025_002 **S**). Known unknown: whether more overlap raises quantisation damage (alternative direction A1; no systematic study found; S-level pointers only). Int8 knowledge capacity is reported to hold (LMT_2024_005, **A**); int4 was not examined.
- **Likely outcome (judgement):** quantisation shrinks, not widens, any small packing gain, so a marginal pre-quantisation win will probably not survive INT4.

## 3. The A-vs-B test: design and outcome rules

All of this is a proposal; metric, margin and sizes need approval (Q-DES-5).

**Scaling the sizes down.** 2 GB / 3 GB is illustrative (research_question.md §5). Keep the **ratio** (B = 1.5 × A in parameters) and run it at a size that can be repeated with ≥ 3 seeds (§2.5): toy width m vs 1.5·m (Stages 0–3), then ≈ 10–15M vs ≈ 15–22M parameters locally, with one rented 124M-class confirmation at most.

**Arms.** B = standard at 1.5·P, tuned. C = standard at P, tuned (including weight decay sweep, negative included; this is the real "no method" baseline for A). A = penalty at P. D = non-SAE penalty of equal cost. E = random-dictionary penalty. Ladder = standard models at several sizes with seeds, giving a loss-versus-size curve L_std(p). Equal tokens and tuning effort across arms; FLOPs of the SAE and penalty reported, and an equal-FLOPs comparison given as well as equal steps.

**Primary outcome: held-out loss** (D-06, not approved). Secondary: task accuracy; synthetic fact-storage bits per parameter (LMT_2024_005 protocol, **A**); SAE-based representation metrics are descriptive only, never primary.

**Two derived numbers.**
- **Fraction of the gap closed:** f = (L_C − L_A) / (L_C − L_B). f = 0 means the penalty did nothing; f = 1 means A matches B (the user's criterion).
- **Equivalent size ratio:** ρ = s\*/P, where s\* is the standard size whose loss on the ladder equals L_A. The user's criterion corresponds to ρ ≥ 1.5.
- **Resolution check first:** compute L_C − L_B and the seed spread before any penalty run. If the gap is not clearly larger than the spread, f is meaningless (division by noise) and the test is *inconclusive by design* at that size.

| Outcome | Toy (Stage 1, G1) | LM (Stage 4) | After quantisation (Stage 5) |
|---|---|---|---|
| **Success** (the user's criterion) | Penalty at width m reaches L\*(1.5·m) within seed noise (f ≈ 1, ρ ≥ 1.5) over several widths, ≥ 5 seeds, at α = 1 and at least one other α, against the *best-tuned and finely refined* weight decay | f ≈ 1 with CI excluding 0 and excluding "just norm effects" (control D), ≥ 3 seeds, FLOPs reported | Same f computed on the quantised models, and also against an equal-bytes standard model |
| **Partial** | Clear gain over best-tuned L\*(m) (f > 0, CI excludes 0) but f clearly below 1 | Same | Same; gain survives but is smaller |
| **Failure** | No gain over best-tuned weight decay at any strength (f ≈ 0 or < 0), or gain only against untuned γ | A no better than C beyond noise, or only as good as D or E (then the SAE adds nothing) | Gain present before quantisation disappears after it (f_q ≈ 0), or A degrades more than C under the same PTQ |
| **Inconclusive** | L\*(m) − L\*(1.5·m) not resolvable against seed spread; Stage 0 gates failed; result flips with α or width; effect smaller than γ-grid spacing artefacts | Gap between sizes within seed spread; fewer seeds than needed; baseline under-tuned | PTQ method differences larger than the effect |

Numeric margins (for example, what lower confidence bound on f counts as "success") are to be fixed in Phase 2 pre-registration, using Stage 0's seed variance (G0.4). I do not propose numbers before that variance is measured.

## 4. Scopes

Durations assume 15–20 h/week, which is unconfirmed (Q-SCOPE-1). Compute figures are the estimates above.

| Scope | Content | GPU need | Duration (judgement) | Difficulty | A reviewer could verify |
|---|---|---|---|---|---|
| **Minimum viable** | Stage 0 reproduction with variance; Stage 1 oracle go/no-go (candidates 1a, optionally 1b) against tuned weight decay; N3 in the toy (superposition level vs quantisation damage); short write-up | Laptop; ≈ 10–30 GPU-hours total | ≈ 1.5–2.5 months by the roadmap's per-stage figures (1–2 weeks per stage) plus reading and review | Moderate | Seeded, pre-registered toy experiments; clean positive or negative answer; tests; reproducible configs |
| **Realistic** | Minimum + Stage 2 (jointly trained SAE penalty with the controls and the HS5 validity check) + Stage 3 + the cheap LM baselines (negative weight decay on a small LM) + N3 on small LMs | Laptop; ≈ 100–200 GPU-hours (rough sum) | ≈ 4–6 months | High | Everything above plus a measured SAE-vs-ground-truth validity result and a first LM test |
| **Ambitious** | Realistic + Stage 4 A-vs-B with arms C/D/E and ladder, + Stage 5 quantised A-vs-B with equal-bytes arm, one rented 124M-class confirmation, optionally KD | Laptop plus ≈ 60–160 rented A100/H100 hours (unverified estimate; cost unpriced) | ≈ 7–10+ months; the roadmap's 4–6 months for Stages 0–5 is a best case that assumes every gate passes first time (judgement) | Very high; recurring cost | A scaled A-vs-B with seeds, or a documented negative result at each gate |

If G1 fails, the project stops the main line and writes up "even a perfect guide cannot beat tuned weight decay in the toy model"; alternative directions A1 (quantisation fragility) or A2 (Muon vs AdamW packing) become the main project. That outcome is a legitimate, completed piece of work, not a collapse.

## 5. Main ways the project could fail or be uninformative

Ordered roughly by how worried I am.

1. **G1 fails: no pressure beats best-tuned weight decay.** My expectation is that this is more likely than a clean 1.5× result (judgement). Clean negative; reportable.
2. **Apparent win from a weak baseline.** Coarse γ grid, 3 seeds instead of 5, untuned learning rate, or the penalty acting as extra weight decay. Mitigations: refine γ, top-up seeds, norm-matched control.
3. **Resolution failure.** The gap between size 1.5·P and P is too small relative to seed spread at feasible scale, so "A ≈ B" is satisfied trivially or cannot be tested. The most likely way for the LM stage to be uninformative. Mitigation: resolution check first (§3).
4. **The SAE ruler does not track truth (HS5).** In the toy model this is checkable; in LMs only against fresh SAEs with other seeds, which themselves overlap only ≈ 30% (SAE_2025_003, **A**).
5. **SAE adds nothing.** Penalty from a random dictionary or a non-SAE statistic does as well (controls D and E). Then the project is not MI-guided.
6. **Toy-to-LM transfer fails.** Independent atomic features at E = 1 active feature per sample versus correlated, hierarchical real features.
7. **Quantisation erases the gain**, or denser packing is more fragile (hypothesis, S-level support).
8. **Penalty design never closes** (Q-DES-1/2): no differentiable measure of "features per dimension" that is rotation-invariant and not gameable.
9. **Optimiser conflicts in joint training:** two loops, two learning rates, co-adaptation.
10. **Compute/budget:** the LM stage needs ≈ 40 runs per size; ≥ 124M-class confirmation needs rented GPUs (Q-HW-1/2 unanswered).
11. **Stage 0 reproduction fails** (G0.1–G0.3), blocking everything until debugged.
12. **Scope creep** back to KD, instruction tuning or deployment.
13. **Prior art appears** (gap analysis §7 pending).

**What an uninformative result looks like:** a toy gain at one α and one width; a gain smaller than the γ grid spacing; an LM gap smaller than seed noise; a gain before quantisation that nobody checked after quantisation; a penalty equal to a random-dictionary control. Each should be flagged as "inconclusive", not "promising".

## 6. Research and portfolio value (short)

- **Research:** an empirical test of a suggestion in a NeurIPS 2025 paper (SUP_2025_002, **L**). A negative toy result alone is modest but honest; a positive one, or a clean quantisation-fragility answer (N3), would be more useful. Realistic venues are workshops (mechanistic-interpretability, efficient-ML). A main-conference paper would need a clear, scaled positive result.
- **Portfolio:** value comes from the rigour (pre-registered gates, tuned baselines, seeds, equivalence margins, a NeurIPS result to be reproduced if the Stage 0 gates pass) and from implementation depth (custom batched trainer with a hand-written fused backward, negative weight decay optimiser, tests), not from the headline claim. See [portfolio_assessment.md](portfolio_assessment.md). I have not repeated its analysis here.
- **Learning:** superposition mechanics, controlled experiments at training time, SAE training, quantisation measurement.

## 7. Recommendation

*(Recommendation, not an approval.)* **Proceed with Stage 0 and Stage 1 as a bounded, cheap go/no-go; do not commit now to Stages 3–5, to rented compute, or to distillation.** Keep N3 (superposition vs quantisation fragility) as the hedge that survives a failed G1. Decide at G1, with the Stage 0 variance in hand, whether Stage 2 is worth starting. This is worth roughly two months of focused effort as a self-contained project whatever the outcome; beyond that it is a conditional bet whose most likely result, in my judgement, is a small or null effect. It is not worth planning around the expectation that a 2 GB model will match a 3 GB one.

## 8. Proposed decisions (not approved)

None of the following is approved. Each needs the full §9.1 format in [architectural_decisions.md](architectural_decisions.md) (Decision, Context, Research evidence, Approaches, Advantages, Disadvantages, Recommendation, Consequences, Approval) before anything is implemented. This list is only a pointer.

| Item | Where it lives | My lean (recommendation only) |
|---|---|---|
| Restated research question wording | D-01; research_question.md §5 | Adopt the restated question |
| Stage 0 gate thresholds and the Stage 1 G1 margin | stage_0_toy_baseline.md §4 (draft) | Fix before the first real run, using seed variance |
| First Stage 1 penalty family and tree design (1a/1b) | roadmap Stage 1; packing_ideas.md; not yet registered | Frame potential first; add norm-matched and finer-γ controls |
| Joint SAE training with stop-gradient, reconstruction gate, independent ruler | D-09 | Option (b) |
| Penalty design and rotation-invariant measure of superposition | Q-DES-1, Q-DES-2 | Unresolved; needs a literature pass |
| Definition of "performs similar" for A vs B (metric, margin, f and ρ, equal-bytes arm) | Q-DES-5; D-06 | f and ρ with a resolution check; pre-register margins |
| Scaled-down proxy for 2 GB vs 3 GB (keep the 1.5× ratio) | research_question.md §5 | Keep ratio, run at ≈ 10–15M locally |
| Data regime, token budget, tokenizer | D-02 | Develop on TinyStories first |
| Compute plan and rented-GPU budget | D-03; Q-HW-1/2 | Local screening, bounded rented confirmation, only after G1 |
| Whether distillation enters at all, and teacher | D-07; U-06 | Postpone |
| Quantisation method and format; PTQ only vs QAT | not yet registered | PTQ (RTN INT8/INT4) first |

## 9. Dependencies and likely bottlenecks

- Stage 0 gates pass; user review of the generated core files; Q-HW-1/2 (GPUs, budget); Q-SCOPE-1/2 (hours, solo or team); prior-art search results (gap analysis §7).
- Bottlenecks: tuning and refining the baseline; penalty design (Q-DES-2); seed count; the resolution check at LM scale; a rented-GPU budget for any confirmation.

## 10. Disposition of the 2026-11-11 content

| Old item | Status |
|---|---|
| Mechanism table (old §1) | Rewritten for the new goal (§1); KD, instruction-tuning and attribution rows kept |
| Compute table by scale (old §2) | Kept with its assumptions (3 TFLOP/s, 6·N·D) and extended (§2.5); still unbenchmarked |
| Run-count estimate (≈ 35–40 runs per size per regime) | Kept (§2.5, ≈ 40) |
| Research value (old §3) | Folded into §6 |
| Scopes and durations (old §4) | Replaced by §4, which uses the new stage structure |
| Old recommendation: narrow to an SAE-derived loss question vs non-MI regularisers | **Obsolete** (it assumed interference reduction). Its controls survive as Stage 2 controls |
| Old "remove from README" list | Still valid: remove "substantially better performance per parameter" as a target, instruction tuning, deployment-as-research, gpt-oss-120b logit KD; postpone attribution losses |
