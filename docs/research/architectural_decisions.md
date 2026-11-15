# Architectural and Methodological Decisions

**Approved decisions: none.** Every item below is PROPOSED and awaits the user's approval (CLAUDE.md §9). A recommendation is not an approval.

Last updated: 2026-11-11 (session 1).

## Decision register

| ID | Decision | Status | Recommendation (short) |
|---|---|---|---|
| D-01 | Research question | Proposed | Narrowed question (research_question.md §3, option A) |
| D-02 | Training regime(s) and dataset | Proposed | Screen on TinyStories; primary regime decided after D-03 (see below) |
| D-03 | Compute plan (local vs rented) | Needs user input | Local screening at ~5–20M params; rented GPU only for confirmation |
| D-04 | First intervention set | Proposed | 1 SAE-derived loss + 1 non-MI regulariser + random-dictionary control |
| D-05 | Scope cuts | Proposed | Remove instruction tuning & deployment-as-research; postpone KD, attribution-based losses |
| D-06 | Primary outcome metric | Proposed | Held-out LM loss (nats/token on a fixed held-out split) |
| D-07 | Teacher / KD | Proposed | Postpone; no gpt-oss-120b logit KD |
| D-08 … | Architecture (layers, width, heads, context, positional encoding, norm, activation, init), tokenizer, optimiser, schedule, SAE architecture/placement | **Not yet proposed** | Phase 3 after a focused literature pass |

---

## D-01 — Research question

1. **Decision:** Replace the README's headline question with a narrowed, testable one.
2. **Context:** Every later choice (data, scale, metrics, controls) depends on what is being tested.
3. **Research evidence:** SUP_2025_002 (LMs already in strong superposition); SUP_2022_003, RAT_2025_004, RAT_2025_005 (interpretability ≠ capability); SAE_2025_003/007/008/009 (SAE signal reliability); RAT_2024_001 vs RAT_2025_003 (contested sign); RAT_2025_009 (interp signals work in fine-tuning).
4. **Approaches:** A narrowed SAE-signal question; B diagnostic-only; C quantisation-focused; D original scope (research_question.md §4).
5. **Advantages:** A keeps the core idea and is publishable either way; B safest; C best-grounded; D most ambitious.
6. **Disadvantages:** A likely null for capability; B drops "influence training"; C narrow, strong baselines exist; D infeasible to do rigorously.
7. **Recommendation:** A, with C's quantisation outcome and B's diagnostic question as secondary outcomes.
8. **Consequences:** KD and instruction tuning leave the core plan; the paper becomes a controlled empirical study.
9. **Approval:** requested.

## D-02 — Training regime and dataset

1. **Decision:** Which data regime(s) to study first.
2. **Context:** Regularisers can only help the LM metric through optimisation or generalisation; generalisation effects appear in data-limited, multi-epoch training.
3. **Evidence:** LMT_2023_002 (TinyStories: coherent < 10M models); LMT_2024_003 (BabyLM data-limited benchmark; more epochs helped); SAE_2025_005 and RAT_2024_008 used TinyStories for interpretability work; LMT_2024_004 (FineWeb single-epoch benchmark).
4. **Approaches:** (i) TinyStories; (ii) BabyLM 10M/100M-word corpus, multi-epoch; (iii) FineWeb/FineWeb-Edu subset, single epoch.
5. **Advantages:** (i) cheapest, interpretable features, fast iteration; (ii) established data-limited benchmark + venue, matches H2; (iii) standard compute-limited setting, matches H1.
6. **Disadvantages:** (i) narrow domain, results may not transfer; (ii) BabyLM evaluation pipeline overhead; (iii) needs more compute to be meaningful, regulariser effects least likely.
7. **Recommendation:** Pipeline development and SAE-loss pilots on (i). Choose the primary regime between (ii) and (iii) after the compute answer (D-03); my lean is (ii) because H2 is the more plausible positive outcome and the data budget fits a 6 GB GPU.
8. **Consequences:** Determines tokenizer, model size, epochs and evaluation suite.
9. **Approval:** requested (at least for "develop on TinyStories").

## D-03 — Compute plan

1. **Decision:** Where experiments run and the budget.
2. **Context:** A controlled study needs ~35–40 runs per size per regime (feasibility_assessment.md §2).
3. **Evidence:** Measured GPU here: GTX 1660 Ti 6 GB. Estimates in papers/language_model_training.md §2 (assumptions stated).
4. **Approaches:** local only; local screening + rented confirmation; rented throughout.
5. **Advantages:** no cost / balanced / fastest.
6. **Disadvantages:** limits to ~5–20M scale / some cost / recurring cost.
7. **Recommendation:** local screening + a bounded rented-GPU confirmation phase, budget to be set by the user.
8. **Consequences:** sets the maximum model size and number of seeds.
9. **Approval:** needs the user's hardware list (is the RTX 4050 laptop also available?), cloud budget and hours per week.

## D-04 — First intervention set (for the pilot in Phase 5)

1. **Decision:** Which auxiliary objectives to test first.
2. **Context:** Testing many at once destroys attribution; cheap ones first.
3. **Evidence:** DecPO decorrelation (RAT_2024_001); LoRA-to-fit-SAE (SAE_2025_002) as an "SAE-friendliness" loss; L2E (RAT_2025_003) opposite sign; CAFT random/PCA baselines (RAT_2025_009).
4. **Approaches (candidates, formulations to be finalised in Phase 2):**
   - (a) SAE-fidelity loss: penalise ‖x − SAE(x)‖² (or spliced-model KL) with the SAE frozen within a refresh window;
   - (b) SAE feature-interference loss: penalise co-activation-weighted decoder overlaps of active features;
   - (c) non-MI: raw-activation decorrelation ‖ẑẑᵀ − I‖²_F;
   - (d) control: (a) or (b) with a random or PCA dictionary.
5. **Advantages:** (a) simplest, closest to known work; (b) directly targets the README's "interference"; (c) the necessary cheap baseline; (d) specificity test.
6. **Disadvantages:** (a) may only make the model SAE-friendly (H6) with no capability effect; (b) most complex, depends on unstable SAE features; (c) not "MI"; (d) adds runs.
7. **Recommendation:** pilot (a), (c), (d) first; add (b) if (a) shows any effect or if the user prefers to test interference directly.
8. **Consequences:** fixes the Phase 5 run grid.
9. **Approval:** requested after D-01.

## D-05 — Scope cuts

1. **Decision:** Remove or postpone parts of the README.
2. **Context:** README contains ≥ 5 projects.
3. **Evidence:** feasibility_assessment.md §1, §5.
4. **Approaches:** keep all; cut as recommended; cut more (also quantisation).
5. **Advantages / disadvantages:** keeping all maximises breadth but makes every part shallow; cutting quantisation too saves little because PTQ is cheap and well-grounded.
6. **Recommendation:** remove instruction tuning and deployment-as-research; postpone KD and attribution/causal-intervention losses; keep PTQ measurement.
7. **Consequences:** roadmap below reflects this.
8. **Approval:** requested.

## D-06 — Primary outcome

1. **Decision:** What "performs better" means.
2. **Context:** README §7 lists eight outcomes without priority.
3. **Evidence:** SAE_2025_006/009 (representation metrics mislead); LMT_2024_004 (loss is the standard, variance-aware metric).
4. **Approaches:** held-out loss; benchmark accuracy (e.g. BLiMP/BabyLM suite); generation quality judged by an LLM.
5. **Recommendation:** held-out loss primary; a benchmark suite as secondary if regime (ii) is chosen; no LLM-judge primary metric.
6. **Approval:** requested.

## D-07 — Knowledge distillation

1. **Decision:** Keep, change, or postpone KD.
2. **Context:** README §5 proposes a gpt-oss-120b-class teacher via synthetic SFT data.
3. **Evidence:** KD_2025_001 (capacity gap; compute thresholds; logits needed); KD_2025_004 (201k vocabulary; 80 GB GPU).
4. **Approaches:** sequence-level KD from gpt-oss-120b; logit KD from a small-vocab teacher; postpone.
5. **Recommendation:** postpone; if revived, logit KD from a teacher sharing the student's tokenizer, with tokens and compute matched.
6. **Approval:** requested.

## User-stated direction (2026-11-11, recorded; formal approval of wording still pending)

These were stated by the user in conversation. They are recorded as the user's direction and override the conflicting parts of D-01, D-04, D-05 and D-07 above. Details still need §9.1 decisions before implementation.

| ID | Direction stated by the user | Effect on earlier proposals |
|---|---|---|
| U-01 | Goal is **more** superposition (more features per dimension) to let a smaller model match a larger one; final-model interpretability is not a goal | Replaces D-01 option A wording (see research_question.md §5) |
| U-02 | Penalty/reward lives **inside the training loop** as a loss term | Fixes the mechanism family for D-04 |
| U-03 | SAE reconstruction loss is considered; anything inserted into the forward pass uses **delta patching only** | Constraint on all SAE-related designs |
| U-04 | Quantisation **stays**: compare our model (≈ 2 GB → ≈ 1 GB quantised) vs a standard model (≈ 3 GB → ≈ 1.5 GB); success = similar performance | Reverses the D-05 suggestion to keep quantisation minimal; defines the headline comparison |
| U-05 | **Toy model first**, then small LMs | Adds a Phase 2b toy-model stage before Phase 3 |
| U-06 | **Smaller teacher** for distillation (later) | Updates D-07 |
| U-07 | Reuse standard components; new work = training path | Narrows Phase 3 |
| U-08 | **Proposed by the user:** train the SAE jointly in the same loop as the transformer | Discussed in D-09 below |

## D-09 — Joint SAE training in the same loop (user proposal; details pending)

1. **Decision:** How the measuring SAE is kept up to date while the model trains.
2. **Context:** An SAE fitted once goes stale as the model's representations change; the penalty would then be computed with an outdated ruler.
3. **Evidence:** SAE_2024_011 (continual SAEs along checkpoints); SAE_2024_001 (SAE trained against the model's output via KL); SAE_2025_003 (seed instability); SAE_2025_009 (metrics can look good on random models).
4. **Approaches:**
   - (a) Separate refits every N steps on cached activations.
   - (b) **Joint, two-objective loop (user's proposal):** every step, the SAE takes its own gradient step on reconstruction + sparsity (optionally + a KL term so splicing it preserves the model's output, as in SAE_2024_001), using *detached* activations; the model takes a step on LM loss + λ·penalty with the SAE held fixed for that backward pass.
   - (c) Fully coupled: one loss, gradients of the penalty flow into both the model and the SAE.
5. **Advantages:** (a) simple; (b) SAE always current, cheap in the toy model, no extra pipeline; (c) simplest code.
6. **Disadvantages:** (a) stale between refits; (b) model and SAE can co-adapt so the metric drifts away from reality — needs an independent check; (c) **the SAE would learn to report more superposition instead of measuring it** (Goodhart), so the penalty would no longer mean anything.
7. **Recommendation:** (b), with three safeguards: a stop-gradient so the penalty never updates the SAE; the penalty switched on only while the SAE's reconstruction quality is above a threshold (this is where the reconstruction loss enters, U-03); and an independent ruler — ground-truth features in the toy model, freshly trained SAEs (other seeds) on saved checkpoints in LMs.
8. **Consequences:** two optimisers and two learning rates; SAE learning rate likely higher so it keeps up; extra compute per step (small in the toy model, to be measured in LMs).
9. **Approval:** requested.

## Stage 0 decisions S0-1…S0-7 — APPROVED 2026-11-11

Full text and reasons: [phases/stage_0_toy_baseline.md](phases/stage_0_toy_baseline.md) §6. Approved by the user in session 1 ("Approve all S0 items").

| ID | Approved choice |
|---|---|
| S0-1 | Re-implement the Liu et al. toy model; reference code used only for reading and cross-checks (no licence) |
| S0-2 | Grids G-A (α = 1, 7 widths incl. 150, 7 γ, 3 seeds, 147 runs) and G-B (α ∈ {0.5, 1.5, 2.0}, 6 widths, γ ∈ {−1, 0.55}, 2 seeds, 72 runs) |
| S0-3 | Fixed seeds: 3 for G-A, 2 for G-B |
| S0-4 | Report γ = 0 and best-tuned γ per width; success claims only against best-tuned |
| S0-5 | Project venv with CUDA PyTorch, numpy, matplotlib (Python 3.13 used — see stage doc implementation note) |
| S0-6 | Git repository for this project; code committed before each run |
| S0-7 | User writes model, data sampler, negative-weight-decay optimiser; Claude writes runner, tests, analysis; mutual review |

Not covered by this approval: the Stage 0 gate thresholds (still a draft), D-01…D-07, D-09.

## Stage 1 candidate list (2026-11-14)

The user agreed to add a tree-structured toy variant (candidate 1b) to Stage 1 alongside the overlap-penalty candidate (1a). This approves the candidate's **inclusion in the plan only**. The penalty form, tree design and gate margins are not approved and need a decision in the section 9.1 format before implementation.

## Stage 2 approvals (2026-11-15, user, via question dialog)

Approved as proposed in [proposals/stage_2_design_proposal.md](proposals/stage_2_design_proposal.md):

| ID | Approved choice |
|---|---|
| D-09 | Joint SAE training, option (b): SAE takes its own step every model step on detached activations; the penalty never trains the SAE (stop-gradient); an independent ruler (true overlap in the toy, fresh SAEs on saved checkpoints) |
| D-S2-2 | The SAE reads the hidden activations h = xW only |
| D-S2-3 | Penalty = activation "leakage" (section 5.2 of the proposal) with all SAE quantities detached; matched-row penalty only as an optional diagnostic; co-activation form not tried |
| D-S2-1 | Run the pre-registered SAE bake-off (ReLU+L1, BatchTopK, JumpReLU) on frozen Stage 0/1 models, selection on seeds 0-2; the winner is chosen by the rule in D-S2-1 item 7 |
| D-S2-10 | Order: S2-0 build and unit tests, S2-A, S2-B, S2-C pilot, user checkpoint, S2-D final. Pilots on the local GPU, final on Kaggle. User approved "start building now"; the heavy runs (S2-A onward) are still announced to the user before they start. |
| Pass rule | Same bar as Stage 1: error more than 10% below the tuned baseline and more than 3 combined standard errors at 2 of 3 widths (15, 25, 63); Stage 2 also requires keeping about half of the oracle's gain (D-S2-8). This explicitly confirms the Stage 1 margin. |

Not yet approved (need discussion before the runs that depend on them): D-S2-4 (SAE schedule), D-S2-5 (frequency weights), D-S2-6 (carried arm: Stage 1 final shows the weighted arm winning at every width, so the proposal's "plain primary" recommendation is out of date), D-S2-7 (controls), D-S2-8 thresholds V1-V4 (to be confirmed after S2-A, then frozen), D-LM-1..16.

## Approvals of 2026-11-15 (later): D-S2-4 and the language-model direction (user, in chat: "Yes, approve both")

| ID | Approved choice |
|---|---|
| D-S2-4 | Joint training loop per step: (1) batch, (2) one model forward pass keeping the graph and the activations h at the chosen site, (3) task loss, (4) SAE step on a detached copy of h with its own optimiser (reconstruction + sparsity), (5) penalty computed with the just-updated SAE (SAE read-out without gradient; leakage penalty on the live h), (6) total loss = task + lambda x penalty, one backward pass, model optimiser step. The SAE trains from step 0; lambda stays 0 until the SAE reconstructs well, then is ramped in (the gate value is still a placeholder, to be calibrated after S2-A). SAE refresh gaps of 10 and 100 steps and a frozen SAE are measured as staleness controls (S2-C). |
| D-LM-1 (modified) | **Budget: no spending.** The literal 2 GB vs 3 GB test is dropped. The language-model study runs at 10.6M (R1) and 24M (R2) parameters with B = 1.5 x A on free or local compute (local GTX 1660 Ti, Kaggle/Colab T4). A 70M model is allowed only as one or two final confirmation runs after the small ones show an effect (about 55 h each locally, about 16 h on a T4; estimates). The 47M and 126M rungs and all rented compute are out. Claims will be stated "at 10-70M parameters". First step, no penalty involved: standard models at 10.6, 16, 24, 36M with 3 seeds to see whether a 1.5x size gap is visible above seed noise. |

Still unapproved: D-LM-2..16 (block design, tokenizer, corpus, optimiser, etc.), D-S2-5..8, the V1-V4 thresholds.

## D-LM-4 approved (user, in chat, 2026-11-15)

Custom byte-level BPE tokenizer with 16,384 entries, trained by us on a few GB of the main training corpus (using a standard library; the algorithm is not written by us), one tokenizer for all model sizes; loss reported as bits per byte so the tokenizer cannot bias comparisons. A ready-made GPT-2 tokenizer (50,257 entries) is a fallback only. 8,192 entries remains the alternative if the data-size fit is preferred. Reason given to the user: at 10-25M parameters a 50k vocabulary makes the embedding table 67% of the model and about two thirds of the per-token compute (my calculation).

## D-LM-2 and D-LM-5 approved (user, in chat, 2026-11-15)

| ID | Approved choice |
|---|---|
| D-LM-2 | Llama-style block as proposed: pre-norm RMSNorm, SwiGLU MLP (hidden size about 8/3 of d, multiple of 64), RoPE (theta 10,000), no bias terms, plain multi-head attention with head size 64. User asked for "the llama style transformer block" after the GPT-2 comparison. B is "standard", not "best possible"; the write-up will say so. |
| D-LM-5 | Option (ii): TinyStories for development, FineWeb-Edu (10B-token sample, only a slice is needed) for the main study. A fixed held-out slice of at least 5M tokens is cut before tokenisation and never trained on; optional second held-out set WikiText-103 test. No SlimPajama. |

Still open in the language-model list: D-LM-3 (tied embeddings), the model shapes (d, layers; proposal d = 256 with 8 layers for the 10.6M model), D-LM-6..16.

## Language-model baseline settings approved (user, in chat, 2026-11-15)

The user approved the ten-row table (their numbered replies 1-10 follow the table's row order; the replies were short, e.g. "ok", "good", "Yes"). One reply ("It takes 12 hours?????") was a misreading: 12 hours is Kaggle's maximum session length, not the training time; explained in chat.

| ID | Approved choice |
|---|---|
| D-LM-3 | Tied input/output embeddings. |
| D-LM-6 | Context 1,024 tokens; documents concatenated with an end-of-text token; no cross-document masking at first. |
| D-LM-7 | About 20 tokens per parameter as the primary regime (the proposal's option (b) adds a 100-per-parameter regime on the two smallest rungs, to be decided when the baseline ladder is done); equal tokens for every model within a size. |
| D-LM-8 | AdamW, betas (0.9, 0.95), eps 1e-8, gradient clip 1.0, linear warmup max(100 steps, 2%), cosine to 10% of peak, weight decay on weight matrices only. **Modified tuning rule (proposed by me, accepted with the table):** first tune only the peak learning rate (3 values between 6e-4 and 3e-3, 1 seed) at 10.6M, then reuse the best value for the larger sizes unless one diverges; widen the search later only if needed. |
| D-LM-9 | Batch size in tokens scaled with model size (about 65,536 tokens per step up to 50M parameters), reached with gradient accumulation. |
| D-LM-10 | fp16 on T4 / GTX 1660 Ti; one hardware class per comparison; every run records GPU, driver and library versions. |
| D-LM-11 | Every run directory stores config, git commit, seed, data order seed, hardware record, loss log, two rolling resumable states, milestone weights, final state, evaluation JSON; never overwritten. Needed because Kaggle sessions last at most 12 hours. |
| D-LM-12 | Primary metric: bits per byte on the held-out slice (at least 5M tokens). Secondary benchmarks only for the larger runs. |
| D-LM-13 | Train on Kaggle's free hours (and the local GPU); no rented compute. |
| D-LM-16 | Our own `lm/` code, with a later exporter to the Hugging Face Llama format. |

Still open: the model shapes (d, layers, heads, d_ff) for each size (proposal for 10.6M: d = 256, 8 layers, 4 heads, d_ff 704; for 24M: d = 384, 10 layers, 6 heads, d_ff 1024; 16M and 36M to be proposed), D-LM-14 (A-versus-B protocol), D-LM-15 (quantisation), D-S2-5..8, the V1-V4 thresholds.

## Change log

- 2026-11-15: language-model baseline settings D-LM-3, 6-13, 16 approved (section above).
- 2026-11-15: Stage 2 approvals above recorded.

- 2026-11-11: register created; nothing approved.
- 2026-11-11 (later, same session): user clarified the proposal; U-01…U-08 recorded; D-09 added.
- 2026-11-11 (later, same session): S0-1…S0-7 approved by the user.
- 2026-11-14: user agreed to add Stage 1 candidate 1b (tree-structured toy features); design not yet proposed.


## S0-8 — Step A scouting grid G-C (2026-11-15) — approved by the user's instruction to run it

Decision: add grid G-C to configs/stage0.json: alpha = 1, the same 7 widths as G-A, weight decay gamma in {-2.0, -1.5, -0.75, -0.3}, 3 seeds = 84 runs, to find where the best gamma lies before the multi-seed baseline runs. Reason: in G-A the best gamma was -1.0 (the edge of the grid) at m = 25-100 and -0.55 at m = 10, 15, 150. Evidence: stage_0_toy_baseline.md section 11. Run on a Colab T4 (full float32, so numbers stay comparable with G-A).
Not decided yet: the final seed count (proposed 10) and the final gamma set for it; both depend on G-C.
Note: the user asked for the run after being shown the four values and the plan; if they only meant the notebook, treat this entry as proposed.

## S0-9 — Step B baseline runs, grid G-D (2026-11-15) — approved by the user's instruction to set it up for two Kaggle accounts

Decision: add grid G-D to configs/stage0.json: alpha = 1, the 7 widths, gamma in {-2, -1, -0.55, -0.3, 0}, seeds 3-9 = 245 runs, fp32. Seeds 0-2 for the same gamma values already exist (G-A: -1, -0.55, 0; G-C: -2, -0.3), so pooled data is 10 seeds per (width, gamma) = 350 runs without repeating 105 runs. Baseline yardstick: best mean over these gamma values at each width; extra seeds only where still noisy. Reason: G-C showed a flat region between about -2 and -0.55 (stage_0_toy_baseline.md section 12). fp16 not used (only about 12% faster).
Caveats: seeds 0-2 come from other GPUs (GTX 1660 Ti, Colab T4) than seeds 3-9, so the pooled data mixes hardware; each saved run records its GPU. Not yet decided: whether more seeds are needed at m = 100-150, and the gate margin for the Stage 1 comparison.

## S0-10 — Baseline follow-up, grid G-E (2026-11-15) — approved by the user's instruction to run items 1 and 2 together

Decision: add grid G-E to configs/stage0.json, two rectangles, alpha = 1, seeds 0-9, fp32: (1) gamma in {-3, -4} at widths 63, 100, 150 (60 runs), to see whether more negative weight decay still helps where -2.0 was the best of the tested values in G-D; (2) width 225 with gamma in {-4, -3, -2, -1, 0} (50 runs), so that L*(1.5 x 150) can be computed for the Stage 1 target. Reason: stage_0_toy_baseline.md section 13. grid_configs now accepts a "parts" list of rectangles (45 tests pass). Estimated about 27 and 31 min per part on two T4s (width 225 speed is extrapolated). Not decided: which gamma values to use at width 225 beyond these five, and whether widths below 63 need gamma <= -3 (the G-D results suggest not).

## Stage 1 proposals D-S1-1 ... D-S1-6 — PROPOSED 2026-11-15, NOT APPROVED (write-up for the user; format of CLAUDE.md section 9.1)

**Context (all six).** Stage 1 is the go/no-go test of the central idea in the toy model: with the true features known ("oracle"), does a penalty that reduces feature overlap beat the best-tuned weight decay at the same width? Baseline and target: stage_0_toy_baseline.md section 13 (10-seed L*(m); the target is L*(1.5 m)); the G-E grid (gamma -3/-4 at m = 63-150 and width 225) is still to be run and may change L*(m) at wide widths. Prior art: prior_art_search.md section 6 (the penalty form already exists in the literature, so the contribution is the controlled comparison, not a new regulariser). Roadmap gate: G1 (research_roadmap.md).

### D-S1-1 Penalty form (candidate 1a)
- **Options:** (A) frame potential: lambda x mean over pairs i != j of cos^2(W_i, W_j) over all rows of W (the same overlap number the bench shows); (B) max-coherence (largest pairwise cosine); (C) hyperspherical-energy style spreading penalty; (D) A weighted by feature frequency, so frequent features are separated first.
- **Evidence:** form already used by Gao et al. 2019 (arXiv:1907.12009), Bank and Giryes (arXiv:1810.06049), Liu et al. MHE (RAT_2018_011, level F for pp. 1-3, 6-8), SUP_2026_004 (applied to an SAE decoder with the opposite aim), and as an illustration in SUP_2025_002 App. D.2. All are about generalisation or interventions; none tests it against tuned weight decay for packing.
- **Advantages / disadvantages:** A is simple, smooth, cheap, already measured by our code; its weakness is that the task loss may already optimise overlaps, so it may add nothing. B is non-smooth (harder to optimise). C needs a new hyperparameter. D adds a design choice that is easy to over-tune.
- **Recommendation:** A as the main arm, with two sub-arms: penalty alone (gamma = 0) and penalty plus norm pulling (gamma set to the best baseline value for that width), since reasoning says they push different levers. B, C, D only if A shows an effect.
- **Consequences:** about 20% extra time per training step (two n x n x m products per run; estimate, to be measured); needs a hand-written gradient in the batched trainer, checked against autograd; per-run lambda like the existing per-run gamma.

### D-S1-2 Penalty strength lambda
- **Options:** fixed hand-picked value; sweep on a log grid with a pilot; adaptive schedule.
- **Recommendation:** pilot sweep of lambda in {1e-5, 1e-4, 1e-3, 1e-2} (on the mean-squared-overlap scale, penalty value about 1/m; task loss is about 1e-4 to 1e-3) at 3 seeds to find the range where the penalty does something, then 3 values around that range in the final runs. Penalty strength is a tuned quantity, exactly as weight decay is in the baseline; fairness requires the same tuning effort.
- **Consequences:** the pilot grid is a guess about scale and may need one more round.

### D-S1-3 Widths and seeds
- **Recommendation (revised 2026-11-15 after a headroom check):** widths 15, 25 and 63, with 39 as a spare; pilot with seeds 0-2, final with seeds 3-9 (7 seeds), compared with the 10-seed baseline. The earlier suggestion (25, 63, 100) is replaced because of the headroom check below.
- **Headroom check (measured, G-A and G-D, gamma = -1):** mean squared overlap of the learned feature rows divided by the smallest value any arrangement can reach for n = 1000 rows in m dimensions (Welch bound, (n - m) / (m (n - 1)); standard result, not looked up this session): 3.05x at m = 10, 3.30x (15), 3.19x (25), 2.22x (39), 1.48x (63), 1.29x (100), 1.20x (150). So the arrangement has most room to improve at small widths and little at m >= 100, where a penalty on overlap cannot do much. The bound is not reachable for free (frequent features deserve cleaner separation than rare ones), so the ratio is an upper limit on the possible gain, not a prediction. Target error ratios to a 1.5x wider model: 1.47x (15), 1.71x (25), 1.86x (63).
- **Why separate seeds:** choosing lambda on the same seeds that are used for the verdict flatters the penalty (same logic as the baseline's best-of-gamma selection). Selecting lambda on seeds 0-2 and judging on seeds 3-9 avoids it.
- **Consequences:** widths 100, 150 and 225 are left out of Stage 1 (little headroom, and noise at m >= 100 is 20-28% per run).

### D-S1-4 Pass criterion (gate G1) — values are my suggestion; thresholds must be fixed before the final runs
- **Proposal:** G1 passes if, at 2 of the 3 widths, the penalty arm's mean test error on seeds 3-9 is below L*(m) by more than 10% and by more than 3 combined standard errors. Stronger success mark: reaches L*(1.5 m) within one standard error at any width.
- **Why 10%:** at m <= 63 the standard error of L* is about 2%, so 10% is about 5 standard errors; a smaller gain would be hard to separate from noise at m = 100.
- **Fail:** if no width passes, G1 fails and the main line stops with a written negative result ("even an oracle overlap penalty does not beat tuned weight decay in this toy model"); alternatives are in alternative_directions.md.
- **Inconclusive:** a gain of 3-10% or one width only is reported as inconclusive, not as a pass.

### D-S1-5 Order: 1a first, then the tree-structured toy (1b)
- **Recommendation:** run 1a first (hours of compute). Treat 1b as a measurement first: does the model already share parent directions without any penalty? Its design (tree depth, branching, parent frequency, penalty form) is a separate proposal after 1a.
- **Prior art:** Park et al. (RAT_2025_013) and Prieto et al. suggest models already organise hierarchy compactly, so 1b may find little to gain; it is exploratory.

### D-S1-6 What Stage 1 does not do
- No SAE (the true features are known), no real data, no language model, no quantisation. The penalty acts on the rows of W directly, which is the "oracle" case. The SAE-measured version is Stage 2 and is only started if G1 passes. A cheap quantisation-fragility check on packed versus unpacked toy solutions is recommended before Stage 2 (prior_art_search.md section 8) but is not part of G1.

### Estimated cost and order
Pilot: 3 widths (15, 25, 63) x 4 lambda values x 2 sub-arms x 3 seeds = 72 runs, about 30 min on a T4. Final: 3 widths x 3 lambda values x 2 sub-arms x 7 seeds = 126 runs, about 1 hour. Build and test the penalty first (gradient check against autograd; tests). Estimates use measured T4 speeds plus the 20% penalty overhead (not measured).

### Amendment to D-S1-1 / D-S1-2 (made while building the code; the user agreed in chat to build and pilot)
- **Weighted penalty promoted to a pilot arm.** The user asked why the frequency-weighted form (option D) should wait; running it later would cost another round trip, while as an extra arm in the same runs it costs about 50% more pilot runs. Arms per width: (1) plain penalty, gamma 0; (2) plain penalty with the best baseline gamma for that width (15: -0.55, 25: -1.0, 63: -2.0); (3) weighted penalty with the same gamma. The plain penalty stays as the control for the weighted one. Pilot grid G-S1P: 3 widths x 3 arms x 4 lambda x 3 seeds = 108 runs.
- **Weighted form (fixed, not tuned):** pair weight w_i w_j with w_i = sqrt(p_i) / mean(sqrt(p)), p_i the feature's activation probability. In this oracle toy p_i is known exactly; in Stage 2 it would have to be estimated from SAE activations, so a win here is an upper bound.
- **lambda pilot grid moved up one decade to {1e-4, 1e-3, 1e-2, 1e-1}.** Scale check at m = 25 on a trained baseline solution (gamma -1): lambda x |grad P| / |grad task loss| = 0.001 / 0.009 / 0.09 / 0.9 at lambda 1e-5 / 1e-4 / 1e-3 / 1e-2, so 1e-5 would be negligible and the interesting range probably extends above 1e-2.
- **Implementation details (not architectural):** dead rows: directions are taken with the row norm floored at 1e-2 so near-zero rows cannot blow up the gradient; the penalty gradient is added to the task gradient before Adam; the penalty is computed in float32.
- **Still open:** the pass margin (10% and 3 standard errors) must be confirmed before the final runs, not before the pilot. Gate thresholds in `configs/stage0.json` are still a draft.
- 2026-11-15 (later): D-S2-4 approved; D-LM-1 approved in modified form (10.6M and 24M ladder, no spending, 70M only as confirmation).
- 2026-11-15 (later): D-LM-4 approved (custom 16,384-entry byte-level BPE).
- 2026-11-15 (later): D-LM-2 (Llama-style block) and D-LM-5 (TinyStories for development, FineWeb-Edu for the main study) approved.
