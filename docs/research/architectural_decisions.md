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

## Change log

- 2026-11-11: register created; nothing approved.
- 2026-11-11 (later, same session): user clarified the proposal; U-01…U-08 recorded; D-09 added.
- 2026-11-11 (later, same session): S0-1…S0-7 approved by the user.
- 2026-11-14: user agreed to add Stage 1 candidate 1b (tree-structured toy features); design not yet proposed.


## S0-8 — Step A scouting grid G-C (2026-11-15) — approved by the user's instruction to run it

Decision: add grid G-C to configs/stage0.json: alpha = 1, the same 7 widths as G-A, weight decay gamma in {-2.0, -1.5, -0.75, -0.3}, 3 seeds = 84 runs, to find where the best gamma lies before the multi-seed baseline runs. Reason: in G-A the best gamma was -1.0 (the edge of the grid) at m = 25-100 and -0.55 at m = 10, 15, 150. Evidence: stage_0_toy_baseline.md section 11. Run on a Colab T4 (full float32, so numbers stay comparable with G-A).
Not decided yet: the final seed count (proposed 10) and the final gamma set for it; both depend on G-C.
Note: the user asked for the run after being shown the four values and the plan; if they only meant the notebook, treat this entry as proposed.
