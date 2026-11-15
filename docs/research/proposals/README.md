# Proposals written overnight (2026-11-15): plain-language summary

Nothing here is approved. Each decision is a proposal in the CLAUDE.md section 9.1 format; the long documents hold the evidence.
The lead (main session) spot-checked the compute arithmetic (the 2 GB cost figures follow from 6 x parameters x tokens and a 40% A100 utilisation) and the Welch-floor numbers; it did not re-verify every source.

## 1. Stage 2: make the penalty work without knowing the true features
Files: [stage_2_design_proposal.md](stage_2_design_proposal.md) (decisions D-S2-1..10), [stage_2_sources.md](stage_2_sources.md).

- **The problem.** In Stage 1 the penalty acts directly on the known feature arrows. In a real model the features are unknown, so a sparse autoencoder (SAE) has to find them first.
- **Key design point (the agent's inference, not from a paper).** A penalty built only from the SAE's decoder directions, with the SAE kept as an independent ruler, gives the model no signal to learn from. So the proposal penalises *leakage*: how much each activation points at dictionary directions that should be off. The SAE supplies the directions and which ones are active; all SAE quantities are detached. In the idealised case the expected gradient equals the Stage 1 pair penalty weighted by feature frequency.
- **Controls proposed:** oracle (true features), a scrambled SAE, a frozen SAE, an SAE-free batch-coherence penalty, and the tuned weight-decay baseline.
- **Unanswered by the literature:** does SAE-measured overlap track true overlap near the minimum? Only an experiment can say; it is the first Stage 2 step.
- **Cost estimate:** about 6 to 7 GPU hours in total (unmeasured). Every success threshold is a placeholder.
- **Only started if Stage 1 passes (gate G1).**

## 2. Standard language model (model B) and the A-vs-B protocol
Files: [lm_baseline_proposal.md](lm_baseline_proposal.md) (decisions D-LM-1..16), [lm_sources.md](lm_sources.md).

- **The literal 2 GB vs 3 GB test is out of reach.** At 20 tokens per parameter, a 2 GB model is 1.0B (fp16), 1.9B (int8) or 3.3B (4-bit) parameters. One training run costs about USD 580 to 6,200 of rented A100 time, with no tuning and no seeds, and the 3 GB case costs USD 1,300 to 14,000 per run. Training state for even the smallest reading needs 16 to 24 GB, which fits neither the 6 GB card nor a 16 GB T4.
- **Proposed instead:** keep the size *ratio* (B = 1.5 x A) on a ladder of smaller models: 10.6M, 24M, 47M and 126M parameters. The full ~40-run rung costs about 50 h on your GPU (10.6M), 77 h on a T4 (24M), about USD 100 rented (47M), about USD 370 on an A100 (126M).
- **Cheapest first step, no penalty involved:** train standard models at 10.6, 16, 24 and 36M with 3 seeds each (about 75 h on the 1660 Ti, 22 h on a T4). It shows whether a 1.5x size gap is even visible above seed noise at these scales.
- **Size of the effect needed:** an illustration (extrapolated beyond the range the curve was fitted on) says matching a 1.5x larger model needs about 2 to 4% lower loss.
- **Main recommendations:** Llama-style blocks, tied embeddings, a 16,384-entry byte-level tokenizer trained on the corpus, TinyStories for development and FineWeb-Edu (10B-token sample) for the ladder, 1,024 context, 20 tokens per parameter, AdamW with tuned learning rate, bits per byte as the main metric, GGUF (llama.cpp) quantisation as the deliverable.
- **Needs from you:** a spending ceiling (renting is only proposed for the largest rung, after Stage 1 and Stage 2).
- **Earliest realistic start:** about 2 to 3 weeks after approval for the first meaningful baseline results.

## 3. Pipeline code (built, tested, placeholders only)
`lm/` and `tests/lm/`: a configurable decoder-only trainer with exact resume, a manifest per run and a hook for an extra loss. Every architecture and training value in it is a placeholder, not an approved choice. 163 tests pass (51 existing, 112 new). fp16 and all CUDA paths are untested because the GPU was busy.

## Decision that changes the project
Dropping the literal "2 GB vs 3 GB" comparison in favour of a ratio-preserving ladder (D-LM-1) is a change to the research plan and needs your explicit approval.
