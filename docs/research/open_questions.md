# Open Questions

Lifecycle: Open → Researching → Answered (questions are never deleted). Last updated 2026-11-11.

## For the user (blocking)

| ID | Question | Status | Blocks |
|---|---|---|---|
| Q-HW-1 | Which GPUs are available? This machine has a GTX 1660 Ti 6 GB; FeatureScalpel ran on an RTX 4050 laptop 6 GB. Any university/lab GPUs? | Open | D-03 |
| Q-HW-2 | Is there a budget for rented GPUs or teacher-model API calls, and roughly how much? | Open | D-03, D-07 |
| Q-SCOPE-1 | Hours per week and target end date? Is a paper submission (which venue/deadline) a goal? | Open | Roadmap timing |
| Q-SCOPE-2 | Is this a solo project, or will FeatureScalpel teammates participate? | Open | Planning, authorship |
| Q-BG-1 | Please confirm/correct the background summary in phases/phase_00_project_understanding.md §5 (which FeatureScalpel parts you personally built). | Open | Learning plan |
| Q-DEC | Decisions D-01 … D-07 in architectural_decisions.md. | Open | Phase 2 |

## Literature (non-blocking for discussion; blocking for approved decisions)

| ID | Question | Status |
|---|---|---|
| Q-LIT-1 | Upgrade central (★) sources from A/S/P to verified full-text reads: SUP_2026_004 (first), SUP_2025_002 remainder, SAE_2025_002, SAE_2025_003, SAE_2025_009, RAT_2025_004 capability numbers, RAT_2024_006, RAT_2023_007, KD_2025_001 App. A & G.4, QNT_2023_001, QNT_2025_002, QNT_2026_003. All are open access; **no user-provided PDFs needed so far.** | **Partly done 2026-11-14.** Read page by page (level F): SUP_2026_004, SAE_2025_002, SAE_2025_003, SAE_2025_009, RAT_2018_011 (pp. 1-3, 6-8), SUP_2022_001 (pp. 3-17). Read as HTML full text (level P): SAE_2024_001, SAE_2024_004, SAE_2024_010, SAE_2025_005, SAE_2025_006, SAE_2024_011, SAE_2025_012, RAT_2024_006, RAT_2025_013, SUP_2022_005, LMT_2025_006, LMT_2024_005, LMT_2025_007, QNT_2023_001, QNT_2025_002, QNT_2026_003, KD_2025_001 (App. A-D.4), plus new SUP_2022_008, SAE_2025_014, LMT_2025_010. **Still open:** RAT_2025_004 capability numbers, RAT_2023_007, KD_2025_001 G.4 and the fitted law, the rest of SUP_2022_001 (geometry, computation in superposition), Henighan 2023, SAE_2025_007/008/013, plus a PDF-level spot-check of the P-level numbers before any paper draft. Two papers could not be located (see reading_log.md "Needs user-provided PDF"). |
| Q-LIT-2 | Prior-art check for gap G1 with alternative terms ("dictionary learning regulariser pretraining", "sparse coding auxiliary loss transformer", "SAE-regularized pretraining", OpenReview search for ICLR 2026 / NeurIPS 2026 submissions). | Open |
| Q-LIT-3 | Examine the "not yet examined" list in paper_library.md (DeCov, Jermyn 2022, Chinchilla, Pythia, Muon, Cramming, KD foundations, PTQ methods). | Partly done 2026-11-14: Jermyn 2022, Scherlis 2022 and "Muon is Scalable" examined (level P). Rest still open. |
| Q-LIT-4 | Does SUP_2026_004's orthogonality regulariser act on model weights during training (direct overlap with G1) or on an SAE? | **Answered at level F (2026-11-14, PDF pp. 1-10 and 14-20 read):** it acts on the decoder of a TopK SAE (Eq. 1); the LM is then LoRA-fine-tuned around the frozen SAE on MetaMathQA (fine-tuning, not pretraining). No overlap with G1; the direction (less overlap) is opposite to this project's. |

**Pointer, Q-LIT-2 (2026-11-14):** first prior-art pass done; see [prior_art_search.md](prior_art_search.md) (search log, closeness table, novelty per Stage 1 candidate). No identical prior work found; the penalty forms for 1a already exist (Gao 2019, Bank and Giryes, MHE); 1b has no trained-toy precedent found. Q-LIT-2 stays open for the unexamined items listed there. Q-LIT-4 answered at level P: SUP_2026_004's penalty acts on the SAE decoder, not on LM weights during pretraining.

## Design (raised by the user's clarification, 2026-11-11)

| ID | Question | Status |
|---|---|---|
| Q-DES-1 | Which quantity measures "amount of superposition" so it counts features per *dimension* (rotation-invariant), not mixing per *neuron*? Candidates: number of live SAE features with low reconstruction error per dimension; SUP_2025_002's overlap/row-norm statistics; ground-truth count (toy only). | Open |
| Q-DES-2 | How does the penalty reach the model's weights (what differentiable function of the model's activations, given the SAE held fixed)? Note: more features active *per token* (higher L0) is denser activation, not more features *stored*; the penalty must target the second. | Open — core design problem for the toy stage |
| Q-DES-3 | Exactly what "SAE deltas" contributes to the measurement (user: reconstruction loss considered; delta patching for anything inserted). | Partly answered |
| Q-DES-4 | Joint SAE training safeguards (D-09): stop-gradient, reconstruction-quality gate, independent ruler. | Proposed |
| Q-DES-5 | How "performs similar" is defined for the A-vs-B quantised comparison (metric and margin). | Open |
| Q-LIT-5 | Literature on knowledge storage per parameter (e.g. Allen-Zhu & Li, "Physics of Language Models 3.3: knowledge capacity scaling laws") — not yet examined; needed for the fact-storage test. | Partly answered 2026-11-14 (level P): Allen-Zhu & Li (~2 bits/param at 1000 exposures, ≥ 1 at 100, int4 halves it) and Morris et al. (~3.6 bits/param on random tokens, small models, lower bound); neither gives a number for ordinary text, so the Stage 4 test needs its own synthetic-fact design. |

## Scientific (to be answered by experiments)

| ID | Question | Status |
|---|---|---|
| Q-SCI-1 | Can an online SAE be kept accurate enough during training at acceptable overhead (refresh schedule, dead latents)? | Open — Phase 4/5 pilot |
| Q-SCI-2 | Does an SAE-derived loss differ from a random/PCA-dictionary loss of the same form? (H3) | Open |
| Q-SCI-3 | Which regime shows any effect: compute-limited or data-limited? (H1/H2) | Open |
| Q-SCI-4 | Do representation metrics predict final loss across runs? (H5) | Open |
| Q-SCI-5 | Does the model "game" the SAE loss (better SAE metrics, no change in capability)? (H6) | Open |
