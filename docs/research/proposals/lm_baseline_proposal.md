# Proposal: the standard language model (model B) and the shared training pipeline

Date: 2026-11-15. Status: **every decision below is PROPOSED, NOT APPROVED.** A recommendation is not an approval (CLAUDE.md section 9). Nothing here has been run; no GPU was used (a sweep is using it). Sources, with verification levels, are in [lm_sources.md](lm_sources.md); source keys appear in brackets, for example [LMT_2022_202].

**Verification levels** (scheme of [../citation_index.md](../citation_index.md)): **F(text)** = the paper or page was downloaded and its text read directly by me, in the sections named in lm_sources.md (this is not a summariser, but figures and page images were not viewed, so I write F(text), not plain F); **P** = read through a summarising fetch tool; **A** = abstract/first page only; **S** = snippet. A claim at A or S is a pointer, not a ground for an approved decision.

**How numbers are labelled.** *Computed* = formula shown, done by me. *Cited* = from a source I opened. *Assumption* = my choice, not measured; each is listed in section 2.1 so you can overrule it.

---

## 1. Plain-language overview

**What this document is for.** Your final test compares model A (trained with your penalty) against model B (a normal model of larger size). For that test to mean anything, B must be a *properly built and properly tuned ordinary model*, trained on the same data, with the same tools. This document proposes what B is, how big, how it is trained, how it is measured and where it runs. It also tells you, with arithmetic, what the literal "2 GB model vs 3 GB model" would cost.

**The short answer on cost.** A model whose file is 2 GB has roughly 1 to 3.3 billion parameters (depending on how hard it is compressed). Training one from scratch at the 20-tokens-per-parameter rule takes about 270 to 2,850 hours on a rented data-centre A100 (about 580 to 6,200 US dollars for a single run, no tuning, no repeats; assumptions in section 3.1), and it cannot even fit in the memory of a free Colab/Kaggle GPU or your GTX 1660 Ti. So the literal test is **not feasible on free compute, and not sensible on a student budget** (section 3.5). The proposal is therefore to run the *same test at a smaller size and keep the ratio* (A is 1.5 times smaller than B): a ladder of models from about 10 million to about 125 million parameters. If your penalty helps, it should show up there first; if it does not show up there, it will not appear at 2 GB either.

**Words used below.**
- *Parameter*: one learned number. "125M" means 125 million of them.
- *Token*: a piece of text (a word or word fragment) the model reads. Training data is counted in tokens.
- *FLOPs*: number of arithmetic operations. A standard rule: training costs about **6 x parameters x tokens** (6ND). This is the same arithmetic your earlier notes used.
- *Chinchilla rule*: for the best result at a fixed compute, train on about **20 tokens per parameter** [LMT_2022_202]. "Over-trained" means many more tokens per parameter (for example 100): worse use of compute, but a better small model; many modern small models are trained this way [LMT_2025_206].
- *Perplexity / bits per byte (BPB)*: how surprised the model is by held-out text; lower is better. BPB counts per byte of text, so it does not depend on the tokenizer.
- *Tokenizer / vocabulary*: the table that cuts text into tokens. Every vocabulary entry costs one row in the embedding matrix; in a small model this can be a large part of all parameters (section 3.2).
- *Tied embeddings*: the input table and the output layer share one matrix, saving parameters.
- *RoPE, RMSNorm, SwiGLU*: three standard building blocks of "Llama-style" models (position encoding, normalisation, feed-forward activation).
- *MFU*: model FLOPs utilisation, the fraction of the GPU's peak speed a training run really reaches [LMT_2022_219]. Real runs reach 20 to 55 percent (computed from published runs, section 3.3).
- *Quantisation*: storing weights with fewer bits (8 or 4 instead of 16). *GGUF/llama.cpp*: the common file format and program for running quantised models on a laptop.
- *Seed*: the random starting point of a run. Two runs differing only in seed give slightly different losses, and that spread is what any "A beats B" claim must exceed.

**The ten-minute path.** Read the approval sheet (section 2), skim section 3 (the cost tables), and approve or change the 16 decisions in section 4. Sections 5 and 6 are the checklist and the honest limits.

---

## 2. Approval sheet

All PROPOSED, NOT APPROVED. "Cost of being wrong" is my judgement.

| ID | Question | Recommendation | Cost of being wrong |
|---|---|---|---|
| D-LM-1 | Size ladder and the 1.5x rule | Ratio-preserving ladder: rungs of about 10.6M / 24M / 47M (local or free) and 126M (rented, only after earlier gates), B = 1.5x A; literal 2 GB vs 3 GB dropped | High: sets all compute |
| D-LM-2 | Block design | Llama-style: pre-norm RMSNorm, SwiGLU, RoPE, no biases, plain multi-head attention, head size 64 | Medium |
| D-LM-3 | Tied embeddings | Tied | Low |
| D-LM-4 | Tokenizer | Custom byte-level BPE, 16,384 entries, trained on the main corpus | Medium |
| D-LM-5 | Corpus | TinyStories for development; FineWeb-Edu 10B-token sample for the main ladder; fixed held-out slice | High |
| D-LM-6 | Context length | 1,024 tokens, documents packed with an end-of-text token | Low |
| D-LM-7 | Token budget | 20 tokens/parameter as primary; 100 tokens/parameter as a second regime on the two smallest rungs | Medium |
| D-LM-8 | Optimiser, schedule, regularisation | AdamW (0.9, 0.95), weight decay tuned (not fixed), grad clip 1.0, cosine to 10 percent, learning rate tuned per size and arm | High (fair baseline) |
| D-LM-9 | Batch size | 65,536 tokens for 10-50M, 262,144 for 126M | Low |
| D-LM-10 | Precision and kernels | fp16 autocast on Turing GPUs, bf16 on rented Ampere/Hopper, PyTorch scaled-dot-product attention; one precision per comparison | Medium |
| D-LM-11 | Checkpoints | Resumable, rolling plus about 20 log-spaced milestones, never overwritten | Low |
| D-LM-12 | Evaluation | Primary: held-out bits per byte; secondary: zero-shot benchmarks with stated noise | High |
| D-LM-13 | Where to train | Local + free for the first three rungs; rented A100/H100 for rung 4 only, budget set by you | High (money) |
| D-LM-14 | A-vs-B protocol and penalty cost | Matched tokens (primary) and matched FLOPs (secondary), at least 3 seeds, resolution check first, equal tuning effort | High |
| D-LM-15 | Quantisation format and runtime | GGUF via llama.cpp (Q8_0, Q4_K_M) plus a controlled round-to-nearest quantiser; equal-bytes arm | Medium |
| D-LM-16 | Code base | Own small PyTorch loop in the style of nanoGPT; export to Hugging Face Llama format for GGUF and evaluation | Medium |

---

## 3. Compute and size

### 3.1 Assumptions (labelled; overrule any of them)

| ID | Assumption | Basis |
|---|---|---|
| A1 | Training FLOPs = 6 x N x D (N total parameters, D tokens). Attention adds an extra term, shown separately. | Kaplan Sec. 2.1 [LMT_2020_201]; Chinchilla Appendix F, Table A4: exact FLOP count divided by 6ND is 0.99 to 1.10 for 73M to 6.8B [LMT_2022_202] |
| A2 | GTX 1660 Ti, fp32: sustained 3 TFLOP/s | This is the earlier project assumption (feasibility_assessment.md section 2.5), about 55 percent of the 5.44 TFLOP/s peak (computed: 2 x 1,536 shaders x 1.77 GHz; shader count, clock from [HW_2026_204], a secondary source, desktop card; your laptop/desktop variant is unverified). If it is really 2 TFLOP/s, multiply all 1660 Ti times by 1.5. **Not measured.** |
| A3 | Colab/Kaggle T4, fp16 autocast: sustained 10 TFLOP/s | **Assumption**, about 15 percent of NVIDIA's 65 TFLOP/s fp16 peak [HW_2026_201]; the T4 is limited to 70 W. I found no measured transformer-training figure for a T4 (one search, none found). Plausible range 6 to 15. The project's own toy runs showed Colab T4 fp32 about 1.33x a 1660 Ti (feasibility notes), consistent with 4 TFLOP/s fp32. |
| A4 | Rented A100 80 GB, bf16: 40 percent MFU (125 TFLOP/s) for models of 100M or more, 20 percent (62 TFLOP/s) below 100M | Between published runs, computed in section 3.3 (Pythia 22 to 34 percent; TinyLlama 51; llm.c 55). I chose a value below the best-optimised runs because your loop will be less tuned. |
| A5 | Rented H100 SXM, bf16: 35 percent of 989.5 dense TFLOP/s (346 TFLOP/s) for 100M or more, 15 percent below 100M | 35 percent computed from the llm.c baseline (124M, 10B tokens, 45 minutes on 8 H100, as quoted by [LMT_2024_004]). Dense peak = half of NVIDIA's "with sparsity" figure of 1,979 [HW_2026_203] (my derivation). Below 100M: assumption. |
| A6 | Prices: A100 80 GB SXM 2.19 USD/h, H100 SXM 4.39 USD/h | RunPod pricing page, published 2026-11-13, on-demand list price [INF_2026_205]. Marketplace prices are lower and vary (Vast.ai median A100 SXM4 80 GB 0.84 USD/h, H100 SXM 2.47 USD/h, same date). Excludes storage, idle time, failed runs. |
| A7 | Real life costs more than the arithmetic | Evaluation, checkpointing, data loading, restarts: plan on about 1.3x (my judgement) |
| A8 | Training memory: 16 bytes per parameter for weights, gradients and Adam states, plus a reserve | ZeRO paper, Sec. 3.1: 2 + 2 + 12 = 16 bytes with mixed precision [LMT_2020_221]; plain fp32 AdamW is 4 + 4 + 8 = 16 too (computed). Reserve for activations and CUDA context: 1.5 GB (6 GB card) and 2 GB (16 GB card), assumption. |

### 3.2 Table 1a: the ladder (Llama-style, tied embeddings, 16,384-entry vocabulary)

Parameter formula (computed, no biases): per layer 4d^2 (attention) + 3 d d_ff (SwiGLU) + 2d (two norms); total N = L x (4d^2 + 3 d d_ff + 2d) + d (final norm) + V d (tied embedding). Heads have size 64 (heads = d/64). d_ff is about 8/3 of d rounded to a multiple of 64, which keeps SwiGLU at the parameter count of a 4x ordinary MLP (the matching trick of [LMT_2020_215]). Shapes are my illustrative picks, not tuned.

| Row | d_model | layers | heads | d_ff | body params (M) | embedding params (M) | **total N (M)** | embedding share at V = 16,384 | embedding share at V = 50,304 (GPT-2 vocabulary) |
|---|---|---|---|---|---|---|---|---|---|
| S10 | 256 | 8 | 4 | 704 | 6.4 | 4.2 | **10.6** | 39% | 67% |
| S25 | 384 | 10 | 6 | 1024 | 17.7 | 6.3 | **24.0** | 26% | 52% |
| S50 | 512 | 12 | 8 | 1408 | 38.5 | 8.4 | **46.9** | 18% | 40% |
| S125 | 768 | 16 | 12 | 2048 | 113.3 | 12.6 | **125.9** | 10% | 25% |
| S350 | 1024 | 26 | 16 | 2752 | 328.9 | 16.8 | **345.7** | 5% | 14% |
| S1B | 2048 | 18 | 32 | 5632 | 924.9 | 33.6 | **958.5** | 4% | 10% |
| S3B | 2560 | 38 | 40 | 6912 | 3013.5 | 41.9 | **3055.5** | 1% | 4% |

Check against published models (computed with the same formula): Pythia-70M has d = 512, 6 layers, 50,304 untied vocabulary entries, so 18.9M body + 2 x 25.8M embedding = 70.4M, of which 73 percent are embeddings [LMT_2023_204, Table 1]; TinyLlama is 1.10B with d = 2,048, 22 layers, d_ff = 5,632, 4 key-value groups and 32,000 vocabulary [LMT_2024_207]. MobileLLM states embeddings are more than 20 percent of a 125M model [LMT_2024_001].

### 3.3 Table 1b: tokens, FLOPs, memory, checkpoint size

Tokens at 20 per parameter (Chinchilla Table 3: 400M gets 8.0B tokens = 20.0 per parameter; 1B gets 20.2B [LMT_2022_202]) and at 100 per parameter ("over-trained", my choice). 6ND at both. "Attention extra" = L x 1,024 x d / N, from Kaplan's forward cost 2N + 2 L n_ctx d [LMT_2020_201]. Checkpoint file sizes use bits per weight measured by llama.cpp on Llama-3.1-8B: Q8_0 8.5008, Q4_K_M 4.8944, F16 16.0005 [QNT_2026_201]; small models with large embedding shares or other tensors may differ.

| Row | N (M) | tokens at 20N (B) | tokens at 100N (B) | 6ND at 20N | 6ND at 100N | attention extra, ctx 1,024 | training states 16N (GB) | fits 6 GB card | fits 16 GB T4 | int8 file (MB) | 4-bit file (MB) | fp16 file (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S10 | 10.6 | 0.21 | 1.06 | 1.35e16 | 6.77e16 | +20% | 0.17 | yes | yes | 11 | 6 | 21 |
| S25 | 24.0 | 0.48 | 2.40 | 6.91e16 | 3.45e17 | +16% | 0.38 | yes | yes | 25 | 15 | 48 |
| S50 | 46.9 | 0.94 | 4.69 | 2.64e17 | 1.32e18 | +13% | 0.75 | yes | yes | 50 | 29 | 94 |
| S125 | 125.9 | 2.52 | 12.59 | 1.90e18 | 9.50e18 | +10% | 2.01 | yes | yes | 134 | 77 | 252 |
| S350 | 345.7 | 6.91 | 34.57 | 1.43e19 | 7.17e19 | +8% | 5.53 | **no** | yes | 367 | 211 | 691 |
| S1B | 958.5 | 19.17 | 95.85 | 1.10e20 | 5.51e20 | +4% | 15.34 | **no** | **no** | 1,018 | 586 | 1,917 |
| S3B | 3055.5 | 61.11 | 305.55 | 1.12e21 | 5.60e21 | +3% | 48.89 | **no** | **no** | 3,247 | 1,869 | 6,111 |

Fit rule (A8): states + reserve must be below 6 GB (local) or 16 GB (T4), nominal sizes (decimal GB), so 6 GB allows about 280M parameters and 16 GB about 875M. Both are approximate until measured.

**Calibration of utilisation (computed from published runs, 6ND over measured GPU-hours; A100 peak 312 TFLOP/s bf16, [HW_2026_202]):**

| Run | Computed throughput | Fraction of peak |
|---|---|---|
| Pythia-70M, 300B tokens, 510 A100-hours (Table 5) [LMT_2023_204] | 69 TFLOP/s | 22% |
| Pythia-160M / 410M (1,030 / 2,540 hours) | 79 / 80 TFLOP/s | 25 / 26% |
| Pythia-1.0B / 1.4B (4,830 / 7,120 hours) | 105 / 99 TFLOP/s | 34 / 32% |
| TinyLlama 1.1B, 24,000 tokens/s per A100-40G [LMT_2024_207] | 158 TFLOP/s | 51% |
| llm.c GPT-2 124M, 10B tokens, about 90 min on 8 x A100 80 GB [LMT_2024_224, P] | 173 TFLOP/s | 55% (the author states "up to about 60%") |
| llm.c baseline on 8 x H100, 45 min (quoted in [LMT_2024_004]) | 346 TFLOP/s | 35% of 989.5 dense |

So the small models of Pythia use the GPU poorly (22 to 26 percent), well-tuned code reaches 50 percent or more. A4 (20 percent below 100M, 40 percent above) is a middle choice.

### 3.4 Table 2: training time and rented cost (computed from Table 1b and A2 to A6; multiply by A7 for planning)

**Primary regime, 20 tokens per parameter.**

| Row | GTX 1660 Ti, fp32, 3 TFLOP/s (hours) | T4, fp16, 10 TFLOP/s (hours) | A100 80 GB (hours / USD at 2.19) | H100 SXM (hours / USD at 4.39) |
|---|---|---|---|---|
| S10 | 1.3 | 0.4 | 0.1 / 0 | 0.0 / 0 |
| S25 | 6.4 | 1.9 | 0.3 / 1 | 0.1 / 1 |
| S50 | 24.5 | 7.3 | 1.2 / 3 | 0.5 / 2 |
| S125 | 176 | 52.8 | 4.2 / 9 | 1.5 / 7 |
| S350 | 1,328 | 398 | 31.9 / 70 | 11.5 / 50 |
| S1B | 10,207 | 3,062 | 245 / 537 | 88.4 / 388 |
| S3B | 103,733 | 31,120 | 2,494 / 5,461 | 899 / 3,945 |

**Over-trained regime, 100 tokens per parameter.**

| Row | GTX 1660 Ti (hours) | T4 (hours) | A100 80 GB (hours / USD) | H100 SXM (hours / USD) |
|---|---|---|---|---|
| S10 | 6.3 | 1.9 | 0.3 / 1 | 0.1 / 1 |
| S25 | 32.0 | 9.6 | 1.5 / 3 | 0.6 / 3 |
| S50 | 122 | 36.7 | 5.9 / 13 | 2.5 / 11 |
| S125 | 880 | 264 | 21.2 / 46 | 7.6 / 33 |
| S350 | 6,639 | 1,992 | 160 / 350 | 57.5 / 252 |
| S1B | 51,037 | 15,311 | 1,227 / 2,687 | 442 / 1,941 |
| S3B | 518,666 | 155,600 | 12,468 / 27,305 | 4,493 / 19,724 |

These are **single runs**. The comparison needs about 40 runs per rung (3.6). Consistency with the earlier feasibility numbers: the old table gave 6 to 10 h for 20-30M at 0.5B tokens (mine: 6.4 h), about 170 h for 124M at 2.5B tokens (mine: 176 h) and 1.5 to 4 h on an A100/H100 (mine: 1.5 to 4.2 h). I am not contradicting section 2.5 of the feasibility assessment; I added the over-trained regime, the memory-fit column and prices.

### 3.5 What "2 GB" and "3 GB" mean, and what they would cost

Your notes (U-04) say 2 GB goes to about 1 GB when quantised; the task statement says A is about 2 GB after quantisation. Both readings are shown. Parameters N = bytes x 8 / bits per weight (computed; GB = 1e9 bytes). Cost = 6 x N x 20N FLOPs on an A100 at 125 TFLOP/s and USD 2.19/h (A4, A6), one run, no tuning, no seeds.

| File size | stored as | bits/weight | parameters | tokens at 20/param | A100 hours | A100 USD | T4 days at 10 TFLOP/s | training states 16N |
|---|---|---|---|---|---|---|---|---|
| 2 GB | 16-bit | 16.0 | 1.00B | 20B | 267 | 584 | 139 | 16 GB |
| 2 GB | int8 (Q8_0) | 8.5 | 1.88B | 38B | 945 | 2,069 | 492 | 30 GB |
| 2 GB | 4-bit (Q4_K_M) | 4.89 | 3.27B | 65B | 2,850 | 6,241 | 1,484 | 52 GB |
| 3 GB | 16-bit | 16.0 | 1.50B | 30B | 600 | 1,314 | 312 | 24 GB |
| 3 GB | int8 | 8.5 | 2.82B | 56B | 2,126 | 4,655 | 1,107 | 45 GB |
| 3 GB | 4-bit | 4.89 | 4.90B | 98B | 6,412 | 14,042 | 3,340 | 78 GB |

**Plain statement.** On free compute the literal test is impossible: even the smallest reading (1.0B and 1.5B, 16-bit) needs 16 to 24 GB just for training states, which exceeds both the 6 GB card and a 16 GB T4, and would take 139 and 312 T4-days per run. On rented compute, the cheapest reading is about USD 584 plus 1,314 for one run each of A and B with no tuning and no seeds; a defensible version (about 40 runs, as in 3.6) would be roughly 40 times that for the smaller of the two and is out of reach. The 4-bit reading (3.3B and 4.9B) is about USD 20,000 for two single runs. The literal test is therefore replaced by the ladder below, which keeps the 1.5 ratio. For scale: the 126M rung is 134 MB / 200 MB quantised to int8, not 2 GB / 3 GB.

### 3.6 The ladder, and what a full test costs at each rung

Arms per rung (see D-LM-14): B (standard, 1.5P), C (standard, P; the real "no method" control for A), A (penalty, P), D (equal-cost non-SAE penalty, P), E (random-dictionary penalty, P), plus standard models at 0.75P and 2P to read the loss-versus-size curve, each with 3 seeds, plus tuning. Cost in units of one run at size P: 0.75^2 + 4 x 1 + 1.5^2 + 2^2 = 10.8, times 3 seeds = 32.4, plus about 8 for learning-rate and weight-decay tuning (shorter runs, my estimate) is about 40 (FLOPs scale with N^2 at a fixed tokens-per-parameter ratio). This reproduces the "about 40 runs" of the feasibility assessment. Computed totals (arms with a penalty also cost extra, section 4, D-LM-14):

| Rung P | Total FLOPs, 40 units, 20 tok/param | 1660 Ti (h / days) | T4 (h) | A100 (h / USD) | 100 tok/param: 1660 Ti (days) / T4 (h) / A100 (h / USD) |
|---|---|---|---|---|---|
| R1: 10.6M (B about 16M) | 5.4e17 | 50 / 2.1 | 15 | 2 / 5 | 10 / 75 / 12 / 26 |
| R2: 24.0M (B about 36M) | 2.8e18 | 256 / 10.7 | 77 | 12 / 27 | 53 / 384 / 62 / 136 |
| R3: 46.9M (B about 70M) | 1.1e19 | 978 / 41 | 293 | 47 / 104 | 204 / 1,466 / 237 / 518 |
| R4: 125.9M (B about 190M) | 7.6e19 | 7,045 / 294 | 2,113 | 169 / 370 | 1,468 / 10,567 / 845 / 1,851 |
| R5: 345.7M (B about 520M) | 5.7e20 | 53,115 / 2,213 | 15,934 | 1,275 / 2,792 | not planned |

Reading: R1 and R2 are comfortable on free or local compute (R2 about 3 days of T4 at 20 tok/param; if Kaggle gives about 30 GPU-hours per week, which I could **not** verify from Kaggle's own documentation, section 3.7, that is under 3 weeks). R3 on free compute is about 10 weeks of weekly quota (unverified) or about USD 100 rented. R4 is rented-only (about USD 370, or about USD 590 if the A100 only reaches 25 percent MFU: 40/25 = 1.6x). R5 is beyond a student budget.

Example shapes for B at 1.5x (computed, same formula; d must be a multiple of 64): R1: d = 320, 8 layers, d_ff = 896 gives 15.4M (1.45x); R2: d = 448, 12 layers, d_ff = 1,216 gives 36.6M (1.52x). Exact B shapes are part of D-LM-1.

**The cheapest, most informative first step does not involve the penalty at all:** train only standard models at 10.6, 16, 24 and 36M with 3 seeds at 20 tokens per parameter. Cost: sum of 120 N^2 over the four sizes, times 3 = 8.1e17 FLOPs, about 75 hours on the 1660 Ti or 22 hours on a T4 (computed). This answers the question feasibility_assessment.md section 3 called the resolution check: is the loss gap between P and 1.5P large compared with the seed-to-seed spread?

**How large is that gap likely to be?** Using Epoch AI's refit of the Chinchilla loss curve, L(N, D) = 1.8172 + 482.01 / N^0.3478 + 2085.43 / D^0.3658 [LMT_2024_203], at 20 tokens per parameter (computed): R1 5.430 vs 5.202 (gap 0.23 nats); R2 4.516 vs 4.344 (0.17); R3 3.942 vs 3.806 (0.14); R4 3.311 vs 3.214 (0.10); R5 2.859 vs 2.791 (0.07). **Caution: these sizes are all below the 70M smallest model of the original study and the data/tokenizer differ, so this is an order-of-magnitude illustration only.** It says that "matching a model 1.5x larger" needs the penalty to lower the loss by about 0.1 to 0.2 nats, 2 to 4 percent. The original fit constants are disputed (the Epoch replication found the reported ones inconsistent), so I used the refit.

### 3.7 Free compute: what is documented

| Resource | Documented | Not documented / unverified |
|---|---|---|
| Kaggle notebooks | 12 hours per CPU/GPU session; T4 x 2 option = 2 Tesla T4, 4 CPU cores, 29 GB RAM; 20 GB working disk; 20 minutes idle limit while editing [INF_2026_203] | **Weekly GPU quota: the documentation page gives no number**; it only says quotas appear in your account settings. The widely repeated "about 30 hours per week" comes from secondary sources and is unverified. Check your own account page. Queue times vary (the page says free GPUs may queue at busy times) |
| Colab free | Runtime "at most 12 hours", limits "dynamic" and unpublished; GPU types vary [INF_2026_204] | Which GPU you get; weekly limits; paid prices (not on the FAQ page) |
| Local | GTX 1660 Ti 6 GB, no tensor cores, no bf16 (project notes); FlashAttention-2 kernels need Ampere or newer [INF_2026_202] | Exact card variant, sustained speed (A2), whether `torch.compile` works on this Windows setup (the nanoGPT README of 2022 said compile was not available on Windows [LMT_2024_223]; not re-tested) |

---

## 4. Decisions (CLAUDE.md section 9.1 format; every one PROPOSED, NOT APPROVED)

### D-LM-1: model-size ladder and the 1.5x rule
1. **Decision:** which sizes to train, and how to scale down the "2 GB vs 3 GB" test.
2. **Context:** section 3.5 shows the literal test is out of reach. The comparison needs A at P and B at 1.5P, repeated with seeds, plus a size ladder for the "equivalent size" number rho of feasibility_assessment.md section 3.
3. **Research evidence:** Chinchilla Table 3 and App. F [LMT_2022_202]; fitted scaling laws hold from 0.011B models [LMT_2024_210]; Table 2 and 3.6 above (computed).
4. **Available approaches:** (a) literal 2 GB vs 3 GB; (b) ladder R1 to R4 as in 3.6, with R4 conditional; (c) R1 to R2 only; (d) start at R4/R5 rented.
5. **Advantages:** (b) keeps the ratio and the whole experiment affordable; R1 and R2 are free. (c) cheapest. (a) is the original wording. (d) closer to "real" models.
6. **Disadvantages:** (b) results at 10 to 126M may not transfer to 1B+. (c) two sizes give a weak curve. (a) infeasible (3.5). (d) USD 2,800+ per rung before any penalty.
7. **Recommendation:** (b). R1, R2 first (free), R3 optional, R4 only if earlier gates pass and you set a budget. Include standard models at 0.75P and 2P at each rung. Fix B's exact shape within 3 percent of 1.5x total parameters (examples in 3.6).
8. **Consequences:** every later phase runs at these sizes; the final claim is "at 10 to 126M parameters", stated as such.
9. **Approval:** requested.

### D-LM-2: block design
1. **Decision:** the transformer block recipe: normalisation, activation, position encoding, biases, attention type.
2. **Context:** U-07 says reuse standard components. The choice also decides how easily the model converts to GGUF and how "neuron-like" the MLP is for later SAE work.
3. **Research evidence:** SwiGLU/GEGLU beat ReLU at equal parameters (heldout log-perplexity 1.944 and 1.942 against 1.997, Table 1 of [LMT_2020_215]); RMSNorm matches LayerNorm and is 7 to 64 percent faster [LMT_2019_216, A]; RoPE and SwiGLU are used by TinyLlama [LMT_2024_207] (which also uses RMSNorm), SmolLM2 (stated to follow the Llama-2 architecture; its Table 6 lists SwiGLU, RoPE, tied embeddings) [LMT_2025_206], OLMo 2 (RMSNorm, RoPE, SwiGLU) [LMT_2025_101] and PaLM (SwiGLU, RoPE, no biases, but LayerNorm) [LMT_2022_219]; OLMo also drops all biases [LMT_2024_205]; Pythia used GELU-style MLPs, rotary, parallel attention/MLP [LMT_2023_204]; RoPE itself was evaluated on classification tasks in its own paper [LMT_2021_217, A]; llama.cpp ships converters for both Llama and GPT-2 architectures [QNT_2026_202]; MobileLLM found deep-and-thin and grouped-query attention helped at 125M [LMT_2024_001].
4. **Available approaches:** (a) nanoGPT/GPT-2 style: LayerNorm, GELU, learned absolute positions, biases; (b) Llama-style as proposed; (c) speedrun style (ReLU-squared, Muon, value embeddings, skip connections, softcap) [LMT_2024_004].
5. **Advantages:** (b) the most widely used small-model recipe, converts cleanly to GGUF, the most evidence. (a) simplest, Karpathy-reproducible baselines. (c) far more token-efficient (README: the 124M-class target in under 330M tokens versus 10B for the llm.c baseline).
6. **Disadvantages:** (b) three-matrix MLP differs from what most interpretability tooling expects (SAEs on the residual stream are unaffected). (a) older. (c) many interacting tricks that change the representation geometry, so a packing penalty would be confounded; harder to export.
7. **Recommendation:** (b) with plain multi-head attention (head size 64), RoPE with theta = 10,000 (the value in [LMT_2025_206] and [LMT_2024_207]; OLMo 2 uses 500,000 for long context, not needed at 1,024), no biases, pre-norm RMSNorm. Grouped-query attention and layer-count changes are postponed. State in the write-up that a speedrun-style baseline would be a much stronger "standard model"; B is "standard", not "best possible".
8. **Consequences:** fixes the parameter formula of 3.2; A and B share it, so the penalty is the only difference.
9. **Approval:** requested.

### D-LM-3: tied embeddings
1. **Decision:** share the input embedding and the output matrix or not.
2. **Context:** at 10 to 50M parameters the embedding is 18 to 39 percent of the model (Table 1a); it also counts towards file size.
3. **Research evidence:** MobileLLM Table 1: sharing saves 16M parameters (135M to 119M) for an average accuracy change from 44.8 to 44.6, and re-spending those parameters on two more layers gives 45.0 [LMT_2024_001]; SmolLM2 uses tied embeddings [LMT_2025_206]; Press and Wolf report lower perplexity from tying in LSTM language models [LMT_2017_218, A]; against: Pythia is untied for interpretability reasons [LMT_2023_204] and the speedrun record "untied embedding and head" cut 12.0 to 10.8 minutes at the 50k GPT-2 vocabulary [LMT_2024_004].
4. **Available approaches:** tied; untied.
5. **Advantages:** tied: fewer parameters, honest file size, usual for sub-billion models. Untied: slightly better per step in some speedrun settings.
6. **Disadvantages:** tied: input and output geometry cannot differ; any later penalty on embedding rows acts on both. Untied: 10.6M model would spend a further 4.2M parameters on a second table.
7. **Recommendation:** tied.
8. **Consequences:** parameter formula in 3.2 assumes it; changing later changes all sizes.
9. **Approval:** requested.

### D-LM-4: tokenizer and vocabulary
1. **Decision:** which tokenizer, and the vocabulary size.
2. **Context:** the embedding share is large in small models (Table 1a), A and B must use the same tokenizer, and FineWeb-Edu is published pre-counted in GPT-2 tokens [DAT_2024_202].
3. **Research evidence:** the compute-optimal vocabulary grows with compute (non-vocabulary parameters N_nv = 0.08 C^0.50, vocabulary parameters N_v = 0.20 C^0.42, fitted on 33M to 3B models, vocabulary 4K to 96K) [LMT_2024_212]. Applying that fit myself (computed; counting N_v as V x d, an assumption about how the paper counts; extrapolating down to 17.7M): about 5k entries at 17.7M body, 7.7k at 38.5M, 12.7k at 113M, 23k at 329M. OLMo 2 found a 100k vocabulary slightly better downstream at 1B/100B tokens but cites the same work saying larger vocabularies are at a slight disadvantage at small compute [LMT_2025_101]. TinyStories kept only the 10,000 most common GPT-Neo tokens [LMT_2023_002]. SmolLM2 uses 49,152 and OLMo 2 uses a cl100k-based 100k vocabulary.
4. **Available approaches:** (a) GPT-2 BPE (50,257, padded to 50,304); (b) custom byte-level BPE, 8,192 or 16,384 entries, trained on the main corpus; (c) top-10k truncation of GPT-2 as in TinyStories; (d) the 32k Llama tokenizer (licence terms not checked by me; not recommended).
5. **Advantages:** (a) zero work, pre-tokenised shards exist, nanoGPT/llm.c compatible. (b) embedding is 18 to 39 percent instead of 40 to 67 percent at 10 to 50M; the body of the network (where the project's packing idea lives) gets the parameters. (c) matches the TinyStories paper.
6. **Disadvantages:** (a) at 25M parameters more than half of all parameters are a lookup table; at 50k entries many rows are rarely trained. (b) a tokenizer to train and the corpus to tokenise ourselves (hours of CPU, not measured); token counts differ from the "10B GPT-2 token" label of the sample; slightly more tokens per byte than GPT-2 (not measured). (c) too narrow for web text.
7. **Recommendation:** (b), 16,384 entries, byte-level BPE trained on a few GB of the main corpus, one tokenizer for all rungs. The fit above leans to 8k for the smallest rungs; 16k is a compromise (round size, fewer tokens per byte). Report **bits per byte** everywhere so the choice cannot bias comparisons. Treat 8,192 as the alternative if you prefer to follow the fit.
8. **Consequences:** parameter counts in section 3 assume V = 16,384; changing V changes N, tokens and cost; GGUF conversion needs the tokenizer in Hugging Face format; embedding bytes enter the quantised file size.
9. **Approval:** requested.

### D-LM-5: training corpus
1. **Decision:** which text the models train on, and the held-out split.
2. **Context:** the existing D-02 proposed developing on TinyStories and choosing between BabyLM and web text later. Data must exist, be big enough for the largest budget and have a usable licence.
3. **Research evidence (verified this session, sizes and licences from dataset cards [DAT_*]):**

| Candidate | Licence | Size | Notes |
|---|---|---|---|
| TinyStories | CDLA-Sharing-1.0 | 2,119,719 train and 21,990 validation stories; train file 1.92 GB (V2 GPT-4 file 2.23 GB) [DAT_2023_201]. Tokens: **estimate** 0.43 to 0.48B (1.92 GB at 4 to 4.5 bytes per token; not counted) | Synthetic, narrow; coherent text from models under 10M parameters [LMT_2023_002]; single V100 under 30 h per model |
| FineWeb-Edu | ODC-By 1.0, plus Common Crawl terms of use | 1.3T tokens (score 3 or more); sample-10BT, 100BT, 350BT of about 10B / 100B / 350B GPT-2 tokens [DAT_2024_202] | Used by SmolLM2 (60 percent of the web mix) [LMT_2025_206]; classifier threshold 3 chosen by the authors |
| FineWeb | ODC-By 1.0 | more than 18.5T tokens; sample-10BT is 27.6 GB as stored [DAT_2024_203] | Less filtered than FineWeb-Edu |
| SlimPajama | "Apache 2.0" per the Cerebras post; 627B tokens [DAT_2023_206, P] | very large | **Not verified**: the dataset card returned HTTP 401 to me; provenance mixed; no advantage at our scale |
| Wikipedia (20231101.en) | CC-BY-SA-3.0 and GFDL | 20.2 GB of text (about 4.7B tokens at 4.3 bytes per token, estimate) [DAT_2023_204] | Small; useful only as held-out text |
| WikiText-103 | CC BY-SA | over 100M tokens [DAT_2016_205] | Classic held-out set |

   Also: repeating data up to 4 epochs is reported to change loss negligibly compared with unique data [LMT_2023_211, A: abstract only]. FineWeb's own ablations used 2 random subsets and 2 seeds per dataset version [LMT_2024_208].
4. **Available approaches:** (i) TinyStories only; (ii) TinyStories for development, FineWeb-Edu 10BT for the main ladder; (iii) FineWeb-Edu only; (iv) BabyLM-style 10M/100M-word corpus (no source opened this session).
5. **Advantages:** (ii) fast iteration plus a realistic corpus; FineWeb-Edu's 10B tokens cover R1 to R4 (R4 at 20 tok/param needs 2.5B; at 100 it needs 12.6B, which is 1.26 passes). (i) cheapest. (iv) established data-limited benchmark.
6. **Disadvantages:** (ii) two data pipelines. (i) the 0.45B-token file supports only 20 tok/param for 24M and above (at 100 tok/param, 10.6M needs 1.06B, about 2.3 epochs, inside the 4-epoch figure); loss gaps between sizes on a synthetic, easy corpus may not transfer. (iii) slower first iterations.
7. **Recommendation:** (ii). A fixed held-out slice of at least 5M tokens, cut before tokenisation and never trained on; optional second held-out set WikiText-103 test. No SlimPajama.
8. **Consequences:** fixes the data pipeline, token budgets (D-LM-7) and the "unique tokens" check per rung.
9. **Approval:** requested (this refines D-02).

### D-LM-6: context length and packing
1. **Decision:** sequence length and how documents are packed.
2. **Context:** attention compute grows with context; Table 1b column "attention extra" is 20 percent at 10.6M and 1,024 tokens.
3. **Research evidence:** TinyStories used 512 [LMT_2023_002]; GPT-2/nanoGPT/llm.c 1,024 [LMT_2024_223]; Chinchilla/Pythia/TinyLlama/SmolLM2 2,048 [LMT_2022_202, LMT_2023_204, LMT_2024_207, LMT_2025_206]; attention overhead formula [LMT_2020_201].
4. **Available approaches:** 512; 1,024; 2,048.
5. **Advantages:** 1,024 is the nanoGPT standard and roughly halves the attention overhead of 2,048 (computed: the extra term is proportional to n_ctx).
6. **Disadvantages:** 512 saves about 10 to 20 percent FLOPs on small models but cuts long-range structure; 2,048 costs more.
7. **Recommendation:** 1,024; documents concatenated with an end-of-text token, no cross-document attention masking at first.
8. **Consequences:** fixes the "attention extra" in Table 1b; RoPE allows later extension.
9. **Approval:** requested.

### D-LM-7: token budget per parameter
1. **Decision:** how many tokens each model sees.
2. **Context:** A-vs-B must use matched data; the regime also changes quantisation damage.
3. **Research evidence:** Chinchilla Table 3 gives about 20 tokens per parameter [LMT_2022_202]; small models are routinely over-trained: SmolLM2-135M saw 2T tokens (14,815 per parameter, computed), SmolLM2-360M 4T (11,111), TinyLlama 3T in v1.0 (2,727; v1.1 used 2T = 1,818) [LMT_2025_206, LMT_2024_207]; extra tokens keep helping up to 10,000 per parameter [LMT_2024_209, A]; a testbed spans 20 to 640 [LMT_2024_210]. More training tokens make post-training quantisation worse, especially for small models [QNT_2025_104, A; QNT_2025_103, A].
4. **Available approaches:** (a) 20 only; (b) 20 as primary and 100 on R1 and R2; (c) 100 or more everywhere.
5. **Advantages:** (b) the cheapest regime where the comparison is classic, plus a second regime that checks whether the result survives the more realistic (and more quantisation-hostile) over-trained setting.
6. **Disadvantages:** (b) doubles some compute (rows in the second table of 3.4); (a) may flatter methods that only help early; (c) 5x cost.
7. **Recommendation:** (b). Equal tokens for A, B, C, D, E within a rung.
8. **Consequences:** doubles or more the R1 and R2 budget; the quantisation test (D-LM-15) should be run at both.
9. **Approval:** requested.

### D-LM-8: optimiser, schedule, regularisation
1. **Decision:** the optimiser and its settings, applied identically to A and B, and how tuning is done.
2. **Context:** the baseline's strength decides whether a result means anything. Weight decay is also, in the toy model of this project, the competitor lever for packing (SUP_2025_002), so it must be tuned, not fixed.
3. **Research evidence:** decoupled weight decay is the AdamW paper [LMT_2019_220, A]. AdamW with betas 0.9 and 0.95 is used by Pythia (Adam), TinyLlama (cosine 4e-4 to 4e-5, 2,000 warmup steps, weight decay 0.1, clip 1.0), OLMo (0.9, 0.95), SmolLM2 (WSD schedule, peak 5e-4 for 1.7B and 3e-3 for 135M/360M), llm.c (6e-4, warmup 700, weight decay 0.1 at 124M) [LMT_2023_204, LMT_2024_207, LMT_2024_205, LMT_2025_206, LMT_2024_224]. OLMo 2: epsilon lowered from 1e-5 to 1e-8; **no weight decay on embeddings** (decay shrank embedding norms and added gradient spikes: spike score 0.16 against 0.092) [LMT_2025_101]. Chinchilla: set the cosine length equal to the token budget; cycle overshoot hurts [LMT_2022_202, App. B]. Learning-rate sensitivity is reduced by QK-layernorm and z-loss; instabilities are reproducible at small scale [LMT_2023_214]; scaled-model testbeds used 3e-3 with QK-layernorm [LMT_2024_210]. Muon is reported to need about 52 percent of AdamW's FLOPs (existing key LMT_2025_010).
4. **Available approaches:** (a) AdamW, cosine to 10 percent; (b) AdamW with warmup-stable-decay (WSD) so one stable run can be branched into several budgets [LMT_2025_206 states this motivation, citing other work I did not open]; (c) Muon for matrices (side direction A2).
5. **Advantages:** (a) best-understood, matches Chinchilla evidence; (b) saves compute across budgets; (c) potentially 2x cheaper.
6. **Disadvantages:** (a) one run per budget; (b) less standard, claim of equivalence not verified by me; (c) changes the geometry the project studies, a confound that belongs in its own experiment.
7. **Recommendation:** (a). AdamW, betas (0.9, 0.95), epsilon 1e-8, gradient clip 1.0, linear warmup of max(100 steps, 2 percent of steps) (my choice, no cited basis), cosine decay to 10 percent of peak, cycle length equal to the step budget. Weight decay applied to weight matrices only, none on embeddings, norms. **Tuning rule:** for each (size, arm) tune peak learning rate over at least three values spanning 6e-4 to 3e-3 and weight decay over {0, 0.1, 0.3}, with the same number of tuning runs for B/C as for A (including A's penalty-strength sweep). QK-norm and z-loss off by default; switched on for **all** arms in a rung if the learning-rate sweep shows divergence.
8. **Consequences:** fair-tuning effort is a fixed cost in 3.6; the weight-decay grid doubles as the "is it just weight decay?" control; Muon stays in the side direction A2.
9. **Approval:** requested.

### D-LM-9: batch size in tokens
1. **Decision:** tokens per optimiser step.
2. **Context:** too large a batch gives too few steps on a small token budget (R1 at 20 tok/param is 0.21B tokens: at 524,288 tokens per step that is 404 steps, computed).
3. **Research evidence:** the critical batch size B_crit = B*/L^(1/alpha_B), B* about 2e8 tokens, alpha_B about 0.21 (Kaplan Eq. 1.4) [LMT_2020_201], which I evaluated (computed): about 19k tokens at loss 7, 94k at 5, 272k at 4, 513k at 3.5, 1.07M at 3.0, so small models early in training want small batches; the noise-scale idea is from [LMT_2018_213, A]. Pythia nevertheless used 2,097,152 tokens for every model and reports no convergence problem 4x to 8x above the usual for sub-billion models [LMT_2023_204]; Chinchilla D.4 used 0.5M; llm.c 0.5M for 124M.
4. **Available approaches:** (a) fixed 0.5M for all; (b) scale with size: 65,536 up to 50M, 262,144 at 126M; (c) Pythia-style 2M.
5. **Advantages:** (b) enough steps for tiny budgets: 3,235 steps (R1), 7,324 (R2), 14,313 (R3), 9,605 (R4 at 262,144), computed at 20 tok/param.
6. **Disadvantages:** batch differs between rungs (comparisons are within a rung, so acceptable); more steps means more per-step overhead on small GPUs.
7. **Recommendation:** (b), with gradient accumulation to reach the target batch.
8. **Consequences:** learning-rate optimum shifts with batch; tuned per rung anyway (D-LM-8).
9. **Approval:** requested.

### D-LM-10: precision and kernels
1. **Decision:** numeric formats and attention kernel.
2. **Context:** the 1660 Ti and T4 are Turing GPUs (FlashAttention's README says its bf16 kernels need Ampere, Ada or Hopper GPUs [INF_2026_202]; the project notes already record no bf16 on the 1660 Ti); A100/H100 support bf16. The project's toy runs stayed in fp32 for comparability and found fp16 only about 12 percent faster there (project note), but an LM has large matrix multiplies where fp16 on a T4's tensor cores matters.
3. **Research evidence:** NVIDIA T4: 8.1 TFLOP/s fp32 against 65 fp16 [HW_2026_201]; 1660 Ti has no tensor cores, 5.4 fp32 and 10.9 fp16 peak [HW_2026_204]; A100 bf16 312 dense [HW_2026_202]; FlashAttention-2 supports Ampere, Ada, Hopper; bf16 needs those; Turing needs a separate project [INF_2026_202]; Pythia, TinyLlama, llm.c train mostly in bf16/fp16 mixed precision [LMT_2023_204, LMT_2024_224].
4. **Available approaches:** (a) fp32 everywhere; (b) fp16 autocast with loss scaling on Turing GPUs, bf16 autocast on Ampere/Hopper, fp32 master weights; (c) fp8 (speedrun) [LMT_2024_004].
5. **Advantages:** (b) up to several-fold faster on a T4 (not measured), memory halved for activations; (a) simplest numerics.
6. **Disadvantages:** (b) fp16 can overflow; different numerics on different hardware (previous toy runs already mixed GPUs and recorded each GPU); (c) too exotic.
7. **Recommendation:** (b). One precision and one hardware class per comparison: all arms of a rung run on the same class (fp16 on T4/1660 Ti, or bf16 on A100) and every run records GPU, driver, torch version. PyTorch scaled-dot-product attention. `torch.compile` only if it works on your setup. First action once the GPU is free: a 10-minute throughput benchmark on each device to replace A2 and A3.
8. **Consequences:** decides whether A2/A3 are replaced by measurements; any later move of a rung to another hardware class is a new experiment, not a continuation.
9. **Approval:** requested.

### D-LM-11: checkpoints and reproducibility
1. **Decision:** what is saved, how often, where.
2. **Context:** Kaggle sessions last 12 hours and Colab at most 12 [INF_2026_203, INF_2026_204]; later SAE or feature-dynamics work needs intermediate models; CLAUDE.md section 14.
3. **Research evidence:** Pythia saved checkpoints at initialisation, 10 log-spaced early steps and every 1,000 steps, 154 per model [LMT_2023_204]; full resumable state per size 12N bytes (fp32 weights plus two Adam moments; computed): 127 MB (10.6M), 288 MB (24M), 563 MB (47M), 1.51 GB (126M), 4.1 GB (346M); Kaggle working disk 20 GB [INF_2026_203].
4. **Available approaches:** (a) final only; (b) resumable rolling plus about 20 log-spaced weight-only milestones; (c) every 1,000 steps.
5. **Advantages:** (b) survives session limits and supports later analysis at small disk cost (milestones in fp16 are 2N bytes).
6. **Disadvantages:** (c) too much disk; (a) loses everything on an interruption.
7. **Recommendation:** (b). Every run directory holds: config file, git commit, seed, data shard order seed, hardware record, loss log, rolling resumable state (last two), milestone weights, final full state, evaluation JSON. Never overwritten; a re-run gets a new identifier.
8. **Consequences:** about 20 x 2N + 3 x 12N bytes per run; storage is a planning item for rung R4.
9. **Approval:** requested.

### D-LM-12: evaluation
1. **Decision:** which measurements define "performs similarly".
2. **Context:** D-06 proposed held-out loss as primary. At small scale most benchmarks are close to chance.
3. **Research evidence:** perplexity or bits per byte have better signal-to-noise than accuracy for small-scale decisions; benchmark signal-to-noise predicts whether small-scale decisions hold (R = 0.791); HellaSwag is "low noise, low signal", ARC-Challenge "high noise, high signal", MMLU "low noise, high signal" in the paper's example [EVL_2025_201, partial text]. FineWeb chose tasks with low variance, monotonic improvement and above-chance scores for 1.7B models [LMT_2024_208]. llm.c's 124M model on 10B tokens scores 29.9 on HellaSwag against 29.4 for GPT-2 124M (chance is 25) [LMT_2024_224, P]. Test-set sizes (HF dataset servers, [DAT_2025_207]) and the worst-case standard errors I computed (p = 0.5; a difference of two independent models has a 95 percent interval of 1.96 x sqrt(2) x SE):

| Benchmark | Items | SE of one accuracy | 95% interval for a difference of two models |
|---|---|---|---|
| HellaSwag (validation) | 10,042 | 0.50 points | plus or minus 1.4 points |
| LAMBADA (test) | 5,153 | 0.70 | 1.9 |
| ARC-Easy (test) | 2,376 | 1.03 | 2.8 |
| WinoGrande (validation) | 1,267 | 1.40 | 3.9 |
| ARC-Challenge (test) | 1,172 | 1.46 | 4.0 |
| MMLU (test) | 14,042 | 0.42 | 1.2 (but near chance for models under about 1B; not planned) |

   These are sampling errors of the test set only; **seed-to-seed variation is separate and unmeasured** (3.6).
4. **Available approaches:** (a) held-out loss only; (b) held-out bits per byte primary, zero-shot benchmarks secondary; (c) GPT-4-graded stories (TinyStories style) [LMT_2023_002].
5. **Advantages:** (b) the statistically strongest primary metric plus a view of task ability; (c) none that I can verify at this scale without paying an API.
6. **Disadvantages:** (b) benchmarks below about 50M parameters carry no signal; (c) paid, noisy.
7. **Recommendation:** (b). Primary: bits per byte on the fixed held-out slice (at least 5M tokens). Secondary: HellaSwag, ARC-Easy, WinoGrande, LAMBADA, and BLiMP (1,000 pairs per phenomenon) through lm-evaluation-harness (MIT licence, active, [INF_2026_201]); report them for R3 and R4 only, always with the intervals above. Synthetic fact-storage (LMT_2024_005) stays with the Stage 4 design.
8. **Consequences:** any claim based on a benchmark difference below about 1.4 to 4 points is noise; BPB is the only metric that can carry the headline.
9. **Approval:** requested (this refines D-06).

### D-LM-13: where to train
1. **Decision:** which machines run which rung, and a spending ceiling.
2. **Context:** you have not stated a paid budget (Q-HW-2); the project's GPU is currently busy with the toy sweep.
3. **Research evidence:** sections 3.4 and 3.7 and A2 to A6.
4. **Available approaches:** (a) free only (1660 Ti, Colab, Kaggle); (b) free for R1 to R3, rented for R4; (c) rented throughout.
5. **Advantages:** (b) free for the decisive early gates, then about USD 370 (about 590 if utilisation is only 25 percent) for the whole R4 rung at 20 tok/param on an A100, or about USD 270 on an H100 (arithmetic below); (a) no cost.
6. **Disadvantages:** (a) caps the claim at about 50M; (b) needs a budget and a hardware change after R3, which is a new experiment (D-LM-10); (c) every iteration costs money.
7. **Recommendation:** (b), with R4 and any rented spending **not started before the Stage 1 go/no-go (G1) and the Stage 2 results**, as feasibility_assessment.md section 0 already says. Cheapest first step: the standard-model ladder of 3.6 on the 1660 Ti or a T4, about 75 or 22 hours. Please give a ceiling for rented compute; until then none is assumed. Note the free-tier caveats in 3.7 (queues, unverified weekly quota, 12-hour sessions, so every run must be resumable).
8. **Consequences:** sets the largest rung. Rented option prices, all as of 2026-11-13 to 15: A100 80 GB 2.19 (RunPod) to 3.19 (Lambda, 8-GPU node, effective 2026-11-16) USD/h; H100 SXM 4.39 to 4.89; marketplace (Vast.ai) medians A100 0.84, H100 2.47, RTX 4090 0.49, T4 0.15 [INF_2026_205]. Spot or marketplace machines can be interrupted (resumable runs needed).
9. **Approval:** requested; **budget answer needed**.

(Arithmetic for R4 on the H100: 40-unit total 7.6e19 FLOPs at 346 TFLOP/s = 61 h, USD 268 at 4.39/h; on the A100 7.6e19 / 125e12 = 169 h, USD 370. Computed.)

### D-LM-14: the A-versus-B protocol, and how the penalty's cost enters
1. **Decision:** how A and B are compared and what counts as a win.
2. **Context:** feasibility_assessment.md section 3 defines arms, f (fraction of the gap closed) and rho (equivalent size) and a resolution check. This decision fixes the LM-specific rules; numeric margins are still pre-registration items (Phase 2) because seed variance is unmeasured.
3. **Research evidence:** record claims in the speedrun require a significance of p < 0.01 on mean validation loss across logged runs because of inter-run variance [LMT_2024_004]; FineWeb used two models per condition and averaged [LMT_2024_208]; Chinchilla-style loss-versus-size curves give rho [LMT_2022_202]. Penalty overhead, computed below.
4. **Available approaches:** (a) equal tokens only; (b) equal tokens as primary, equal total FLOPs as secondary; (c) equal FLOPs only.
5. **Advantages:** (b) matches your stated goal (same data), and also answers "is it just extra compute?".
6. **Disadvantages:** (b) two analyses; equal-FLOPs needs B or C to get extra tokens equal to A's overhead, which pushes against the data-matched setting.
7. **Recommendation (proposal; margins to be fixed after the resolution check):**
   - **Data:** same corpus, same tokenizer, same token budget and the same data order for every arm within a rung.
   - **Seeds:** at least 3 per arm and size (5 for the arms in the headline claim), seeds fixed in advance; separate seeds for tuning and for the verdict, as the project already does in Stage 1 (D-S1-3).
   - **Resolution check first:** train only standard models at P and 1.5P (and 0.75P, 2P); proceed only if L_C - L_B exceeds about 4 times the seed standard deviation (the factor 4 is my proposal).
   - **Tuning:** equal number of tuning runs for every arm (D-LM-8), best-of-grid learning rate and weight decay per arm, final verdict on fresh seeds.
   - **Win:** f >= 0.75 with a 95 percent interval above 0 (f = (L_C - L_A)/(L_C - L_B)), A not worse than D or E beyond noise (otherwise the SAE adds nothing), and rho >= 1.5 read from the standard-size curve; **loss and bits per byte only** decide, never SAE-based metrics. A loss between 0 and 0.75 is "partial", as in the outcome table of the feasibility assessment.
   - **Penalty cost (arithmetic, assumptions stated):** a jointly trained SAE with F = 8d latents on one layer adds 3 x (2dF + 2Fd) = 96 d^2 FLOPs per token (forward and backward), against 6N for the model. That is +10 percent at 10.6M, +10 at 24M, +9 at 47M, +7 at 126M, +5 at 346M per SAE layer, and about +40 percent at 10.6M if four layers carry an SAE. A full Gram-matrix overlap penalty on all 16,384 embedding rows, once per step, adds 3 x 2 x V^2 x d FLOPs: +9.9, +6.6, +4.5, +2.5, +1.2 percent at the five rungs (batch 65,536 tokens; subsampling rows lowers it). (The earlier toy-model estimate of 2 to 3x for the SAE in Stage 2 is a different, much smaller setting; these LM numbers do not replace it.) Always report measured wall-clock and FLOPs overhead per arm.
   - **How cost enters the verdict:** the primary comparison is at equal tokens (your criterion: matched training data); the overhead is reported next to it; as a secondary analysis, plot loss against total FLOPs including the penalty and give C and B the matching extra tokens. At deployment the penalty and any SAE are not part of the model (they are training-time tools), so file size and inference cost of A equal those of C.
   - **Resources:** the five-arm design is about 40 runs per rung (3.6).
8. **Consequences:** the test can end as "inconclusive by design" at a rung where the gap is not resolvable; that is a valid outcome, not a failure of the pipeline.
9. **Approval:** requested; numeric thresholds after the resolution check.

### D-LM-15: quantisation format and runtime
1. **Decision:** how A and B are compressed and measured.
2. **Context:** U-04 keeps quantisation in the headline; the test must compare equal bytes, not only equal ratios (feasibility assessment section 2.7).
3. **Research evidence:** llama.cpp (MIT, active) converts Hugging Face models to GGUF and quantises with `llama-quantize`; measured bits per weight Q8_0 8.5008, Q4_K_M 4.8944, F16 16.0005; it reports perplexity and KL divergence; it has options to keep the embedding or output tensor at higher precision [QNT_2026_201]; its conversion code includes Llama, GPT-2 and GPT-NeoX architectures [QNT_2026_202]. GPTQ: OPT-125M WikiText2 perplexity 27.65 at 16 bits, 31.12 with GPTQ 4-bit, 37.28 with round-to-nearest 4-bit, 3-bit GPTQ 53.85 and RTN 1,300; the same method loses 0.03 at 175B [QNT_2022_201]. So small models are far more fragile than large ones, and the saved model files of the GPTQ and AWQ libraries are a poor fit for our sizes. AWQ (protect the 0.1 to 1 percent salient channels) was shown on OPT-6.7B and larger [QNT_2023_202, partial]; LLM.int8's outlier problem appears only from about 6.7B parameters, so our sizes are below it [QNT_2022_203, partial]; bitsandbytes provides int8 and NF4 [QNT_2023_204, A]. **AutoGPTQ and AutoAWQ are archived (read-only) on GitHub as of 2026-11-15; bitsandbytes and llama.cpp are active** [INF_2026_201].
4. **Available approaches:** (a) llama.cpp GGUF (Q8_0, Q4_K_M) with perplexity/KL tools; (b) bitsandbytes int8/NF4 on GPU; (c) GPTQ/AWQ; (d) own round-to-nearest fake-quantiser in PyTorch.
5. **Advantages:** (a) the deployment format you named (laptop inference), exact byte counts; (d) a controlled, identical quantiser for A and B and for feature-survival measurements (RTN is the method of the quantisation-and-features paper QNT_2026_003); (b) fast on GPU.
6. **Disadvantages:** (a) quantisation recipes (k-quants mix types per tensor) make A/B differences harder to attribute; (b) different kernels from llama.cpp; (c) unmaintained libraries and weak small-model evidence.
7. **Recommendation:** (d) for the research comparison (per-channel or group-wise round-to-nearest at 8 and 4 bits, applied to all linear layers; embeddings handled the same way for both arms and reported both ways), plus (a) for the deliverable (Q8_0 and Q4_K_M files, perplexity and KL against the fp16 model). Include an **equal-bytes arm**: A quantised to size S against any standard model that also lands at size S (including a larger standard model quantised harder). No GPTQ/AWQ. Run at both token regimes of D-LM-7, because more tokens per parameter hurt quantisation [QNT_2025_104, A].
8. **Consequences:** parameter counts translate into bytes as in Table 1b; embeddings are 10 to 39 percent of the bytes at 10 to 126M, so how they are quantised matters.
9. **Approval:** requested.

### D-LM-16: code base
1. **Decision:** what trains the models.
2. **Context:** the penalty needs hooks into the residual stream, an auxiliary loss and eventually a second optimiser for a jointly trained SAE; U-07 says reuse standard components, new work is the training path.
3. **Research evidence:** nanoGPT is a short, readable, hackable loop (MIT; last push 2025-11-12) [LMT_2024_223, INF_2026_201]; modded-nanogpt is a heavily optimised speedrun (MIT) [LMT_2024_004]; nanotron (Apache-2.0) trained the FineWeb ablations and SmolLM2 across many GPUs [LMT_2024_208, LMT_2025_206]; TinyLlama used lit-gpt [LMT_2024_207]; lm-evaluation-harness (MIT) evaluates Hugging Face models [INF_2026_201].
4. **Available approaches:** (a) own loop, nanoGPT-style, about 300 lines, Llama blocks; (b) Hugging Face Trainer; (c) nanotron or lit-gpt; (d) fork the speedrun code.
5. **Advantages:** (a) every tensor accessible for hooks, small enough to audit and test, easy to checkpoint per D-LM-11. (b/c) less code to write.
6. **Disadvantages:** (a) more of your own code to get right (you already wrote and tested the toy core); (b/c) heavy abstractions and multi-GPU machinery not needed at 126M, harder to insert a second optimiser; (d) confounds (D-LM-2).
7. **Recommendation:** (a), with an exporter to Hugging Face LlamaForCausalLM format so that GGUF conversion and evaluation use standard tools.
8. **Consequences:** Phase 3 includes unit tests (shape, parameter count against Table 1a, determinism under a fixed seed, resume equals uninterrupted run).
9. **Approval:** requested.

---

## 5. What must be decided before any training starts

| # | Item | Status |
|---|---|---|
| 1 | D-LM-1 to D-LM-16 approved or changed by you | open |
| 2 | Stage 1 go/no-go (G1) outcome and the decision to enter Stage 4 at all (feasibility section 0: do not commit to rented spending before G1) | open; the baseline ladder alone can start earlier because it does not depend on the penalty |
| 3 | Throughput benchmark on the 1660 Ti and a T4 to replace A2/A3 (needs the GPU, so after the current sweep) | open |
| 4 | Your rented-compute ceiling (Q-HW-2) and hours per week (Q-SCOPE-1) | open |
| 5 | Tokenizer trained and checked; corpus downloaded, tokenised, held-out slice fixed and hashed | open |
| 6 | Pre-registration of the win rule, margins and seed lists (Phase 2), after the resolution check | open |
| 7 | Evaluation harness installed and tested on a public small checkpoint (this checks the pipeline, not our models) | open |
| 8 | Parameter-count test and resume test pass (D-LM-16) | open |
| 9 | Your own Kaggle weekly quota read from your account page | open |

**Earliest realistic start (judgement, not a promise).** After approval: about 1 week of work to train the tokenizer, stage the data and write and test the loop on CPU and a tiny model; the first real GPU runs (learning-rate sweep and the standard-model ladder at R1/R2, about 75 GPU-hours on the 1660 Ti or about 22 on a T4 for the seeds alone) only when a GPU is free. So **earliest first meaningful baseline results: about 2 to 3 weeks after approval, if approval arrives while Stage 1 is still running**; the penalty arms cannot start before Stage 2 designs exist. The roadmap's Stage 4 (6 to 8 weeks) remains the place where the full test lives. No training is started by this proposal.

---

## 6. Honest limits: what I could not verify

- **No measured throughput** for the 1660 Ti or the T4 on a transformer; A2 and A3 are assumptions and could be off by 1.5 to 2x. Everything in Table 2 for those devices scales with them.
- **Kaggle weekly GPU quota:** not stated in Kaggle's own documentation page; the 30-hour figure is secondary and unverified. Kaggle docs were read through a third-party text renderer because the page is script-driven (INF_2026_203); I did not log in.
- **Colab:** limits are unpublished; paid prices and compute-unit rates are not on the FAQ page; not checked elsewhere.
- **Rented prices** are list prices read on 2026-11-13 to 15 and change; Lambda's table is marked as taking effect 2026-11-16; marketplace prices fluctuate; availability not tested. I used on-demand RunPod prices; no account was created.
- **GTX 1660 Ti specification** comes from a secondary source (Wikipedia via a summariser; TechPowerUp refused access and NVIDIA's page returned 404); your card may be the laptop variant with a lower clock.
- **GPT-2 tokenizer licence**, Llama tokenizer licence and SlimPajama's dataset-card licence were not verified; SlimPajama's card returned HTTP 401.
- **TinyStories token count** (about 0.45B) is an estimate from file size, not a count; the token count of FineWeb-Edu under our own 16k tokenizer is not measured.
- **Seed-to-seed spread** of small LMs is unmeasured; I do not know whether a 1.5x gap is resolvable at R1 to R4. The Chinchilla-law gap estimate (0.1 to 0.23 nats) is an extrapolation outside the fitted range and uses a disputed fit.
- **Papers read at abstract or first-page level only (A):** Muennighoff (data-constrained), Sardana (inference-adjusted laws), McCandlish (noise scale), RMSNorm, RoPE, AdamW, Press and Wolf, QLoRA, Kumar (precision laws), Ouyang (undertrained quantisation). Claims that depend on them are flagged "A" above and should be checked before they ground an approved decision; Ouyang and Kumar especially, since D-LM-7 and D-LM-15 lean on them for "more tokens hurt quantisation". The WSD-equivalence claim rests on a paper I did not open.
- **Papers read in the sections named in lm_sources.md, not end to end (F(text)):** the large papers (Chinchilla, Kaplan, Pythia, OLMo 2, SmolLM2, FineWeb, Gadre, Tao, Wortsman, GPTQ, AWQ, LLM.int8, Signal and Noise); tables and numbers I quote were read in the extracted text, figures were not viewed. PDF text extraction can garble tables; I cross-checked the Pythia table against its GPU-hour totals and against TinyLlama's quoted figure (both agree).
- **Penalty-cost numbers** (SAE +5 to 10 percent per layer, Gram +1 to 10 percent) are FLOP arithmetic with stated settings; wall-clock overhead is typically larger because of memory traffic, and was not measured.
- **llm.c 1.6B/350M claims and commenter timings** (for example 8 H100 in about 40 minutes) were taken from a discussion thread through a summariser and are not used in any number above except the 124M run.
- **No paper on joint SAE-plus-LM training at LM scale** was re-searched here; this proposal covers the standard baseline only.
- **Needing the user's help:** no paywalled paper was needed. If you want any A-level paper upgraded, they are all on arXiv and can be read by me in a later session.
