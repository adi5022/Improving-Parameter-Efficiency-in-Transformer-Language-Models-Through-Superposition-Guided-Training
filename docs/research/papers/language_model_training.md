# Language-Model Training from Scratch (Small Scale)

Session 1 (2026-11-11). Records: [../paper_library.md](../paper_library.md). Coverage in session 1 is thin (mostly A/S); this area needs a dedicated pass before Phase 3 decisions.

## 1. Findings so far

- **Small-model design** (LMT_2024_001, MobileLLM, ICML 2024, abstract): at sub-billion scale, depth over width, embedding sharing and grouped-query attention matter; block-wise weight sharing adds ~0.7–0.8%.
- **Tiny settings that still produce coherent language** (LMT_2023_002, TinyStories): < 10M-parameter models generate coherent stories on a narrow synthetic corpus. Cheapest credible test bed; also used for interpretability (Matryoshka SAEs SAE_2025_005, gradient routing RAT_2024_008 used TinyStories).
- **Data-limited regime** (LMT_2024_003, BabyLM 2024, snippet): fixed 10M / 100M-word budgets; strongest entries came from architecture (LTG-BERT, GPT-BERT), more epochs helped, curriculum learning mostly failed, some teams distilled from a teacher. Established evaluation suite and a workshop venue.
- **Baseline strength** (LMT_2024_004, modded-nanogpt, snippet): the 124M FineWeb benchmark (val loss 3.28) is reached in minutes on 8×H100 with a heavily tuned recipe (Muon optimiser and more). Records must show p < 0.01 because seed variance is comparable to many claimed improvements. **Implication: any regulariser compared against an untuned baseline will look better than it is.**

- **Optimiser and architecture claims checked 2026-11-14 (level P):**
  - Muon (LMT_2025_010, "Muon is Scalable"): the "~2× efficiency" is a fitted scaling law over 399M-1.5B models (loss 2.506·C^−0.052 vs 2.608·C^−0.054, ≈ 52% of AdamW's FLOPs); AdamW was grid-searched but Muon reused AdamW's hyperparameters; proprietary data, single runs; Muon-SFT underperformed Adam-SFT on Qwen2.5-7B. Nothing is shown below ~400M. Matrix parameters use Muon; embeddings, head and norm gains use AdamW.
  - nGPT (LMT_2025_006): the 4-20× is in tokens/iterations at 0.5B-1B, step time is 60-80% higher, only the initial LR was tuned, and weight decay/warmup were removed for nGPT only. No sub-0.5B result, and the paper does not discuss superposition. Any "encourages superposition" interpretation is from Liu, Liu & Gore (SUP_2025_002), not from the nGPT authors.
  - Knowledge capacity (LMT_2024_005, LMT_2025_007): ~2 bits/parameter is the best case at 1000 exposures per fact (≥ 1 at 100 exposures, int4 more than halves it); ~3.6 bits/parameter comes from random uniform tokens in ≲ 20M-parameter models and is a lower bound; neither gives a number for ordinary single-pass web text.

## 2. Hardware reality (measured on this machine, 2026-11-11)

`nvidia-smi` reports an **NVIDIA GeForce GTX 1660 Ti, 6 GB** (Turing TU116). Relevant properties (from general knowledge; verify in Phase 3): no tensor cores; no native bf16; FlashAttention-2 requires Ampere or newer, so PyTorch SDPA falls back to memory-efficient or math kernels; peak ≈ 5.4 TFLOPS FP32 / ≈ 10.9 TFLOPS FP16. The FeatureScalpel docs mention an RTX 4050 laptop GPU (6 GB) — possibly a different machine (open question Q-HW-1).

Rough training cost with C ≈ 6·N·D FLOPs and an **assumed** effective 3 TFLOP/s (≈ 30% of FP16 peak — to be benchmarked):

| Params N | Tokens D (≈ 20·N) | FLOPs | Est. time on 1660 Ti | Est. time on one H100 at ~400 TFLOP/s effective |
|---|---|---|---|---|
| 10M | 200M | 1.2e16 | ≈ 1.1 h | < 1 min |
| 30M | 600M | 1.1e17 | ≈ 10 h | ≈ 5 min |
| 124M | 2.5B | 1.9e18 | ≈ 170 h | ≈ 1.3 h |

Non-embedding parameter counts and attention cost at long context change these figures; they are order-of-magnitude planning numbers only. A controlled study needs **many** runs (arms × seeds × baseline tuning), so the local GPU supports screening at ≈ 5–30M parameters; anything near GPT-2 Small scale needs rented GPUs.

## 3. Open items (Phase 3 prerequisites)

- Examine: Chinchilla, Pythia, Cramming, Muon, small-scale instability proxies (Wortsman et al.), tokeniser/vocabulary-size studies for small models.
- Benchmark actual throughput on the user's GPU(s).
- All architecture and training choices (layers, width, heads, context, positional encoding, norm, activation, tokenizer, corpus, optimiser, schedule) are **pending decisions** — see [../architectural_decisions.md](../architectural_decisions.md).
