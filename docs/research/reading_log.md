# Reading Log (user)

Your notes and answers to the questions in [reading_list.md](reading_list.md). One section per item. Claude will read this before design discussions and fold confirmed points into the paper library and citation index.

Template:

## R# — Short title
- Date read:
- Sections read:
- Key points (own words):
- Answers to the questions:
- Disagreements / doubts:
- Ideas for our project:

---

## Needs user-provided PDF (Q-LIT-1 verification pass, 2026-11-14)

Written by Claude. Every paper below is either not located or not readable in full by the tools used; nothing here is a paper whose full text was read. "What I could not verify" says what remains unchecked. Papers that are freely available but simply were not reached are in the second list.

**A. Could not be located or read in full (a copy from you or a supervisor would help)**

| Title | Why it matters | What I could not verify |
|---|---|---|
| Lai & Heimersheim (2024) and Lai & Huang (2024) — SAEs and language models trained concurrently (titles not recorded; cited in the related work of Chen, Engels & Tegmark, SAE_2025_002) | The closest prior art to joint SAE + model training (project gap G1, Q-LIT-2). SAE_2025_002 says they train SAEs and models together, find better spliced loss, "only in toy settings". A LessWrong post by peterlai (gpt-circuits) found in the Q-LIT-2 search may be the same line of work. | Two searches did not return the papers. Not verified: exact setup, what losses were added, scale, whether the model's own loss improved, seeds. Please send the reference list entry or PDFs if you can find them. |
| Henighan et al. 2023 — Superposition, Memorization, and Double Descent (transformer-circuits.pub) | Data-limited regime: models store datapoints instead of features; relevant to the question whether more superposition means more memorisation (R5). | Not attempted this pass: the web page is very large and the fetch tool cannot read it reliably. A saved PDF of the page would let me read it. Only the first part was seen in session 1 (level A/S). |

**B. Open access, read only partly or not yet read (no action needed from you; listed so the gap is visible)**

| Item | Read so far | Not verified |
|---|---|---|
| SUP_2022_001 Toy Models of Superposition (arXiv PDF) | pages 3-17 (level F, partial) | geometry of superposition beyond uniform case, correlated features, privileged basis, computation in superposition, adversarial examples, appendices |
| SAE_2025_007 AxBench, SAE_2025_008 sparse probing, SAE_2025_013 DeepMind post | only abstract / secondary (levels A, S) | all methods and numbers |
| RAT_2025_004 weight-sparse transformers (capability numbers), RAT_2023_007 codebook features | level P / A as before | performance tables |
| KD_2025_001 Distillation Scaling Laws | App. A-D.4 (level P) | scaling-law equation and constants, G.1-G.4 (temperature, mixing, truncation) |
| LMT_2024_001 MobileLLM, LMT_2023_002 TinyStories, LMT_2024_003 BabyLM, LMT_2024_004 modded-nanogpt, KD_2024_002 MiniLLM, KD_2024_003 Gemma 2, KD_2025_004 gpt-oss card, RAT_2025_010 persona vectors | abstract / snippet only | methods, experiments, limitations (not central to the superposition-control plan, so deprioritised) |
| All level-P entries upgraded this session | HTML text through a summarising tool | exact numbers must be spot-checked against the PDFs before use in a paper; known weak spots: OSP Table 2 (garbled), SAEBench Table 1 (architecture count), Matryoshka and SAEBench figure-only results, Duan 2026 appendix inconsistencies |
| Venue claims (NeurIPS 2025 oral for Chanin et al., ICLR 2025 for Park et al., ICLR 2025 for Gao et al. and Allen-Zhu & Li, NeurIPS 2023 for Bondarenko et al., ACL 2025 for OSP, ICLR 2026 for Ge et al., ICML 2025 for SAEBench and Matryoshka) | taken from arXiv comments or search results in earlier sessions | not visible in the paper text that was read; check on the proceedings pages before citing a venue |
