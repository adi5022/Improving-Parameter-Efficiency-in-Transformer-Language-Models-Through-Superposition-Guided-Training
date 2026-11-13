# Knowledge Distillation

Session 1 (2026-11-11). Records: [../paper_library.md](../paper_library.md).

## 1. Findings so far

- **Distillation scaling laws** (KD_2025_001, ICML 2025, level P): pure logit KD (λ = 1, τ = 1), students 143M–12.6B, C4.
  - **Capacity gap:** as teacher strength grows, student loss first falls then rises — a much stronger teacher can produce a worse student, attributed to the growing teacher–student KL.
  - KD beats supervised pretraining only (a) below a student-compute threshold that grows with student size and (b) when the teacher already exists or serves other purposes. Counting teacher training cost, supervised learning wins.
  - Requires full teacher logits over the vocabulary (shared vocabulary implied).
- **On-policy reverse-KL KD** (KD_2024_002, MiniLLM, ICLR 2024, abstract): better generative students 120M–13B.
- **KD as the pretraining objective** (KD_2024_003, Gemma 2, abstract): 2B/9B trained by KD instead of next-token prediction.
- **Sequence-level KD as data generation** is what TinyStories (LMT_2023_002) is in practice.

## 2. The README's proposed teacher: gpt-oss-120b (KD_2025_004, snippet level)

- 116.8B total / ~5.1B active parameters; MXFP4 MoE weights; fits one 80 GB GPU (H100 / MI300X).
- Tokenizer **o200k_harmony, 201,088 tokens.**

Consequences:

1. **Logit-level KD from gpt-oss-120b into a small student is impractical.** It requires either the student to adopt the 201k vocabulary — at d_model = 512 a tied embedding alone is ≈ 103M parameters, which dwarfs a 10–30M model — or a cross-tokenizer KD method (not yet examined). Storing top-k logits per token for hundreds of millions of tokens is also a large storage cost.
2. **Sequence-level KD (synthetic text) is feasible** via API or a rented 80 GB GPU, but it changes the training *data*. Then "KD vs baseline" mostly measures data quality and distribution, not a training method. The README's 2×2 design must hold **token count and compute** fixed and should include a control where the baseline gets equally clean non-teacher data.
3. **Capacity gap** (KD_2025_001) argues against the largest available teacher for a tiny student.
4. The README's instruction-tuning goal for a model of this size is a separate project; instruction following at ≤ 30M parameters is very limited outside narrow domains (TinyStories-style).

## 3. Interaction with representation-aware training

Not found in session-1 searches: work testing whether teacher supervision changes the student's internal features or interacts with representation regularisers. Closest: SAE-FD (2026 preprint, snippet only) uses SAE feature space as an anchor in continual learning, not teacher→student. This interaction is open but is a *third-order* question; it should not be attempted before the first-order question (does the regulariser do anything?) has an answer.

## 4. Recommendation (pending approval — see architectural_decisions.md D-07)

Postpone KD to an optional late phase. If kept: a teacher with a small or shared vocabulary (e.g. a model trained by us, or an open model whose tokenizer the student can adopt), logit KD on the same tokens as the baseline, and the teacher choice justified against the capacity-gap result.

## 5. Open items

- Read KD_2025_001 Appendix A (limitations) and G.4 (top-k/top-p truncation).
- Examine Hinton 2015, Kim & Rush 2016, cross-tokenizer KD (e.g. ULD, arXiv:2402.12030).
