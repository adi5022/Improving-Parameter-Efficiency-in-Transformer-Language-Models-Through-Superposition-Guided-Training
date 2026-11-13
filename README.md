# Improving Parameter Efficiency in Transformer Language Models by Controlling Superposition During Training

A research project on one question: **can we train a language model so that it packs more concepts into the same number of dimensions, and does that make a smaller model as good as a bigger one?**

> **Status: early research.** The toy-model baseline is being run. There are no results on language models yet, and nothing here is a claim that the idea works.

## The idea

A language model knows far more concepts than it has dimensions to store them. It copes by **superposition**: each concept gets a direction in the model's activation space, and many directions overlap. Overlap saves space but causes interference between concepts, so the model has to balance the two.

How much a model packs is not fixed. Recent work shows it can be steered, for example through weight decay. This project asks whether we can steer it on purpose during training, so the model packs more at an acceptable cost in interference.

Interpretability tools (such as sparse autoencoders) are used here to **measure** superposition and to turn the measurement into a training signal. They are not used to make the model easier to read. The target is a model that is denser and probably harder to interpret, but smaller for the same quality.

## How it will be judged

The end test is a fair comparison under a memory budget:

- **Model A:** trained with our method, about 2 GB after quantisation.
- **Model B:** trained normally, about 3 GB after quantisation.
- If A matches B on the same benchmarks, the method bought roughly a third of the memory. Better-looking features or nicer interpretability plots do not count as evidence.

## Plan

| Stage | What | State |
|---|---|---|
| 0 | Toy model of superposition: reproduce known behaviour and fix the baseline to beat | running |
| 1 | Toy model with a training penalty that increases packing, tested against tuned weight decay | designed, not approved |
| 2+ | Small language models trained from scratch, a jointly trained sparse autoencoder as the measuring tool, distillation from a larger teacher, then quantisation and local inference | not started |

Every significant design decision is written down and approved before it is built. Open decisions are listed in `docs/research/architectural_decisions.md`.

## Repository layout

```
toy/        toy model, data, batched GPU trainer, sweep runner, analysis
tests/      unit and equivalence tests
configs/    experiment grids and pass/fail gates (committed before runs start)
docs/
  research/ literature review, hypotheses, roadmap, decisions, citations
  notes/    numbered session journal
```

Start with [docs/research/README.md](docs/research/README.md) for the current direction, status and next step.

## Running the toy baseline

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-lock.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m toy.sweep --grid G-A
.venv/Scripts/python -m toy.sweep --grid G-B
.venv/Scripts/python -m toy.analysis
```

The sweep refuses to start if `toy/` or `configs/` have uncommitted changes, and every saved result records the commit it came from. An interrupted sweep resumes where it stopped. A CUDA GPU is used if present (developed on a 6 GB GTX 1660 Ti); the sweep also runs on CPU, much more slowly.

## Reproducibility

Experiments are configured by seed, grid and thresholds in `configs/stage0.json`. Results go to `results/` (not committed). Failed experiments and negative results are recorded in the session journal rather than left out.

## License

Not yet chosen.
