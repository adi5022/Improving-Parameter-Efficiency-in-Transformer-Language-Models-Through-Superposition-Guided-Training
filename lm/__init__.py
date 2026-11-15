"""Training pipeline for a standard decoder-only transformer language model.

This package is machinery only. Every architecture and training choice is configuration;
the default values in the code are PLACEHOLDER, NOT APPROVED (CLAUDE.md section 9).
See lm/README.md.

    model.py   the transformer, and an exact parameter-count formula
    data.py    token shards (memmap), seeded random-window batches, a byte tokenizer, a synthetic corpus
    eval.py    validation loss / perplexity / bits-per-byte on a fixed set of windows
    train.py   command-line trainer: config, manifest, checkpoints, exact resume, auxiliary-loss hook
"""
