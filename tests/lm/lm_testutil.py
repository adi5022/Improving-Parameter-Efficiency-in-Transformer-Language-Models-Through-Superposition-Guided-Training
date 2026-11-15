"""Helpers shared by the lm/ tests (imported by name; tests/lm is on sys.path under pytest's default import mode)."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only: never touch a GPU that a sweep may be using

import pytest

from lm.data import ByteTokenizer, synthetic_documents, write_shards
from lm.model import ModelConfig
from lm.train import TrainConfig

CLEAN_GIT = {"commit": "abc123", "dirty_paths": []}


def tiny_model(**kw) -> ModelConfig:
    base = dict(n_layer=2, d_model=32, n_head=4, d_ff=64, vocab_size=256, context_len=32)
    base.update(kw)
    return ModelConfig(**base)


def tiny_cfg(data_dir, model=None, **kw) -> TrainConfig:
    base = dict(run_name="t", seed=3, data_dir=str(data_dir), model=model or tiny_model(), steps=20, batch_size=4,
                grad_accum=2, peak_lr=3e-3, warmup_steps=3, eval_every=5, eval_windows=8, eval_batch_size=4,
                log_every=1, ckpt_every=5)
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory):
    """Small synthetic byte-level corpus: a few train shards and one val shard.
    (A fixture imported into each test module; a tests/lm/conftest.py would shadow tests/conftest.py.)"""
    out = tmp_path_factory.mktemp("corpus")
    tok = ByteTokenizer()
    write_shards(synthetic_documents(150, seed=1), tok.encode, out, vocab_size=256, tokens_per_shard=6000,
                 val_shards=1, tokenizer_name=tok.name)
    return out
