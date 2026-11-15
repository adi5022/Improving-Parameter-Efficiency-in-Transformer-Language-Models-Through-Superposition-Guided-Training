"""Shards, memmap reading, seeded batching, split by shard, byte tokenizer, synthetic corpus."""

import json

import numpy as np
import pytest
import torch

from lm.data import BatchSampler, ByteTokenizer, TokenShards, synthetic_documents, write_shards
from lm_testutil import corpus_dir  # noqa: F401  (fixture)


def test_byte_tokenizer_round_trip():
    tok = ByteTokenizer()
    s = "hello, world é"
    assert tok.decode(tok.encode(s)) == s and max(tok.encode(s)) < 256


def test_synthetic_corpus_is_deterministic():
    assert list(synthetic_documents(5, seed=3)) == list(synthetic_documents(5, seed=3))
    assert list(synthetic_documents(5, seed=3)) != list(synthetic_documents(5, seed=4))


def test_write_shards_split_meta_and_raw_format(tmp_path):
    tok = ByteTokenizer()
    docs = list(synthetic_documents(60, seed=0))
    meta = write_shards(docs, tok.encode, tmp_path, vocab_size=256, tokens_per_shard=2000, val_shards=2,
                        tokenizer_name=tok.name)
    splits = [s["split"] for s in meta["shards"]]
    assert splits.count("val") == 2 and splits[-2:] == ["val", "val"] and splits[0] == "train"
    assert json.loads((tmp_path / "meta.json").read_text()) == meta
    # flat uint16 file with no header; byte-level, so tokens == bytes
    first = np.fromfile(tmp_path / meta["shards"][0]["file"], dtype=np.uint16)
    assert len(first) == meta["shards"][0]["n_tokens"] == meta["shards"][0]["n_bytes"]
    assert sum(s["n_tokens"] for s in meta["shards"]) == sum(len(d.encode()) for d in docs)
    assert TokenShards(tmp_path, "val").bytes_per_token == 1.0


def test_uint32_used_for_large_vocab(tmp_path):
    meta = write_shards(["a b c " * 50] * 6, lambda t: [100_000 + len(w) for w in t.split()], tmp_path,
                        vocab_size=200_000, tokens_per_shard=100, val_shards=1)
    assert meta["dtype"] == "uint32"
    assert TokenShards(tmp_path, "train").arrays[0].dtype == np.uint32


def test_token_id_out_of_vocab_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_shards(["abc"], lambda t: [300], tmp_path, vocab_size=256, tokens_per_shard=1, val_shards=0)


def test_needs_a_train_shard(tmp_path):
    with pytest.raises(ValueError):
        write_shards(["abc"] * 3, ByteTokenizer().encode, tmp_path, vocab_size=256, tokens_per_shard=10_000, val_shards=1)


def test_windows_are_shifted_and_come_from_the_split(corpus_dir):
    train = TokenShards(corpus_dir, "train")
    x, y = train.sample(8, 16, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 16) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    for row in torch.cat([x[:, :1], y], dim=1).tolist():      # each window occurs contiguously in some train shard
        assert any(any(a[s:s + 17].tolist() == row for s in range(len(a) - 16)) for a in train.arrays)


def test_split_by_shard_no_overlap(corpus_dir):
    meta = json.loads((corpus_dir / "meta.json").read_text())
    train, val = TokenShards(corpus_dir, "train"), TokenShards(corpus_dir, "val")
    n_train = sum(s["n_tokens"] for s in meta["shards"] if s["split"] == "train")
    assert train.n_tokens == n_train
    assert train.n_tokens + val.n_tokens == sum(s["n_tokens"] for s in meta["shards"])
    assert len(train.arrays) + len(val.arrays) == len(meta["shards"])


def test_sampler_is_seeded_and_restorable(corpus_dir):
    shards = TokenShards(corpus_dir, "train")
    a, b = BatchSampler(shards, 16, seed=5), BatchSampler(shards, 16, seed=5)
    assert torch.equal(a.next_batch(4)[0], b.next_batch(4)[0])
    assert not torch.equal(BatchSampler(shards, 16, seed=5).next_batch(4)[0], BatchSampler(shards, 16, seed=6).next_batch(4)[0])
    state = a.state_dict()
    expected = [a.next_batch(4)[0] for _ in range(3)]
    c = BatchSampler(shards, 16, seed=999)
    c.load_state_dict(state)
    assert all(torch.equal(e, c.next_batch(4)[0]) for e in expected) and c.batches_drawn == a.batches_drawn


def test_sampler_reaches_every_shard(corpus_dir):
    """Window starts are drawn uniformly over all shards (checked through the sampled windows themselves)."""
    shards = TokenShards(corpus_dir, "train")
    assert len(shards.arrays) > 1
    x, y = shards.sample(3000, 8, torch.Generator().manual_seed(0))
    rows = {tuple(r) for r in torch.cat([x[:, :1], y], dim=1).tolist()}
    hit = set()
    for s, a in enumerate(shards.arrays):
        windows = {tuple(a[i:i + 9].tolist()) for i in range(len(a) - 8)}
        if rows & windows:
            hit.add(s)
    assert hit == set(range(len(shards.arrays)))


def test_fixed_windows_are_deterministic_and_tile_the_text(corpus_dir):
    val = TokenShards(corpus_dir, "val")
    x1, y1 = val.fixed_windows(16, 10)
    x2, y2 = val.fixed_windows(16, 10)
    assert torch.equal(x1, x2) and torch.equal(y1, y2) and x1.shape == (10, 16)
    assert torch.equal(x1[1, :1], y1[0, -1:])        # consecutive windows are adjacent, not overlapping
    assert torch.equal(x1[:, 1:], y1[:, :-1])


def test_missing_split_is_an_error(corpus_dir):
    with pytest.raises(ValueError):
        TokenShards(corpus_dir, "test")
