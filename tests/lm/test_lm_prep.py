"""Text preparation: document splitting, tokenizer training, shard writing with a separate held-out shard."""

import json

import pytest

pytest.importorskip("tokenizers")

from lm.data import TokenShards, synthetic_documents, write_shards          # noqa: E402
from lm.prep_text import EOS, iter_docs, prepare                              # noqa: E402


def write_corpus(path, docs):
    path.write_text(EOS.join(docs) + EOS, encoding="utf-8")


def test_iter_docs_splits_on_the_marker_even_across_read_blocks(tmp_path):
    docs = [d.strip() for d in synthetic_documents(50, seed=1)]
    f = tmp_path / "a.txt"
    write_corpus(f, docs)
    for block in (7, 64, 1 << 20):                      # tiny blocks cut the marker and the documents in half
        assert list(iter_docs(f, block_bytes=block)) == docs
    assert len(list(iter_docs(f, max_docs=5))) == 5


def test_write_shards_boundary_and_batch_encoding_match_plain_encoding(tmp_path):
    texts = ["ab", "cd", None, "ef", "gh"]
    enc = lambda t: [ord(c) for c in t]
    a = write_shards(texts, enc, tmp_path / "a", vocab_size=256, tokens_per_shard=1000, val_shards=1, eos_id=0)
    b = write_shards(texts, enc, tmp_path / "b", vocab_size=256, tokens_per_shard=1000, val_shards=1, eos_id=0,
                     encode_batch=lambda xs: [enc(t) for t in xs], batch_docs=1)
    for meta in (a, b):
        assert [s["split"] for s in meta["shards"]] == ["train", "val"]       # the None closed the train shard
        assert [s["n_tokens"] for s in meta["shards"]] == [6, 6]
    assert (tmp_path / "a" / "shard_00001.bin").read_bytes() == (tmp_path / "b" / "shard_00001.bin").read_bytes()
    assert TokenShards(tmp_path / "a", "val").arrays[0].tolist() == [101, 102, 0, 103, 104, 0]


def test_prepare_keeps_heldout_text_apart_and_round_trips(tmp_path):
    from tokenizers import Tokenizer
    train = [d.strip() for d in synthetic_documents(300, seed=2)]
    valid = [d.strip() for d in synthetic_documents(40, seed=3)]
    write_corpus(tmp_path / "train.txt", train)
    write_corpus(tmp_path / "valid.txt", valid)
    out = tmp_path / "data"
    s = prepare(tmp_path / "train.txt", tmp_path / "valid.txt", out, vocab_size=400, tokens_per_shard=5000, log=lambda *_: None)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["vocab_size"] == 400 and meta["eos_id"] is not None
    assert [x["split"] for x in meta["shards"]][-1] == "val" and s["n_val_shards"] == 1
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    assert tok.get_vocab_size() == 400
    val = TokenShards(out, "val")
    ids = val.arrays[0].tolist()
    docs = [tok.decode(part) for part in _split(ids, meta["eos_id"])]
    assert docs == valid                                    # the held-out shard holds exactly the held-out documents
    assert s["val_bytes"] == sum(len(d.encode()) for d in valid)


def _split(ids, eos):
    cur = []
    for i in ids:
        if i == eos:
            yield cur
            cur = []
        else:
            cur.append(i)
