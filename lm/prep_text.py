"""Train our byte-level BPE tokenizer and write token shards from plain-text files (decisions D-LM-4, D-LM-5).

    python -m lm.prep_text --train TinyStoriesV2-GPT4-train.txt --valid TinyStoriesV2-GPT4-valid.txt \
        --out results/lm_data/tinystories_v0 --vocab-size 16384

Input: two text files; documents are separated by the marker `<|endoftext|>` (the TinyStories convention).
The tokenizer is trained on the TRAIN file only, so the held-out file never influences it. Output directory:
`tokenizer.json`, `shard_*.bin` and `meta.json` (format described in lm/data.py). The held-out text gets its
own shard(s); the last shard is the held-out one. The library used is Hugging Face `tokenizers`; the BPE
algorithm is theirs, not ours (D-LM-4).
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import chain
from pathlib import Path

from .data import write_shards

EOS = "<|endoftext|>"


def iter_docs(path, separator: str = EOS, block_bytes: int = 1 << 26, max_docs: int = 0):
    """Documents of a text file, streamed (64 MB blocks), split on `separator`, stripped, empty ones skipped."""
    carry, n = "", 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            block = f.read(block_bytes)
            if not block:
                break
            parts = (carry + block).split(separator)
            carry = parts.pop()
            for p in parts:
                p = p.strip()
                if p:
                    yield p
                    n += 1
                    if max_docs and n >= max_docs:
                        return
        tail = carry.strip()
        if tail and not (max_docs and n >= max_docs):
            yield tail


def train_tokenizer(docs, vocab_size: int, out_path):
    """Byte-level BPE (GPT-2 style: all 256 bytes are in the vocabulary, so any text can be encoded)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOS], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs, trainer)
    tok.save(str(out_path))
    return tok


def prepare(train_path, valid_path, out_dir, vocab_size: int, tokens_per_shard: int = 10_000_000,
            max_train_docs: int = 0, max_valid_docs: int = 0, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tok = train_tokenizer(iter_docs(train_path, max_docs=max_train_docs), vocab_size, out_dir / "tokenizer.json")
    log(f"tokenizer trained: {tok.get_vocab_size()} entries, {time.perf_counter() - t0:.0f}s")
    eos_id = tok.token_to_id(EOS)
    texts = chain(iter_docs(train_path, max_docs=max_train_docs), [None], iter_docs(valid_path, max_docs=max_valid_docs))
    meta = write_shards(texts, lambda t: tok.encode(t).ids, out_dir, vocab_size=vocab_size, tokens_per_shard=tokens_per_shard,
                        val_shards=1, tokenizer_name=f"byte-level BPE {vocab_size} (HF tokenizers {_version()})",
                        eos_id=eos_id, encode_batch=lambda b: [e.ids for e in tok.encode_batch(b)])
    shards = meta["shards"]
    val = [s for s in shards if s["split"] == "val"]
    trn = [s for s in shards if s["split"] == "train"]
    summary = {"train_tokens": sum(s["n_tokens"] for s in trn), "val_tokens": sum(s["n_tokens"] for s in val),
               "train_bytes": sum(s["n_bytes"] for s in trn), "val_bytes": sum(s["n_bytes"] for s in val),
               "n_train_shards": len(trn), "n_val_shards": len(val), "seconds": round(time.perf_counter() - t0)}
    summary["bytes_per_token_train"] = summary["train_bytes"] / max(1, summary["train_tokens"])
    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary))
    if summary["val_tokens"] < 5_000_000:
        log(f"WARNING: the held-out text has {summary['val_tokens']} tokens, less than the 5M that D-LM-5 asks for")
    return summary


def _version() -> str:
    import tokenizers
    return tokenizers.__version__


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--tokens-per-shard", type=int, default=10_000_000)
    p.add_argument("--max-train-docs", type=int, default=0, help="0 = all (use a small number only for tests)")
    p.add_argument("--max-valid-docs", type=int, default=0)
    args = p.parse_args(argv)
    prepare(args.train, args.valid, args.out, args.vocab_size, args.tokens_per_shard, args.max_train_docs, args.max_valid_docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
