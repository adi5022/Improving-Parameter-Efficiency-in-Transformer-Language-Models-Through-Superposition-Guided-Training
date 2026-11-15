"""Token shards on disk, seeded random-window batches, and a tiny synthetic corpus for dry runs.

Shard format: a directory with `meta.json` and flat binary files `shard_00000.bin`, ... of
uint16 (vocab <= 65536) or uint32 token ids, no header. meta.json lists each shard's split
("train" or "val"), token count and the number of UTF-8 bytes of text it came from (so
bits-per-byte can be computed). The split is by shard: the last `val_shards` shards are val.

No dataset or tokenizer is downloaded here. The dry-run corpus is generated procedurally:

    python -m lm.data --out results/lm_data/synthetic_v0 --docs 4000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

META = "meta.json"
DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


# ---------------------------------------------------------------- tokenizer

class ByteTokenizer:
    """Byte-level tokenizer: a token is one UTF-8 byte. vocab_size 256, so bytes per token is exactly 1."""
    vocab_size = 256
    name = "byte-level utf-8"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------- writing shards

def write_shards(texts: Iterable[str], encode: Callable[[str], list[int]], out_dir, *, vocab_size: int,
                 tokens_per_shard: int, val_shards: int = 1, tokenizer_name: str = "unknown",
                 eos_id: int | None = None) -> dict:
    """Tokenise `texts` and write shards. A shard is closed after the document that takes it to
    at least `tokens_per_shard` tokens, so a document never straddles two shards. If `eos_id`
    is given it is appended to every document. Returns the meta dict (also saved as meta.json)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = "uint16" if vocab_size <= 2**16 else "uint32"
    shards, buf, n_bytes = [], [], 0

    def flush():
        nonlocal buf, n_bytes
        if not buf:
            return
        name = f"shard_{len(shards):05d}.bin"
        np.asarray(buf, dtype=DTYPES[dtype]).tofile(out_dir / name)
        shards.append({"file": name, "n_tokens": len(buf), "n_bytes": n_bytes})
        buf, n_bytes = [], 0

    for text in texts:
        ids = encode(text) + ([eos_id] if eos_id is not None else [])
        if ids and max(ids) >= vocab_size:
            raise ValueError(f"token id {max(ids)} >= vocab_size {vocab_size}")
        buf.extend(ids)
        n_bytes += len(text.encode("utf-8"))
        if len(buf) >= tokens_per_shard:
            flush()
    flush()
    if len(shards) <= val_shards:
        raise ValueError(f"{len(shards)} shards written but {val_shards} are reserved for val; "
                         "need at least one train shard (lower tokens_per_shard or add text)")
    for i, s in enumerate(shards):
        s["split"] = "val" if i >= len(shards) - val_shards else "train"
    meta = {"format": 1, "dtype": dtype, "vocab_size": vocab_size, "tokenizer": tokenizer_name,
            "eos_id": eos_id, "shards": shards}
    (out_dir / META).write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------- reading shards

class TokenShards:
    """Read-only view of one split ("train" or "val") of a shard directory (memory-mapped)."""

    def __init__(self, data_dir, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.meta = json.loads((self.data_dir / META).read_text())
        self.split = split
        self.vocab_size = self.meta["vocab_size"]
        entries = [s for s in self.meta["shards"] if s["split"] == split]
        if not entries:
            raise ValueError(f"no '{split}' shards in {self.data_dir}")
        dtype = DTYPES[self.meta["dtype"]]
        self.arrays = [np.memmap(self.data_dir / s["file"], dtype=dtype, mode="r") for s in entries]
        for a, s in zip(self.arrays, entries):
            if len(a) != s["n_tokens"]:
                raise ValueError(f"{s['file']} has {len(a)} tokens, meta.json says {s['n_tokens']}")
        self.n_tokens = sum(len(a) for a in self.arrays)
        self.bytes_per_token = sum(s["n_bytes"] for s in entries) / self.n_tokens

    def _window(self, shard: int, start: int, length: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.arrays[shard][start:start + length]).astype(np.int64))

    def n_windows(self, block_size: int) -> list[int]:
        """Number of possible window start positions (window = block_size + 1 tokens) in each shard."""
        return [max(len(a) - block_size, 0) for a in self.arrays]

    def sample(self, batch_size: int, block_size: int, gen: torch.Generator):
        """batch_size random windows, uniform over all valid start positions in all shards.
        Returns x, y [batch_size, block_size] with y = x shifted by one token."""
        counts = self.n_windows(block_size)
        total = sum(counts)
        if total == 0:
            raise ValueError(f"every {self.split} shard is shorter than block_size + 1 = {block_size + 1}")
        cum = torch.tensor(counts).cumsum(0)
        draws = torch.randint(total, (batch_size,), generator=gen)
        shard_ids = torch.searchsorted(cum, draws, right=True)
        rows = []
        for r, s in zip(draws.tolist(), shard_ids.tolist()):
            start = r - (int(cum[s - 1]) if s > 0 else 0)
            rows.append(self._window(s, start, block_size + 1))
        w = torch.stack(rows)
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()

    def fixed_windows(self, block_size: int, n_windows: int):
        """A fixed, RNG-free evaluation set: the first n_windows non-overlapping windows, taken
        shard by shard from the start. Returns x, y [n, block_size] (fewer rows if the split is small)."""
        rows = []
        for s, a in enumerate(self.arrays):
            for start in range(0, len(a) - block_size, block_size):
                rows.append(self._window(s, start, block_size + 1))
                if len(rows) == n_windows:
                    break
            if len(rows) == n_windows:
                break
        if not rows:
            raise ValueError(f"no {self.split} window of {block_size + 1} tokens fits in any shard")
        w = torch.stack(rows)
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()


class BatchSampler:
    """Seeded stream of random-window batches. Its state (generator state + batches drawn) is the
    'data position' stored in checkpoints, so a resumed run continues the same stream."""

    def __init__(self, shards: TokenShards, block_size: int, seed: int):
        self.shards, self.block_size = shards, block_size
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
        self.batches_drawn = 0

    def next_batch(self, batch_size: int):
        self.batches_drawn += 1
        return self.shards.sample(batch_size, self.block_size, self.gen)

    def state_dict(self) -> dict:
        return {"generator_state": self.gen.get_state(), "batches_drawn": self.batches_drawn}

    def load_state_dict(self, state: dict) -> None:
        self.gen.set_state(state["generator_state"])
        self.batches_drawn = state["batches_drawn"]


# ---------------------------------------------------------------- synthetic corpus

_ADJ = ["red", "small", "old", "happy", "quiet", "bright", "tall", "clever", "lazy", "brave"]
_NOUN = ["fox", "bird", "girl", "boy", "dog", "cat", "farmer", "king", "robot", "teacher"]
_VERB = ["sees", "finds", "follows", "helps", "likes", "calls", "watches", "meets"]
_ACT = ["sing", "run", "sleep", "read", "play", "dance", "think"]
_PLACE = ["garden", "forest", "river", "village", "market", "castle", "hill"]
_TEMPLATES = [
    "the {a} {n} {v} the {a2} {n2}.",
    "a {n} lives in the {p}.",
    "every {n} likes to {act} in the {p}.",
    "the {a} {n} and the {n2} went to the {p}.",
    "in the {p} the {n} {v} a {a2} {n2}.",
]


def synthetic_documents(n_docs: int, seed: int = 0) -> Iterable[str]:
    """Procedurally generated sentences (3 to 8 per document, one document per line). Deterministic in `seed`."""
    rng = random.Random(seed)
    for _ in range(n_docs):
        sentences = []
        for _ in range(rng.randint(3, 8)):
            t = rng.choice(_TEMPLATES)
            sentences.append(t.format(a=rng.choice(_ADJ), a2=rng.choice(_ADJ), n=rng.choice(_NOUN),
                                      n2=rng.choice(_NOUN), v=rng.choice(_VERB), act=rng.choice(_ACT),
                                      p=rng.choice(_PLACE)))
        yield " ".join(sentences) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Write a synthetic byte-level shard directory for dry runs.")
    p.add_argument("--out", required=True)
    p.add_argument("--docs", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens-per-shard", type=int, default=150_000)
    p.add_argument("--val-shards", type=int, default=1)
    args = p.parse_args(argv)
    tok = ByteTokenizer()
    meta = write_shards(synthetic_documents(args.docs, args.seed), tok.encode, args.out, vocab_size=tok.vocab_size,
                        tokens_per_shard=args.tokens_per_shard, val_shards=args.val_shards, tokenizer_name=tok.name)
    for s in meta["shards"]:
        print(f"{s['file']}: {s['n_tokens']} tokens, split={s['split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
