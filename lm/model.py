"""A GPT-style decoder-only transformer, fully configurable.

Every default below is PLACEHOLDER, NOT APPROVED: architecture choices need the user's
approval (CLAUDE.md section 9, decision register D-08). Nothing here is a recommendation.

Structure (pre-norm):  x = emb(tokens) [+ learned positions]
                       for each block:  x = x + attn(norm(x));  x = x + mlp(norm(x))
                       logits = lm_head(final_norm(x))        (lm_head tied to the embedding or not)
Dropout (one rate, `dropout`): on the embedding output, on the attention weights, and on the
output of the attention and MLP sub-layers (the nanoGPT convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

POS_ENCODINGS = ("learned", "rope")
NORMS = ("layernorm", "rmsnorm")
MLPS = ("gelu", "swiglu")


@dataclass(frozen=True)
class ModelConfig:
    # PLACEHOLDER, NOT APPROVED: all values in this class.
    n_layer: int = 4
    d_model: int = 128
    n_head: int = 4
    d_ff: int = 0                    # hidden width of the MLP; 0 means round(d_ff_ratio * d_model)
    d_ff_ratio: float = 4.0          # used only when d_ff == 0 (note: swiglu has 3 matrices, gelu has 2)
    vocab_size: int = 256
    context_len: int = 128
    pos_encoding: str = "learned"    # "learned" | "rope"
    rope_base: float = 10000.0       # used only for rope
    norm: str = "layernorm"          # "layernorm" | "rmsnorm"
    mlp: str = "gelu"                # "gelu" | "swiglu"
    tie_embeddings: bool = True      # lm_head weight shared with the token embedding
    bias: bool = True                # biases in linear layers and layernorm (rmsnorm never has one)
    dropout: float = 0.0
    init_std: float = 0.02           # normal init for linear and embedding weights
    scale_residual_init: bool = True  # GPT-2 style: output projections get init_std / sqrt(2 * n_layer)

    def __post_init__(self):
        if self.pos_encoding not in POS_ENCODINGS or self.norm not in NORMS or self.mlp not in MLPS:
            raise ValueError(f"pos_encoding in {POS_ENCODINGS}, norm in {NORMS}, mlp in {MLPS}")
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if self.pos_encoding == "rope" and (self.d_model // self.n_head) % 2:
            raise ValueError("rope needs an even head dimension")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.n_layer < 1 or self.context_len < 1 or self.vocab_size < 1:
            raise ValueError("n_layer, context_len and vocab_size must be positive")

    @property
    def ff_dim(self) -> int:
        return self.d_ff if self.d_ff > 0 else round(self.d_ff_ratio * self.d_model)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def make_norm(cfg: ModelConfig) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d_model)
    return nn.LayerNorm(cfg.d_model, bias=cfg.bias)


def rope_tables(context_len: int, head_dim: int, base: float):
    """cos and sin tables [context_len, head_dim / 2] for rotary embeddings (half-split layout)."""
    inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.arange(context_len, dtype=torch.float32)[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x [B, H, T, hd]; rotate the pair (x[..., i], x[..., i + hd/2]) by the angle for position t."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.rope = cfg.pos_encoding == "rope"
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if self.rope:
            q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0,
                                           is_causal=True)
        return self.resid_dropout(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, ff = cfg.d_model, cfg.ff_dim
        self.swiglu = cfg.mlp == "swiglu"
        if self.swiglu:
            self.gate = nn.Linear(d, ff, bias=cfg.bias)
            self.up = nn.Linear(d, ff, bias=cfg.bias)
            self.down = nn.Linear(ff, d, bias=cfg.bias)
        else:
            self.fc = nn.Linear(d, ff, bias=cfg.bias)
            self.proj = nn.Linear(ff, d, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.swiglu:
            return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm, self.attn = make_norm(cfg), Attention(cfg)
        self.mlp_norm, self.mlp = make_norm(cfg), MLP(cfg)

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model) if cfg.pos_encoding == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        if cfg.pos_encoding == "rope":
            cos, sin = rope_tables(cfg.context_len, cfg.head_dim, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        cfg = self.cfg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, cfg.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, cfg.init_std)
        if cfg.scale_residual_init:
            out_std = cfg.init_std / math.sqrt(2 * cfg.n_layer)
            for blk in self.blocks:
                nn.init.normal_(blk.attn.proj.weight, 0.0, out_std)
                nn.init.normal_(blk.mlp.down.weight if blk.mlp.swiglu else blk.mlp.proj.weight, 0.0, out_std)

    def forward(self, idx):
        """idx [B, T] of token ids (T <= context_len) -> logits [B, T, vocab_size]."""
        B, T = idx.shape
        if T > self.cfg.context_len:
            raise ValueError(f"sequence length {T} exceeds context_len {self.cfg.context_len}")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        cos, sin = (self.rope_cos, self.rope_sin) if self.cfg.pos_encoding == "rope" else (None, None)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.lm_head(self.final_norm(x))


def lm_loss(logits, targets, reduction: str = "mean"):
    """Next-token cross entropy in float32 (nats). logits [B, T, V], targets [B, T]."""
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction)


# ---------------------------------------------------------------- parameter counting

def param_counts(cfg: ModelConfig) -> dict[str, int]:
    """Exact parameter count by component, from the config alone (no model is built).

    Components: token_embedding, position_embedding (learned only), attention and mlp (all
    layers), norms (all blocks + the final norm), lm_head (only when untied; a tied head adds
    nothing). `total` is the sum; `embeddings` = token + position + lm_head; `non_embedding` =
    total - embeddings. Biases are included when cfg.bias is true; rope and the causal mask have
    no parameters.
    """
    d, ff, L, b = cfg.d_model, cfg.ff_dim, cfg.n_layer, cfg.bias
    norm = d * (2 if (cfg.norm == "layernorm" and b) else 1)
    attn = d * 3 * d + d * d + ((3 * d + d) if b else 0)
    mlp = (2 * d * ff + ((ff + d) if b else 0)) if cfg.mlp == "gelu" else (3 * d * ff + ((2 * ff + d) if b else 0))
    comps = {
        "token_embedding": cfg.vocab_size * d,
        "position_embedding": cfg.context_len * d if cfg.pos_encoding == "learned" else 0,
        "attention": L * attn,
        "mlp": L * mlp,
        "norms": (2 * L + 1) * norm,
        "lm_head": 0 if cfg.tie_embeddings else cfg.vocab_size * d,
    }
    comps["total"] = sum(comps.values())
    comps["embeddings"] = comps["token_embedding"] + comps["position_embedding"] + comps["lm_head"]
    comps["non_embedding"] = comps["total"] - comps["embeddings"]
    return comps


def component_of(name: str) -> str:
    """Which `param_counts` component a parameter name belongs to."""
    if name.startswith("tok_emb"):
        return "token_embedding"
    if name.startswith("pos_emb"):
        return "position_embedding"
    if name.startswith("lm_head"):
        return "lm_head"
    if "norm" in name:
        return "norms"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    raise ValueError(f"unclassified parameter {name}")


def count_actual(model: nn.Module) -> dict[str, int]:
    """The real parameter count by the same components (shared tensors are counted once)."""
    out = {k: 0 for k in ("token_embedding", "position_embedding", "attention", "mlp", "norms", "lm_head")}
    for name, p in model.named_parameters():
        out[component_of(name)] += p.numel()
    out["total"] = sum(out.values())
    return out


def check_param_count(model: GPT) -> dict[str, int]:
    """Raise AssertionError unless the formula equals the real count, component by component and in total."""
    formula, actual = param_counts(model.cfg), count_actual(model)
    for k, v in actual.items():
        assert formula[k] == v, f"parameter count mismatch for {k}: formula {formula[k]}, actual {v}"
    assert actual["total"] == sum(p.numel() for p in model.parameters())
    return formula
