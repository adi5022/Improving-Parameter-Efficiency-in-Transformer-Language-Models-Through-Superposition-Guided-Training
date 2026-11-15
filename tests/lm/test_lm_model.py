"""Model: shapes, causality, parameter counts, variants."""

import itertools

import pytest
import torch

from lm.model import GPT, ModelConfig, apply_rope, check_param_count, count_actual, param_counts, rope_tables
from lm_testutil import tiny_model


def all_variants():
    for pos, norm, mlp, tie, bias in itertools.product(("learned", "rope"), ("layernorm", "rmsnorm"),
                                                       ("gelu", "swiglu"), (True, False), (True, False)):
        yield tiny_model(pos_encoding=pos, norm=norm, mlp=mlp, tie_embeddings=tie, bias=bias)


def variant_id(c):
    return f"{c.pos_encoding}-{c.norm}-{c.mlp}-tie{c.tie_embeddings}-bias{c.bias}"


@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_output_shape(pos):
    m = GPT(tiny_model(pos_encoding=pos))
    idx = torch.randint(0, 256, (3, 17))
    assert m(idx).shape == (3, 17, 256)


def test_rejects_too_long_sequence():
    m = GPT(tiny_model(context_len=16))
    with pytest.raises(ValueError):
        m(torch.zeros(1, 17, dtype=torch.long))


@pytest.mark.parametrize("cfg", list(all_variants()), ids=variant_id)
def test_causality_all_variants(cfg):
    """Changing a token must not change the logits at earlier positions (but does change later ones)."""
    torch.manual_seed(0)
    m = GPT(cfg).eval()
    for p in m.parameters():                      # larger weights so the logits are not ~0 and the check means something
        p.data.add_(0.05 * torch.randn_like(p))
    idx = torch.randint(0, 256, (2, 20))
    for pos in (19, 10, 1):
        idx2 = idx.clone()
        idx2[:, pos] = (idx2[:, pos] + 7) % 256
        a, b = m(idx), m(idx2)
        assert torch.allclose(a[:, :pos], b[:, :pos], atol=1e-6)
        assert not torch.allclose(a[:, pos:], b[:, pos:], atol=1e-6)


@pytest.mark.parametrize("cfg", list(all_variants()), ids=variant_id)
def test_param_count_formula_equals_real_count(cfg):
    m = GPT(cfg)
    counts = check_param_count(m)             # raises on any component mismatch
    assert counts["total"] == sum(p.numel() for p in m.parameters())
    assert counts["non_embedding"] == counts["total"] - counts["embeddings"]


def test_param_count_other_shapes_and_d_ff_ratio():
    for cfg in [ModelConfig(n_layer=1, d_model=16, n_head=2, vocab_size=50, context_len=8, d_ff=0, d_ff_ratio=2.5),
                ModelConfig(n_layer=5, d_model=48, n_head=6, vocab_size=1000, context_len=100, mlp="swiglu", d_ff=130,
                            tie_embeddings=False, pos_encoding="rope", norm="rmsnorm", bias=False)]:
        check_param_count(GPT(cfg))
    assert ModelConfig(d_model=16, n_head=2, d_ff=0, d_ff_ratio=2.5).ff_dim == 40
    assert ModelConfig(d_model=16, n_head=2, d_ff=33).ff_dim == 33


def test_known_count_by_hand():
    # n_layer=1, d=4, ff=8, V=10, T=6, gelu, layernorm, bias, tied, learned positions
    cfg = ModelConfig(n_layer=1, d_model=4, n_head=2, d_ff=8, vocab_size=10, context_len=6)
    c = param_counts(cfg)
    assert c["token_embedding"] == 40 and c["position_embedding"] == 24
    assert c["attention"] == 4 * 12 + 4 * 4 + 12 + 4 == 80
    assert c["mlp"] == 4 * 8 + 8 + 8 * 4 + 4 == 76
    assert c["norms"] == 3 * 8 == 24 and c["lm_head"] == 0
    assert c["total"] == 40 + 24 + 80 + 76 + 24 == 244


def test_tied_embeddings_share_one_tensor():
    tied, untied = GPT(tiny_model(tie_embeddings=True)), GPT(tiny_model(tie_embeddings=False))
    assert tied.lm_head.weight is tied.tok_emb.weight
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert count_actual(untied)["lm_head"] == 256 * 32 and count_actual(tied)["lm_head"] == 0


def test_bias_off_means_no_bias_parameters():
    m = GPT(tiny_model(bias=False, norm="layernorm"))
    assert not [n for n, _ in m.named_parameters() if n.endswith("bias")]


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_head=4)
    with pytest.raises(ValueError):
        ModelConfig(d_model=36, n_head=4, pos_encoding="rope")   # head_dim 9 is odd
    with pytest.raises(ValueError):
        ModelConfig(norm="batchnorm")


def test_rope_is_a_rotation_and_position_zero_is_identity():
    cos, sin = rope_tables(10, 8, 10000.0)
    x = torch.randn(2, 3, 10, 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(y[:, :, 0], x[:, :, 0])                      # angle 0
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)   # a rotation keeps length


def test_rope_score_depends_on_relative_position_only():
    """q.k after rope is unchanged when both positions shift by the same amount."""
    cos, sin = rope_tables(32, 8, 10000.0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def score(i, j):
        return (apply_rope(q, cos[i:i + 1], sin[i:i + 1]) * apply_rope(k, cos[j:j + 1], sin[j:j + 1])).sum()

    assert torch.allclose(score(5, 2), score(15, 12), atol=1e-5)
    assert not torch.allclose(score(5, 2), score(5, 3), atol=1e-5)


def test_dropout_only_active_in_train_mode():
    m = GPT(tiny_model(dropout=0.5))
    idx = torch.randint(0, 256, (2, 16))
    m.eval()
    assert torch.equal(m(idx), m(idx))
    m.train()
    assert not torch.equal(m(idx), m(idx))
