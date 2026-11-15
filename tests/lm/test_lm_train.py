"""Trainer: optimiser groups, schedule, learning, exact resume, manifest, refusals, auxiliary-loss hook, eval."""

import json
import math
from dataclasses import asdict

import pytest
import torch

import lm.train as lt
from lm.data import ByteTokenizer, TokenShards, write_shards
from lm.eval import evaluate
from lm.model import GPT, check_param_count, lm_loss
from lm.train import TrainConfig, decay_param_names, lr_at, make_optimizer, train
from lm_testutil import CLEAN_GIT, corpus_dir, tiny_cfg, tiny_model  # noqa: F401  (corpus_dir is a fixture)


def read_log(run_dir):
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]


def deterministic_view(records):
    """The log fields that must be identical between runs (everything except wall-clock time)."""
    return [{k: v for k, v in r.items() if k != "elapsed_s"} for r in records]


# ---------------------------------------------------------------- schedule and optimiser

def test_lr_schedule_warmup_peak_and_floor():
    kw = dict(peak_lr=1.0, total_steps=100, warmup_steps=10, final_frac=0.1)
    assert lr_at(0, **kw) == pytest.approx(0.1) and lr_at(9, **kw) == pytest.approx(1.0)
    assert lr_at(10, **kw) == pytest.approx(1.0)
    assert lr_at(99, **kw) == pytest.approx(0.1, abs=1e-3)
    mid = [lr_at(s, **kw) for s in range(10, 100)]
    assert all(a >= b for a, b in zip(mid, mid[1:]))                         # cosine decays monotonically


@pytest.mark.parametrize("variant", [dict(), dict(pos_encoding="rope", norm="rmsnorm", mlp="swiglu", bias=False),
                                     dict(tie_embeddings=False)])
def test_weight_decay_groups(variant):
    cfg = tiny_cfg("unused", model=tiny_model(**variant), weight_decay=0.1)
    model = GPT(cfg.model)
    decayed = decay_param_names(model)
    names = [n for n, _ in model.named_parameters()]
    for n, p in model.named_parameters():
        should_decay = p.ndim == 2 and ".attn." in n or p.ndim == 2 and ".mlp." in n
        assert (n in decayed) == should_decay, n
        if "norm" in n or n.endswith("bias") or n.startswith(("tok_emb", "pos_emb", "lm_head")):
            assert n not in decayed, n
    assert decayed and decayed < set(names)
    # the optimiser's groups partition the parameters exactly: decay group has wd 0.1, the rest 0
    opt = make_optimizer(model, cfg)
    wd = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    assert len(wd) == len(list(model.parameters()))
    assert {n for n, p in model.named_parameters() if wd[id(p)] == 0.1} == decayed
    # opting in adds the embeddings
    assert decay_param_names(model, decay_embeddings=True) - decayed >= {"tok_emb.weight"}


# ---------------------------------------------------------------- learning

@pytest.fixture()
def repeating_corpus(tmp_path):
    tok = ByteTokenizer()
    write_shards(["abcabcabcabcabcabcabcabc\n"] * 200, tok.encode, tmp_path / "rep", vocab_size=256,
                 tokens_per_shard=1500, val_shards=1, tokenizer_name=tok.name)
    return tmp_path / "rep"


@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_loss_decreases_on_repeating_corpus(tmp_path, repeating_corpus, pos):
    cfg = tiny_cfg(repeating_corpus, model=tiny_model(pos_encoding=pos), steps=60, peak_lr=1e-2, warmup_steps=5,
                   eval_every=30, ckpt_every=60)
    res = train(cfg, tmp_path / "run", git=CLEAN_GIT)
    log = read_log(tmp_path / "run")
    vals = [r for r in log if r["type"] == "val"]
    assert vals[0]["step"] == 0 and vals[0]["val_loss"] == pytest.approx(math.log(256), abs=0.3)   # starts at chance
    assert vals[-1]["val_loss"] < 0.3 * vals[0]["val_loss"]
    assert res["status"] == "ok" and res["final_step"] == 60


@pytest.mark.parametrize("variant", [dict(pos_encoding="rope", norm="rmsnorm", mlp="swiglu", bias=False, tie_embeddings=False),
                                     dict(pos_encoding="learned", norm="layernorm", mlp="gelu", dropout=0.1)])
def test_variants_train_without_error(tmp_path, corpus_dir, variant):
    cfg = tiny_cfg(corpus_dir, model=tiny_model(**variant), steps=6, eval_every=3, ckpt_every=6)
    train(cfg, tmp_path / "run", git=CLEAN_GIT)
    losses = [r["train_loss"] for r in read_log(tmp_path / "run") if r["type"] == "train"]
    assert len(losses) == 6 and all(math.isfinite(v) for v in losses)


def test_bf16_autocast_runs_on_cpu(tmp_path, corpus_dir):
    cfg = tiny_cfg(corpus_dir, steps=3, precision="bf16", eval_every=3, ckpt_every=3)
    assert train(cfg, tmp_path / "run", git=CLEAN_GIT)["status"] == "ok"


def test_fp16_needs_cuda(tmp_path, corpus_dir):
    with pytest.raises(ValueError):
        train(tiny_cfg(corpus_dir, steps=1, precision="fp16"), tmp_path / "run", device="cpu", git=CLEAN_GIT)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp16_runs_on_cuda(tmp_path, corpus_dir):
    cfg = tiny_cfg(corpus_dir, steps=3, precision="fp16", eval_every=3, ckpt_every=3)
    assert train(cfg, tmp_path / "run", device="cuda", git=CLEAN_GIT)["status"] == "ok"


def test_grad_accumulation_matches_one_big_batch_gradient(corpus_dir):
    """Mean of two micro-batch gradients equals the gradient of the mean loss (what the loop relies on)."""
    torch.manual_seed(0)
    model = GPT(tiny_model())
    x = torch.randint(0, 256, (4, 16))
    y = torch.randint(0, 256, (4, 16))
    model.zero_grad()
    lm_loss(model(x), y).backward()
    full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for xs, ys in ((x[:2], y[:2]), (x[2:], y[2:])):
        (lm_loss(model(xs), ys) / 2).backward()
    for g, p in zip(full, model.parameters()):
        assert torch.allclose(g, p.grad, atol=1e-6)


# ---------------------------------------------------------------- exact resume

@pytest.mark.parametrize("variant", [dict(dropout=0.1), dict(pos_encoding="rope", norm="rmsnorm", mlp="swiglu", bias=False)])
def test_exact_resume_matches_uninterrupted_run(tmp_path, corpus_dir, variant):
    cfg = tiny_cfg(corpus_dir, model=tiny_model(**variant), steps=20, ckpt_every=5, keep_last_ckpts=0)
    full_dir, crash_dir = tmp_path / "full", tmp_path / "crash"
    train(cfg, full_dir, git=CLEAN_GIT)

    partial = train(cfg, crash_dir, git=CLEAN_GIT, stop_at=12)       # "crashes" after step 12; last checkpoint is step 10
    assert partial["status"] == "interrupted" and partial["final_step"] == 12
    assert read_log(crash_dir)[-1]["step"] == 12 and (crash_dir / "ckpt_0000010.pt").exists()
    assert not (crash_dir / "ckpt_0000012.pt").exists()
    resumed = train(cfg, crash_dir, git=CLEAN_GIT, resume=True)
    assert resumed["status"] == "ok" and resumed["final_step"] == 20

    a, b = read_log(full_dir), read_log(crash_dir)
    assert deterministic_view(a) == deterministic_view(b)            # same losses, lrs, grad norms, val losses, no duplicates
    assert [r["step"] for r in b if r["type"] == "train"] == list(range(1, 21))
    ca = torch.load(full_dir / "ckpt_0000020.pt", weights_only=True)
    cb = torch.load(crash_dir / "ckpt_0000020.pt", weights_only=True)
    assert all(torch.equal(ca["model"][k], cb["model"][k]) for k in ca["model"])
    assert ca["data"]["batches_drawn"] == cb["data"]["batches_drawn"] == 40
    assert json.loads((crash_dir / "manifest.json").read_text())["resumes"][0]["from_step"] == 10


def test_resume_refuses_changed_config(tmp_path, corpus_dir):
    cfg = tiny_cfg(corpus_dir, steps=10)
    train(cfg, tmp_path / "run", git=CLEAN_GIT, stop_at=7)
    with pytest.raises(ValueError, match="config differs"):
        train(tiny_cfg(corpus_dir, steps=10, peak_lr=1e-3), tmp_path / "run", git=CLEAN_GIT, resume=True)
    with pytest.raises(ValueError, match="code is at"):
        train(cfg, tmp_path / "run", git={"commit": "other", "dirty_paths": []}, resume=True)


def test_existing_run_directory_is_never_overwritten(tmp_path, corpus_dir):
    cfg = tiny_cfg(corpus_dir, steps=2, eval_every=2, ckpt_every=2)
    train(cfg, tmp_path / "run", git=CLEAN_GIT)
    with pytest.raises(FileExistsError):
        train(cfg, tmp_path / "run", git=CLEAN_GIT)


def test_old_checkpoints_are_pruned(tmp_path, corpus_dir):
    cfg = tiny_cfg(corpus_dir, steps=20, ckpt_every=5, keep_last_ckpts=2)
    train(cfg, tmp_path / "run", git=CLEAN_GIT)
    assert sorted(p.name for p in (tmp_path / "run").glob("ckpt_*.pt")) == ["ckpt_0000015.pt", "ckpt_0000020.pt"]


# ---------------------------------------------------------------- manifest and refusals

def test_manifest_contents(tmp_path, corpus_dir):
    cfg = tiny_cfg(corpus_dir, steps=4, eval_every=2, ckpt_every=4)
    git = {"commit": "deadbeef", "dirty_paths": [" M lm/x.py"]}
    train(cfg, tmp_path / "run", device="cpu", git=git, command=["python", "-m", "lm.train"])
    m = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert m["config"] == json.loads(json.dumps(asdict(cfg))) and m["seed"] == cfg.seed
    assert m["git_commit"] == "deadbeef" and m["git_dirty"] is True and m["git_dirty_paths"] == [" M lm/x.py"]
    assert m["torch_version"] == torch.__version__ and m["device"] == "cpu" and m["precision"] == "fp32"
    assert m["param_counts"]["total"] == sum(p.numel() for p in GPT(cfg.model).parameters())
    assert m["data"]["tokenizer"] == "byte-level utf-8" and m["data"]["train_tokens"] > 0 and m["data"]["meta"]
    assert m["status"] == "ok" and m["final_step"] == 4 and m["final_val"]["step"] == 4
    assert m["run_id"] == "t_s3" and m["command"] == ["python", "-m", "lm.train"]


def write_config(path, cfg):
    path.write_text(json.dumps(asdict(cfg)))
    return path


def test_real_run_refuses_dirty_tree_but_smoke_and_allow_dirty_go(tmp_path, corpus_dir, monkeypatch, capsys):
    cfg_path = write_config(tmp_path / "cfg.json", tiny_cfg(corpus_dir, steps=3, eval_every=3, ckpt_every=3))
    out = ["--out-root", str(tmp_path / "out"), "--device", "cpu"]
    monkeypatch.setattr(lt, "git_state", lambda paths=("lm",): {"commit": "c1", "dirty_paths": ["?? lm/new.py"]})
    assert lt.main(["--config", str(cfg_path), *out]) == 1
    assert "Refusing to run" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()                                     # nothing was started
    assert lt.main(["--config", str(cfg_path), "--smoke", *out]) == 0          # smoke runs on a dirty tree...
    assert (tmp_path / "out" / "smoke" / "t_s3" / "manifest.json").exists()    # ...in its own folder
    assert lt.main(["--config", str(cfg_path), "--allow-dirty", *out]) == 0    # explicit override
    m = json.loads((tmp_path / "out" / "t_s3" / "manifest.json").read_text())
    assert m["git_dirty"] is True and m["smoke"] is False


def test_real_run_on_clean_tree_and_no_commit_refusal(tmp_path, corpus_dir, monkeypatch):
    cfg_path = write_config(tmp_path / "cfg.json", tiny_cfg(corpus_dir, steps=3, eval_every=3, ckpt_every=3))
    out = ["--out-root", str(tmp_path / "out"), "--device", "cpu"]
    monkeypatch.setattr(lt, "git_state", lambda paths=("lm",): {"commit": None, "dirty_paths": []})
    assert lt.main(["--config", str(cfg_path), *out]) == 1                     # no commit at all
    monkeypatch.setattr(lt, "git_state", lambda paths=("lm",): {"commit": "c1", "dirty_paths": []})
    assert lt.main(["--config", str(cfg_path), *out]) == 0
    assert json.loads((tmp_path / "out" / "t_s3" / "manifest.json").read_text())["git_dirty"] is False


def test_config_json_loading_ignores_notes_and_rejects_unknown_fields():
    cfg = TrainConfig.from_dict({"_note": "x", "run_name": "r", "model": {"_note": "y", "n_layer": 2, "d_model": 32,
                                                                       "n_head": 4}})
    assert cfg.run_name == "r" and cfg.model.n_layer == 2
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"stepz": 3})


def test_dryrun_config_file_loads():
    cfg = TrainConfig.from_json(lt.ROOT / "lm" / "configs" / "dryrun.json")
    assert cfg.run_id == "dryrun_s0" and cfg.model.pos_encoding == "rope"
    assert 3e5 < check_param_count(GPT(cfg.model))["total"] < 5e5          # "a few hundred thousand" parameters


# ---------------------------------------------------------------- auxiliary-loss hook

def test_aux_loss_hook_adds_to_the_loss_and_changes_training(tmp_path, corpus_dir):
    cfg = tiny_cfg(corpus_dir, steps=6, eval_every=6, ckpt_every=6, grad_accum=2)
    calls = []

    def aux(model, batch):
        x, y = batch
        calls.append(x.shape)
        penalty = 0.5 * model.blocks[0].attn.qkv.weight.pow(2).sum()
        return penalty, {"qkv_sq": penalty.item()}

    train(cfg, tmp_path / "base", git=CLEAN_GIT)
    train(cfg, tmp_path / "aux", git=CLEAN_GIT, aux_loss_fn=aux)
    assert len(calls) == 6 * 2                                                 # once per micro-batch
    base = [r for r in read_log(tmp_path / "base") if r["type"] == "train"]
    with_aux = [r for r in read_log(tmp_path / "aux") if r["type"] == "train"]
    assert "aux_loss" not in base[0]
    for r in with_aux:
        assert r["aux_loss"] > 0 and r["total_loss"] == pytest.approx(r["train_loss"] + r["aux_loss"])
        assert r["aux/qkv_sq"] == pytest.approx(r["aux_loss"], rel=1e-5)
    assert with_aux[0]["train_loss"] == base[0]["train_loss"]                  # step 1 forward is unchanged...
    assert with_aux[-1]["train_loss"] != base[-1]["train_loss"]                # ...but the gradient of the extra term moved the weights


def test_aux_loss_gradient_reaches_the_parameters(tmp_path, corpus_dir):
    """An auxiliary term that depends on one weight must produce a gradient there, and only through the term."""
    def aux(model, batch):
        return 7.0 * model.final_norm.weight.sum(), {}

    cfg = tiny_cfg(corpus_dir, model=tiny_model(norm="rmsnorm"), steps=1, grad_accum=1, grad_clip=0.0,
                   eval_every=1, ckpt_every=1, peak_lr=1e-3, warmup_steps=1)
    train(cfg, tmp_path / "run", git=CLEAN_GIT, aux_loss_fn=aux)
    ck = torch.load(tmp_path / "run" / "ckpt_0000001.pt", weights_only=True)
    assert (ck["model"]["final_norm.weight"] < 1.0).all()                       # pushed down by the +7*sum(w) term


# ---------------------------------------------------------------- evaluation

def test_evaluate_perplexity_and_bits_per_byte(corpus_dir):
    torch.manual_seed(0)
    model = GPT(tiny_model())
    val = TokenShards(corpus_dir, "val")
    x, y = val.fixed_windows(32, 8)
    res = evaluate(model, x, y, batch_size=3, device="cpu", bytes_per_token=val.bytes_per_token)
    assert res["loss"] == pytest.approx(math.log(256), abs=0.1)                  # untrained model is near chance
    assert res["perplexity"] == pytest.approx(math.exp(res["loss"]))
    assert res["bits_per_byte"] == pytest.approx(res["loss"] / math.log(2))      # byte-level: 1 byte per token
    assert res["n_tokens"] == 8 * 32 and model.training                          # mode is restored
    assert evaluate(model, x, y, batch_size=8, device="cpu")["loss"] == pytest.approx(res["loss"], rel=1e-6)
    assert "bits_per_byte" not in evaluate(model, x, y, batch_size=8, device="cpu")
