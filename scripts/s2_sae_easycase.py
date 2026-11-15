"""Positive control for toy/sae.py: SAEs on an EASY problem where the right answer is obvious.

    python scripts/s2_sae_easycase.py

Data: K = 20 random unit directions in m = 64 dimensions (nearly orthogonal), exactly one direction on per sample
with value Uniform(0.5, 1.5). Any correct SAE with N = 40 latents (2 x K) and sparsity near the truth should
recover the 20 directions almost perfectly (MCC close to 1, per-latent F1 close to 1, FVU near 0, L0 near 1).
A failure here means a bug in the SAE code, not a limit of the method. The crowded toy (1000 features in 15-63
dimensions) is NOT tested here; this only checks that the code can do the job at all.
"""

import sys
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy.sae import SAEBank   # noqa: E402

M, K, N, B, STEPS, LR = 64, 20, 40, 2048, 4000, 3e-3
dev = "cuda" if torch.cuda.is_available() else "cpu"


def sample(gen, dirs, B):
    idx = torch.randint(0, K, (B,), generator=gen, device=dev)
    val = 0.5 + torch.rand(B, generator=gen, device=dev)
    return val.unsqueeze(1) * dirs[idx], idx


g = torch.Generator(device=dev).manual_seed(0)
dirs = torch.randn(K, M, generator=torch.Generator().manual_seed(1)).to(dev)
dirs = dirs / dirs.norm(dim=1, keepdim=True)
print(f"max |cos| between two true directions: {(dirs @ dirs.T - torch.eye(K, device=dev)).abs().max():.2f}")
print(f"{'type':10s} {'sparsity':>8s} | {'MCC':>5s} {'F1':>5s} {'FVU':>6s} {'L0':>5s} {'dead':>5s} {'uniq':>5s}")
ok = True
for kind, grid in (("relu_l1", [0.003, 0.01, 0.03]), ("batchtopk", [1.0]), ("jumprelu", [1e-3, 3e-3, 1e-2])):
    R = len(grid)
    kw = {"k": grid} if kind == "batchtopk" else ({"l1": grid} if kind == "relu_l1" else {"l0": grid})
    sae = SAEBank(R, M, N, kind=kind, lr=LR, device=dev, seed=3, **kw)
    for step in range(STEPS):
        h, _ = sample(g, dirs, B)
        with torch.enable_grad():
            sae.step(h.unsqueeze(0).expand(R, B, M).contiguous())
    ge = torch.Generator(device=dev).manual_seed(99)
    h, idx = sample(ge, dirs, 8000)
    for r, sp in enumerate(grid):
        z = sae.encode_one(r, h)
        D = sae.W_dec[r].detach()
        C = (D @ dirs.T).abs()                                   # [N,K]
        rows, cols = linear_sum_assignment((-C).cpu().numpy())
        mcc = float(C[rows, cols].mean())
        fires = z > 0
        alive = fires.any(0)
        best = C.argmax(1)
        xs = (idx.unsqueeze(1) == best.unsqueeze(0))             # feature "on" for each latent's best match
        tp = (fires & xs).sum(0).float()
        prec, rec = tp / fires.sum(0).clamp_min(1), tp / xs.sum(0).clamp_min(1)
        f1 = float((2 * prec * rec / (prec + rec).clamp_min(1e-12))[alive].mean())
        rec_h = z @ D + sae.b_dec[r]
        fvu = float(((rec_h - h) ** 2).sum() / ((h - h.mean(0)) ** 2).sum())
        uniq = len(set(best[alive].tolist())) / max(1, int(alive.sum()))
        print(f"{kind:10s} {sp:8g} | {mcc:5.2f} {f1:5.2f} {fvu:6.3f} {float(fires.float().sum(1).mean()):5.2f} "
              f"{1 - int(alive.sum()) / N:5.2f} {uniq:5.2f}")
        ok = ok and (mcc > 0.9)
print("\nat least one setting per type should reach MCC > 0.9; all settings above 0.9:", ok)
