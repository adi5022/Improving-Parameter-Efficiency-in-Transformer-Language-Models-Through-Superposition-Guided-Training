"""AdamW with signed weight decay.

Weight decay gamma is the knob that moves the toy model between weak and strong
superposition:
    gamma >= 0 : ordinary decoupled decay, p <- p * (1 - lr * gamma). Shrinks rows;
                 rare features lose their rows (weak superposition).
    gamma <  0 : row-wise "norm-to-one" pull,
                 p_i <- p_i + lr * gamma * p_i * (1 - 1/||p_i||).
                 A row with norm > 1 shrinks, a row with norm < 1 grows, so every row
                 norm is pulled toward 1 and nearly all features get stored (strong
                 superposition). Plain anti-decay (p * (1 + lr*|gamma|)) would grow
                 without bound; the (1 - 1/||p_i||) factor makes norm 1 a fixed point.

As in the reference code (and torch.optim.AdamW), the decay is applied first, using the
current lr, and outside the adaptive Adam scaling; then the standard Adam update runs.
"""

import math

import torch
from torch.optim import Optimizer


class AdamWNegDecay(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, eps, gamma = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]

                # 1. weight decay
                if gamma >= 0:
                    p.mul_(1 - lr * gamma)
                else:
                    row_norm = p.norm(dim=1, keepdim=True).add_(eps)
                    p.add_(gamma * p * (1 - 1 / row_norm), alpha=lr)

                # 2. Adam update with bias correction
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** t)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr / (1 - beta1 ** t))
        return loss
