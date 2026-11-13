"""Learning-rate schedule matching the reference code (linear warm-up, cosine to a floor)."""

import math


def lr_at(step: int, peak_lr: float, n_steps: int, warmup_steps: int = 2000, final_frac: float = 0.05) -> float:
    """LR for 0-based `step`.

    Same as `get_lr` in the reference scripts (exp-10.py): it uses step + 1, a linear
    warm-up for (step + 1) < warmup_steps, then a cosine from peak_lr down to
    final_frac * peak_lr at the last step. If warmup_steps >= n_steps there is no
    warm-up and the cosine covers the whole run.
    """
    s = step + 1
    min_lr = final_frac * peak_lr
    if warmup_steps < n_steps:
        if s < warmup_steps:
            return peak_lr * s / warmup_steps
        progress = (s - warmup_steps) / (n_steps - warmup_steps)
    else:
        progress = s / n_steps
    return (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress)) + min_lr
