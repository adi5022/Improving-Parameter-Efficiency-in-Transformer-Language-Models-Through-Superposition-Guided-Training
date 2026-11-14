"""Poke at a trained toy model.

    python scripts/probe.py --width 25 --gamma -1.0            # try one trained model
    python scripts/probe.py --width 25 --gamma 1.0 --features 0 1 500

Loads results/stage0/G-A/a1_m<width>_g<gamma>_s<seed>.pt, switches on the features you
name (value 1.0 each), runs the model, and prints how well each one comes back out.
Then it measures reconstruction quality for frequent vs rare features on fresh data.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from toy.data import feature_probs, sample_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", default="G-A")
    p.add_argument("--features", type=int, nargs="*", default=[0, 3, 200],
                   help="feature numbers to switch on (0 = the most frequent feature)")
    a = p.parse_args()

    run = f"a1_m{a.width}_g{a.gamma:+.2f}_s{a.seed}"
    res = torch.load(ROOT / "results" / "stage0" / a.grid / f"{run}.pt", weights_only=False)
    W, b = res["W"], res["b"]
    n = W.shape[0]
    print(f"model {run}: {n} features squeezed into {a.width} numbers; test error {res['test_loss']:.2e}\n")

    def model(x):
        return torch.relu(x @ W @ W.T + b)

    x = torch.zeros(1, n)
    x[0, a.features] = 1.0
    y = model(x)[0]
    print("You switched on features", a.features, "with value 1.0 each.")
    print("What came out for those features:", [round(float(y[i]), 3) for i in a.features])
    top = torch.topk(y, 6)
    print("Six strongest outputs (feature, value):", [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)])
    print(f"Biggest output on a feature you did NOT switch on: "
          f"{float(y[[i for i in range(n) if i not in a.features]].max()):.3f}  (0 = no interference)\n")

    g = torch.Generator().manual_seed(123)
    data = sample_batch(20000, feature_probs(n, 1.0), g)
    out = model(data)
    err = ((out - data) ** 2).mean(dim=0)          # error per feature
    for name, sl in [("most frequent 10", slice(0, 10)), ("features 10-100", slice(10, 100)),
                     ("features 100-500", slice(100, 500)), ("rarest 500", slice(500, n))]:
        print(f"average error, {name:>17}: {float(err[sl].mean()):.2e}")


if __name__ == "__main__":
    main()
