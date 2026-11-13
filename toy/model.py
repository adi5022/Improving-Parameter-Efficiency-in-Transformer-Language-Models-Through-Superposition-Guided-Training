"""Toy model of superposition (Liu, Liu & Gore 2025; Elhage et al. 2022).

n_features features, each a sparse input coordinate. The model squeezes them through a
bottleneck of `width` dimensions and tries to reconstruct them:

    h = x W          (encode: [batch, n] @ [n, m] -> [batch, m])
    y = ReLU(h W^T + b)   (decode with the same W, then ReLU)

Row W_i is the direction in the m-dimensional bottleneck where feature i is stored.
If ||W_i|| is near 0 the feature is dropped; if it is near 1 it is stored. With m < n
the rows cannot all be orthogonal, so storing more than m features means the rows
overlap ("superposition"); the ReLU and the bias b filter out the resulting crosstalk.
"""

import math

import torch
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, n_features: int, width: int):
        super().__init__()
        # Same init as the reference code: W ~ N(0,1)/sqrt(m), b ~ N(0,1).
        self.W = nn.Parameter(torch.randn(n_features, width) / math.sqrt(width))
        self.b = nn.Parameter(torch.randn(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W @ self.W.T + self.b)
