"""
The neural network: same architecture as the thesis
(feed-forward, 1 hidden layer with 8 neurons), plus dropout.

Dropout is what lets us estimate uncertainty later with MC dropout:
if we keep it switched on at prediction time, every forward pass gives
a slightly different prediction, and their spread measures the model's
(epistemic) uncertainty.
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_inputs=14, n_hidden=8, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.Sigmoid(),          # logistic activation, as in R's nnet
            nn.Dropout(dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
