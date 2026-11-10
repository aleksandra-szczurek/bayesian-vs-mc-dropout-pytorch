"""
Training loop.

weight_decay plays the same role as `decay` in R's nnet (L2 penalty on the weights).
"""

import numpy as np
import torch
from torch import nn


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, X_train, y_train, epochs=2000, lr=0.01, weight_decay=1e-3, verbose=True):
    """Full-batch training with Adam and mean squared error.

    X_train, y_train: standardised numpy arrays.
    Returns the list of training losses (one per epoch), useful for a plot.
    """
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose and (epoch + 1) % 500 == 0:
            print(f"  epoch {epoch + 1:5d}  train MSE (standardised) = {loss.item():.4f}")

    return losses
