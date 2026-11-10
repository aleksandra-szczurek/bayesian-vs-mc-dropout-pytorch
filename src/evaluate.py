"""
Evaluation: point metrics and predictive uncertainty with MC dropout.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Point predictions and metrics
# ---------------------------------------------------------------------------

def predict(model, X):
    """Deterministic prediction: dropout switched OFF (model.eval())."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Uncertainty with MC dropout
# ---------------------------------------------------------------------------

def mc_dropout_predict(model, X, n_samples=500):
    """Run the model n_samples times with dropout ON; returns (n_samples, n_points)."""
    model.train()
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(X_t).numpy())
    return np.stack(preds)


def predictive_interval(samples, sigma_noise, level=0.90, n_noise_draws=20, seed=0):
    """Predictive interval combining MC dropout samples (epistemic) and Gaussian observation noise (aleatoric)."""
    rng = np.random.default_rng(seed)
    noisy = samples[:, None, :] + rng.normal(0, sigma_noise,
                                             size=(samples.shape[0], n_noise_draws, samples.shape[1]))
    noisy = noisy.reshape(-1, samples.shape[1])
    alpha = (1 - level) / 2
    lower = np.quantile(noisy, alpha, axis=0)
    upper = np.quantile(noisy, 1 - alpha, axis=0)
    return lower, upper


def coverage(y_true, lower, upper):
    """Share of observations that fall inside [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
