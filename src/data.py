"""
Data loading and feature engineering.

Same setup as the R thesis:
- 12 monthly lags of the temperature
- cyclical month encoding (sine and cosine)  -> 14 inputs
- chronological 80/20 split
- standardisation fitted on the training set only (no data leakage)
"""

import numpy as np
import pandas as pd


def load_nottem():
    """Nottingham monthly temperatures 1920-1939 (the R dataset `nottem`).

    Returns
    -------
    y : np.ndarray, shape (240,)   temperatures in degrees Fahrenheit
    months : np.ndarray, shape (240,)   month number 1..12
    """
    from statsmodels.datasets import get_rdataset

    df = get_rdataset("nottem", "datasets").data
    y = df["value"].to_numpy(dtype=float)
    months = (np.arange(len(y)) % 12) + 1  # the series starts in January 1920
    return y, months


def load_csv(path, value_col="temp", month_col="month"):
    """Load a monthly series from a CSV (e.g. the Rome NASA POWER data exported from R).

    The CSV needs one row per month, in chronological order, with a temperature
    column and a month column (1..12).
    """
    df = pd.read_csv(path)
    y = df[value_col].to_numpy(dtype=float)
    months = df[month_col].to_numpy(dtype=int)
    return y, months


def make_features(y, months, n_lags=12):
    """Build the 14 inputs: n_lags past values + sin/cos of the month.

    Row t predicts y[t] from y[t-1], ..., y[t-n_lags] and the month of t.
    """
    X, target = [], []
    for t in range(n_lags, len(y)):
        lags = y[t - n_lags:t][::-1]  # y[t-1], y[t-2], ..., y[t-12]
        angle = 2 * np.pi * months[t] / 12
        X.append(np.concatenate([lags, [np.sin(angle), np.cos(angle)]]))
        target.append(y[t])
    return np.array(X), np.array(target)


def chronological_split(X, y, train_frac=0.8):
    """First 80% of the time series for training, last 20% for testing. No shuffling."""
    n_train = int(len(y) * train_frac)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


class Standardizer:
    """Mean/std scaling fitted on the training data only."""

    def fit(self, a):
        self.mean = a.mean(axis=0)
        self.std = np.where(a.std(axis=0) == 0, 1.0, a.std(axis=0))  # avoid dividing by zero
        return self

    def transform(self, a):
        return (a - self.mean) / self.std

    def inverse_transform(self, a):
        return a * self.std + self.mean
