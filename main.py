"""
Bayesian vs frequentist neural networks - Python replication with PyTorch.

Run:  python main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_nottem, load_csv, make_features, chronological_split, Standardizer
from src.model import MLP
from src.train import set_seed, train_model
from src.evaluate import (
    predict, rmse, mae,
    mc_dropout_predict, predictive_interval, coverage,
)

# Results of the Bayesian network from the R thesis (Stan, HMC/NUTS), for comparison
THESIS_BNN = {
    "Nottingham": {"RMSE": 2.14, "MAE": 1.74, "coverage_90": 0.913, "width": 7.51, "unit": "°F"},
    "Rome":       {"RMSE": 1.46, "MAE": 1.09, "coverage_90": 0.847, "width": 4.27, "unit": "°C"},
}

ROME_START = "1990-01-01"   # first month of the Rome series


def run(name, y, months, start_date):
    set_seed(42)
    thesis = THESIS_BNN[name]
    unit = thesis["unit"]

    # data
    X, target = make_features(y, months)
    X_tr, X_te, y_tr, y_te = chronological_split(X, target)
    x_scaler = Standardizer().fit(X_tr)
    y_scaler = Standardizer().fit(y_tr)
    X_tr_s, X_te_s = x_scaler.transform(X_tr), x_scaler.transform(X_te)
    y_tr_s = y_scaler.transform(y_tr)
    print(f"\n===== {name} =====")
    print(f"Train: {len(y_tr)} months, test: {len(y_te)} months, inputs: {X.shape[1]}")

    # model
    model = MLP(n_inputs=X.shape[1], n_hidden=8, dropout=0.1)
    train_model(model, X_tr_s, y_tr_s, verbose=False)
    pred_te = y_scaler.inverse_transform(predict(model, X_te_s))

    # uncertainty
    pred_tr = y_scaler.inverse_transform(predict(model, X_tr_s))
    sigma_noise = np.std(y_tr - pred_tr)
    samples = y_scaler.inverse_transform(mc_dropout_predict(model, X_te_s, n_samples=500))
    lower, upper = predictive_interval(samples, sigma_noise)

    print(f"  PyTorch + MC dropout  RMSE = {rmse(y_te, pred_te):.2f}  MAE = {mae(y_te, pred_te):.2f}  "
          f"coverage = {coverage(y_te, lower, upper):.1%}  width = {np.mean(upper - lower):.2f} {unit}")
    print(f"  Thesis BNN (Stan)     RMSE = {thesis['RMSE']:.2f}  MAE = {thesis['MAE']:.2f}  "
          f"coverage = {thesis['coverage_90']:.1%}  width = {thesis['width']:.2f} {unit}")

    # plot
    dates = pd.date_range(start_date, periods=len(y), freq="MS")
    test_dates = dates[-len(y_te):]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(test_dates, lower, upper, alpha=0.25, label="90% interval (MC dropout)")
    ax.plot(test_dates, pred_te, lw=2, label="Forecast (PyTorch MLP)")
    ax.plot(test_dates, y_te, "o", ms=4, color="black", label="Observed")
    ax.set_title(f"{name} monthly temperature – test set")
    ax.set_ylabel(f"Temperature ({unit})")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"results/forecast_{name.lower()}.png"
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"  Plot saved to {path}")


def main():
    y, months = load_nottem()
    run("Nottingham", y, months, "1920-01-01")

    y, months = load_csv("data/rome.csv", value_col="temp", month_col="month")
    run("Rome", y, months, ROME_START)


if __name__ == "__main__":
    main()