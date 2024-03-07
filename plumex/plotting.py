import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import pysindy as ps

from .types import Float1D, PolyData

def plot_smoothing_step(
    t: Float1D, data: PolyData, smooth_data: PolyData, feature_names: list[str]
) -> Figure:
    fig = plt.figure(figsize=(12, 4))
    for i, feat in enumerate(feature_names):
        ax = fig.add_subplot(1, 3, 1+i)
        ax.plot(t, data[:, i], label=f"Data: {feat}")
        ax.plot(t, smooth_data[:, i], label=f"Smoothed: {feat}")
        ax.set_xlabel("Time")
        ax.legend()
    fig.suptitle("Smoothing")
    fig.tight_layout()
    return fig


def plot_predictions(
    t: Float1D, x_dot_est: PolyData, x_dot_pred: PolyData, feature_names: list[str]
) -> tuple[Figure, Figure]:
    fig1 = plt.figure(figsize=(12, 4))
    for i, feat in enumerate(feature_names):
        ax = fig1.add_subplot(1, 3, 1+i)
        ax.plot(t, x_dot_est[:, i], label=fr"$Smoothed: \dot {feat}$")
        ax.plot(t, x_dot_pred[:, i], label=fr"$SINDy: \dot {feat}$")
        ax.set_xlabel("Time")
        ax.legend()
    fig1.suptitle("Predictions")
    fig1.tight_layout()
    fig2 = plt.figure(figsize=(12, 4))
    for i, feat in enumerate(feature_names):
        ax = fig2.add_subplot(1, 3, 1+i)
        ax.scatter(x=x_dot_est[:, i], y=x_dot_pred[:, i])
        ax.set_xlabel("estimated-smoothing")
        ax.set_ylabel("estimated-SINDy")
        ax.set_title(f"{feat}")
    fig2.suptitle("Predictions")
    fig2.tight_layout()
    return fig1, fig2


def print_diagnostics(t: Float1D, model: ps.SINDy, precision: int) -> None:
    print(rf"Time runs from {t.min()} to {t.max()}, $\delta t$={t[1]-t[0]}", flush=True)
    print("Identified model:")
    model.print(precision=precision)
    print(flush=True)