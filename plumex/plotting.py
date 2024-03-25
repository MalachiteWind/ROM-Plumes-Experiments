import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

import pysindy as ps

from .types import Float1D, PolyData

CMAP = mpl.color_sequences["tab10"]
CMEAS = CMAP[0]
CEST = CMAP[1]

def plot_smoothing_step(
    t: Float1D, data: PolyData, smooth_data: PolyData, feature_names: list[str]
) -> Figure:
    bigfig = plt.figure(figsize=(12, 12), layout="constrained")
    gs = GridSpec(3, 3, figure=bigfig)
    compare_fig = bigfig.add_subfigure(gs[0, :])
    zoom_fig = bigfig.add_subfigure(gs[1, :])
    fft_fig = bigfig.add_subfigure(gs[2, :])
    for i, feat in enumerate(feature_names):
        ax = compare_fig.add_subplot(1, 3, 1+i)
        ax.plot(t, data[:, i], label=f"Data: {feat}", color=CMEAS)
        ax.plot(t, smooth_data[:, i], label=f"Smoothed: {feat}", color=CEST)
        ax.set_xticks([])
        ax_zoom = zoom_fig.add_subplot(1, 3, 1+i)
        ax_zoom.plot(t, smooth_data[:, i], color=CEST)
        ax_zoom.set_xlabel("Time")
        ax_fft = fft_fig.add_subplot(1, 3, 1+i)
        freqs = np.fft.rfftfreq(2000)
        data_psd = np.abs(np.fft.rfft(data[:, i]))**2
        smooth_psd = np.abs(np.fft.rfft(smooth_data[:, i]))**2
        data_pwr = data_psd.sum()
        smooth_pwr = smooth_psd.sum()
        ax_fft.plot(freqs[1:], data_psd[1:], color=CMEAS)
        ax_fft.plot(freqs[1:], smooth_psd[1:], color=CEST)
        ax_fft.set_yscale("log")
        ax_fft.set_xlabel("Frequency (Hz)")
    bigfig.suptitle("Step 1: How effective is Smoothing/Denoising?", size="large")
    compare_fig.suptitle("Compare (normalized) measurements and smoothed trajectories")
    zoom_fig.suptitle("Does the smoothed trajectory look smooth?")
    fft_fig.suptitle("Compare PSD of (normalized) and smoothed measurements")
    bigfig.axes[0].legend()
    return bigfig


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
    print(rf"Time runs from {t.min()} to {t.max()}, Δt={t[1]-t[0]}", flush=True)
    print("Identified model:")
    model.print(precision=precision)
    print(flush=True)


def plot_simulation(
    t: Float1D, x_true: PolyData, x_sim: PolyData, *, feat_names: list[str], title: str
) -> None:
    """Plot the true vs simulated data"""
    m = min(x_true.shape[0], x_sim.shape[0])
    fig, axs = plt.subplots(
        1,
        x_true.shape[1],
        sharex=True,
        figsize=(15, 4)
    )
    for i, ax in enumerate(axs):
        ax.plot(t[:m], x_true[:m, i], "k", label="true normalized data")
        ax.plot(t[:m], x_sim[:m, i], "r--", label="model simulation")
        ax.set(xlabel="t")
        ax.set_title("Coeff {}".format(feat_names[i]))

    last_ax = fig.axes[-1]
    last_ax.legend()
    fig.suptitle(title)
