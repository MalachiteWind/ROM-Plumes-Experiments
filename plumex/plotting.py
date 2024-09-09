from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .types import Float1D
from .types import Float2D
from .types import PolyData

CMAP = mpl.color_sequences["tab10"]
CMEAS = CMAP[0]
CSMOOTH = CMAP[1]
CEST = CMAP[2]

BGROUND = mcolors.CSS4_COLORS["lightgrey"]


def plot_smoothing_step(
    t: Float1D, data: Float2D, smooth_data: Float2D, feature_names: list[str]
) -> Figure:
    n_feat = len(feature_names)
    bigfig = plt.figure(figsize=(12, 12), layout="constrained")
    gs = GridSpec(3, n_feat, figure=bigfig)
    compare_fig = bigfig.add_subfigure(gs[0, :])
    zoom_fig = bigfig.add_subfigure(gs[1, :])
    fft_fig = bigfig.add_subfigure(gs[2, :])
    for i, feat in enumerate(feature_names):
        ax = compare_fig.add_subplot(1, n_feat, 1 + i)
        ax.plot(t, data[:, i], label="Data", color=CMEAS)
        ax.plot(t, smooth_data[:, i], label="Smoothed", color=CSMOOTH)
        ax.set_xticks([])
        ax.set_title(f"Coeff {feat}")
        ax_zoom = zoom_fig.add_subplot(1, n_feat, 1 + i)
        ax_zoom.plot(t, smooth_data[:, i], color=CSMOOTH)
        ax_zoom.set_xlabel("Time")
        ax_fft = fft_fig.add_subplot(1, n_feat, 1 + i)
        freqs = np.fft.rfftfreq(len(t))
        data_psd = np.abs(np.fft.rfft(data[:, i])) ** 2
        smooth_psd = np.abs(np.fft.rfft(smooth_data[:, i])) ** 2
        ax_fft.plot(freqs[1:], data_psd[1:], color=CMEAS)
        ax_fft.plot(freqs[1:], smooth_psd[1:], color=CSMOOTH)
        ax_fft.set_yscale("log")
        ax_fft.set_xlabel("Frequency (Hz)")
    bigfig.suptitle("Step 1: How effective is Smoothing/Denoising?", size="x-large")
    compare_fig.suptitle("Compare (normalized) measurements and smoothed trajectories")
    zoom_fig.suptitle("Does the smoothed trajectory look smooth?")
    fft_fig.suptitle("Compare PSD of (normalized) and smoothed measurements")
    bigfig.axes[0].legend()
    bigfig.patch.set_facecolor(BGROUND)
    return bigfig


def plot_predictions(
    t: Float1D, x_dot_est: Float2D, x_dot_pred: Float2D, feature_names: list[str]
) -> Figure:
    n_feat = len(feature_names)
    bigfig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = GridSpec(2, n_feat, figure=bigfig)
    plot_fig = bigfig.add_subfigure(gs[0, :])
    scat_fig = bigfig.add_subfigure(gs[1, :])
    for i, feat in enumerate(feature_names):
        ax = plot_fig.add_subplot(1, n_feat, 1 + i)
        ax.plot(t, x_dot_est[:, i], label=r"Smoothed", color=CSMOOTH)
        ax.plot(t, x_dot_pred[:, i], label=r"SINDy predicted", color=CEST)
        ax.set_title(f"Coeff {feat} dot")
        ax.set_xlabel("Time")
    bigfig.axes[0].legend()
    plot_fig.suptitle("Time Series Derivative Accuracy")
    for i, feat in enumerate(feature_names):
        ax = scat_fig.add_subplot(1, n_feat, 1 + i)
        ax.axhline(0, color="black")
        ax.axvline(0, color="black")
        ax.scatter(x=x_dot_est[:, i], y=x_dot_pred[:, i])
        ax.set_aspect("equal")
        ax.set_box_aspect(1.0)
        ax.set_xlabel("Smoothed derivative")
        ax.set_ylabel("SINDy predicted derivative")
    scat_fig.suptitle("Are certain regions more error-prone?")
    bigfig.suptitle(
        "Step 2: How well did SINDy fit the (smoothed) data?", size="x-large"
    )
    bigfig.patch.set_facecolor(BGROUND)
    return bigfig


def print_diagnostics(t: Float1D, model: ps.SINDy, precision: int) -> None:
    print(rf"Time runs from {t.min()} to {t.max()}, Δt={t[1]-t[0]}", flush=True)
    print("Identified model:")
    model.print(precision=precision)
    print(flush=True)


def plot_simulation(
    t: Float1D, x_true: Float2D, x_sim: Float2D, *, feat_names: list[str], title: str
) -> None:
    """Plot the true vs simulated data"""
    n_feat = len(feat_names)
    m = min(x_true.shape[0], x_sim.shape[0])
    fig = plt.figure(figsize=(12, 4), layout="constrained")
    for i in range(n_feat):
        ax = fig.add_subplot(1, n_feat, 1 + i)
        ax.plot(t[:m], x_true[:m, i], "k", label="smoothed data", color=CSMOOTH)
        ax.plot(t[:m], x_sim[:m, i], "r--", label="model simulation", color=CEST)
        ax.set(xlabel="t")
        ax.set_title("Coeff {}".format(feat_names[i]))

    first_ax = fig.axes[0]
    first_ax.legend()
    fig.suptitle(
        f"Step 3: How well does SINDy simulate the system? ({title})", size="x-large"
    )
    fig.patch.set_facecolor(BGROUND)


# Plotting for Hankel Analysis
def plot_time_series(
    t: Float1D,
    data: PolyData,
    feature_names: list[str],
    smooth_data: Optional[PolyData] = None,
) -> Figure:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for i, feat in enumerate(feature_names):
        ax[i].plot(t, data[:, i], label="Data", color=CMEAS)

        if smooth_data is not None:
            ax[i].plot(t, smooth_data[:, i], label="Smoothed", color=CSMOOTH)

        ax[i].set_title(f"Coeff {feat}")
        if i == 0:
            ax[i].legend()
    fig.suptitle("Timeseries", size="x-large")
    # fig.legend()
    plt.tight_layout()
    return fig


def plot_hankel_variance(
    S_norm: Float1D,
    locs: list[int],
    variances: list[float],
    S_norm_smooth: Optional[Float1D] = None,
    locs_smooth: Optional[list[int]] = None,
    variances_smooth: Optional[list[float]] = None,
) -> Figure:

    if (S_norm_smooth is not None) and locs_smooth and variances_smooth:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")

        # Plot unsmoothed results
        draw_singval_plot(S_norm, locs, variances, ax[0])
        ax[0].set_title("Data")

        # Plot smoothed results
        draw_singval_plot(S_norm_smooth, locs_smooth, variances_smooth, ax[1])
        ax[1].set_title("Smoothed Data")
        fig.suptitle("Singular Values of Hankel Matrix", size="x-large")

    else:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.gca()
        draw_singval_plot(S_norm, locs, variances, ax)
        plt.title("Singular Values of Hankel Matrix")

    return fig


def draw_singval_plot(
    singvals: Float1D, locs: list[int], variance: list[float], ax: Axes
) -> None:
    t = range(len(singvals))
    index = locs[-1] + 1
    c1 = CMAP[(len(locs) - 1) % len(CMAP)]
    c2 = "dimgray"
    ax.scatter(t[:index], singvals[:index], c=c1)
    ax.scatter(t[index:], singvals[index:], c=c2)
    for i in range(len(variance)):
        loc_i = locs[i]
        var_i = variance[i]
        ax.vlines(
            loc_i,
            linestyles="--",
            ymin=0,
            ymax=np.max(singvals),
            color=CMAP[i % len(CMAP)],
            label=rf"{int(var_i*100)} Var ({loc_i + 1} $\sigma$)",
        )
        ax.legend(loc="upper right")


def plot_dominant_hankel_modes(
    V: np.ndarray,
    mode_indices: np.int64,
    variance: np.float64,
    V_smooth: Optional[np.ndarray] = None,
    mode_indices_smooth: Optional[np.int64] = None,
    variance_smooth: Optional[np.float64] = None,
) -> Figure:
    if (V_smooth is not None) and mode_indices_smooth and variance_smooth:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
        # plot unsmothed modes
        ax[0].set_title(f"Data ({int(variance*100)}% var) ")
        for i in range(mode_indices):
            ax[0].plot(V[:, i], label=f"Mode {i}")
        ax[0].legend(loc="upper right")

        # Plot smoothed modes
        ax[1].set_title(f"Smoothed Data ({int(variance_smooth*100)}% var)")
        for i in range(mode_indices_smooth):
            ax[1].plot(V_smooth[:, i], label=f"Mode {i}")
        ax[1].legend(loc="upper right")

        fig.suptitle("Dominate modes of V and V_smooth", size="x-large")

    else:
        fig = plt.figure(figsize=(7, 5))
        plt.title(f"Dominant Modes of V ({int(variance*100)}% var)")
        for i in range(mode_indices):
            plt.plot(V[:, i], label=f"Mode {i}")
        plt.legend(loc="upper right")
    return fig


def plot_data_and_dmd(
    t,
    data,
    dmd_data,
    var,
    svd_rank,
    feature_names,
    smooth_data=None,
    smooth_dmd_data=None,
    var_smooth: float = 0.0,
    svd_rank_smooth=None,
):
    if (smooth_data is not None) and (smooth_dmd_data is not None):
        fig, ax = plt.subplots(3, 2, figsize=(15, 10), layout="constrained")
        for i, feat in enumerate(feature_names):
            ax[i][0].plot(t, data[:, i], label="Data", color=CMEAS)
            ax[i][0].plot(t, dmd_data[:, i], label="DMD", color="k")
            ax[i][0].set_ylabel(f"Coeff {feat}")

            ax[i][1].plot(t, smooth_data[:, i], label="Smooth Data", color=CSMOOTH)
            ax[i][1].plot(t, smooth_dmd_data[:, i], label="DMD", color="k")

            if i == 0:
                ax[i][0].set_title(f"Data ({int(100*var)}% var, svd_rank={svd_rank})")
                ax[i][0].legend()
                ax[i][1].set_title(
                    f"Smoothed Data ({int(100*var_smooth)}% var,"
                    + f"svd_rank={svd_rank_smooth})"
                )
                ax[i][1].legend()
    else:
        fig, ax = plt.subplots(3, 1, figsize=(7, 5), layout="constrained")
        for i, feat in enumerate(feature_names):
            ax[i].plot(t, data[:, i], label="Data", color=CMEAS)
            ax[i].plot(t, dmd_data[:, i], label="DMD", color="k")
            ax[i].set_ylabel(f"Coeff {feat}")

            if i == 0:
                ax[i].legend()

    fig.suptitle("Exact DMD Reconstruction", size="x-large")
    plt.tight_layout()
    return fig
