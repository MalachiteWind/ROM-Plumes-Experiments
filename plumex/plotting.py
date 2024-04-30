import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mcolors

import pysindy as ps

from .types import Float1D, PolyData

CMAP = mpl.color_sequences["tab10"]
CMEAS = CMAP[0]
CSMOOTH = CMAP[1]
CEST = CMAP[2]

BGROUND = mcolors.CSS4_COLORS["lightgrey"]

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
        ax.plot(t, data[:, i], label=f"Data", color=CMEAS)
        ax.plot(t, smooth_data[:, i], label=f"Smoothed", color=CSMOOTH)
        ax.set_xticks([])
        ax.set_title(f"Coeff {feat}")
        ax_zoom = zoom_fig.add_subplot(1, 3, 1+i)
        ax_zoom.plot(t, smooth_data[:, i], color=CSMOOTH)
        ax_zoom.set_xlabel("Time")
        ax_fft = fft_fig.add_subplot(1, 3, 1+i)
        freqs = np.fft.rfftfreq(len(t))
        data_psd = np.abs(np.fft.rfft(data[:, i]))**2
        smooth_psd = np.abs(np.fft.rfft(smooth_data[:, i]))**2
        data_pwr = data_psd.sum()
        smooth_pwr = smooth_psd.sum()
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
    t: Float1D, x_dot_est: PolyData, x_dot_pred: PolyData, feature_names: list[str]
) -> Figure:
    bigfig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = GridSpec(2, 3, figure=bigfig)
    plot_fig = bigfig.add_subfigure(gs[0, :])
    scat_fig = bigfig.add_subfigure(gs[1, :])
    for i, feat in enumerate(feature_names):
        ax = plot_fig.add_subplot(1, 3, 1+i)
        ax.plot(t, x_dot_est[:, i], label=fr"Smoothed", color=CSMOOTH)
        ax.plot(t, x_dot_pred[:, i], label=fr"SINDy predicted", color=CEST)
        ax.set_title(f"Coeff {feat} dot")
        ax.set_xlabel("Time")
    bigfig.axes[0].legend()
    plot_fig.suptitle("Time Series Derivative Accuracy")
    for i, feat in enumerate(feature_names):
        ax = scat_fig.add_subplot(1, 3, 1+i)
        ax.axhline(0, color="black")
        ax.axvline(0, color="black")
        ax.scatter(x=x_dot_est[:, i], y=x_dot_pred[:, i])
        ax.set_aspect("equal")
        ax.set_box_aspect(1.0)
        ax.set_xlabel("Smoothed derivative")
        ax.set_ylabel("SINDy predicted derivative")
    scat_fig.suptitle("Are certain regions more error-prone?")
    bigfig.suptitle("Step 2: How well did SINDy fit the (smoothed) data?", size="x-large")
    bigfig.patch.set_facecolor(BGROUND)
    return bigfig


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
    fig = plt.figure(figsize=(12, 4), layout="constrained")
    for i in range(x_true.shape[1]):
        ax = fig.add_subplot(1, x_true.shape[1], 1+i)
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

def plot_hankel_variance(
        S_norm, 
        locs, 
        vars, 
        S_norm_smooth=None,
        locs_smooth=None,
        vars_smooth=None
    ):
    color_pallet = ['r','g','b','c','m','k','w']

    # If ALL smooth params are non-empty, then plot both
    # smooth and non-smoothed data. 
    if (
        isinstance(S_norm_smooth,np.ndarray) and
        isinstance(locs_smooth,list) and
        isinstance(vars_smooth,list)
    ):
        fig, ax = plt.subplots(1,2,figsize=(15,5),layout="constrained")

        # Plot unsmoothed results
        ax[0].set_title("Data")
        ax[0].scatter(range(len(S_norm)),S_norm)
        for i in range(len(vars)):
            loc_i = locs[i]
            var_i = vars[i]
            ax[0].vlines(
                loc_i,
                linestyles='--',
                ymin=0,
                ymax=np.max(S_norm),
                color=color_pallet[i%len(color_pallet)],
                label=fr"{int(var_i*100)} Var ({loc_i + 1}) $\sigma$"
            )
            ax[0].legend(loc='upper right')

        # Plot smoothed results
        ax[1].set_title("Smoothed Data")
        ax[1].scatter(range(len(S_norm_smooth)),S_norm_smooth)
        for i in range(len(vars_smooth)):
            loc_i = locs_smooth[i]
            var_i = vars_smooth[i]
            ax[1].vlines(
                loc_i,
                linestyles='--',
                ymin=0,
                ymax=np.max(S_norm_smooth),
                color=color_pallet[i%len(color_pallet)],
                label=fr"{int(var_i*100)} Var ({loc_i + 1}) $\sigma$"
            )
            ax[1].legend(loc='upper right')
        fig.suptitle("Singular Values of Hankel Matrix",size='x-large')

    # Otherwise print just unsmoothed data
    else:
        fig = plt.figure(figsize=(7,5))
        plt.title("Singular Values of Hankel Matrix")
        plt.scatter(range(len(S_norm)),S_norm)
        for i in range(len(vars)):
            loc_i = locs[i]
            var_i = vars[i]
            plt.vlines(
                loc_i,
                linestyles='--',
                ymin=0,
                ymax=np.max(S_norm),
                color = color_pallet[i%len(color_pallet)],
                label=fr"{int(var_i*100)} Var ({loc_i+1} $\sigma$)"
            )
        plt.legend()

    return fig

def plot_dominate_hankel_modes(
        V,
        num_of_modes, 
        variance,
        V_smooth=None,
        num_of_modes_smooth=None,
        variance_smooth=None):
    if (
        isinstance(V_smooth,np.ndarray) and
        isinstance(num_of_modes_smooth,np.int64) and
        isinstance(variance_smooth,np.float64)
    ):
        fig, ax = plt.subplots(1,2,figsize=(15,5), layout='constrained')
        # plot unsmothed modes
        ax[0].set_title(f"Data ({int(variance*100)}% var) ")
        for i in range(num_of_modes):
            ax[0].plot(V[:,i], label=f"Mode {i}")
        ax[0].legend(loc="upper right")

        # Plot smoothed modes
        ax[1].set_title(f"Smoothed Data ({int(variance_smooth*100)}% var)")
        for i in range(num_of_modes_smooth):
            ax[1].plot(V_smooth[:,i], label=f"Mode {i}")
        ax[1].legend(loc="upper right")

        fig.suptitle("Dominate modes of V and V_smooth")

    else:
        fig = plt.figure(figsize=(7,5))
        plt.title(f"Dominant Modes of V ({int(variance*100)}% var)")
        for i in range(num_of_modes):
            plt.plot(V[:,i], label=f"Mode {i}")
        plt.legend(loc='upper right')
    return fig

def plot_time_series(
        t: Float1D, data: PolyData, feature_names: list[str], smooth_data: PolyData=None
)-> Figure:
    fig, ax = plt.subplots(1,3,figsize=(15,5), layout="constrained")
    for i, feat in enumerate(feature_names):
        ax[i].plot(t, data[:,i], label="Data", color=CMEAS)

        if isinstance(smooth_data,np.ndarray):
            ax[i].plot(t,smooth_data[:,i], label="Smoothed", color=CSMOOTH)

        ax[i].set_title(f"Coeff {feat}")
        if i==0:
            ax[i].legend()
    fig.suptitle("Timeseries",size='x-large')
    # fig.legend()
    plt.tight_layout()
    return fig