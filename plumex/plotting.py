import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import pysindy as ps

from .types import TimeData, PolyData

def plot_smoothing_step(
    t: TimeData, data: PolyData, model: ps.SINDy, feature_names: list[str]
) -> Figure:
    fig, ax = plt.subplots(1, 1)
    for i, feat in enumerate(feature_names):
        ax.plot(t, data[:, i], label=f"{feat} data")
        ax.plot(
            t,
            model.differentiation_method.smoothed_x_[:, i],
            label=f"Smoothed {feat}"
        )
    ax.set_xlabel("Time")
    ax.set_title("Did smoothing work at all?")
    ax.legend()
    return fig


def print_diagnostics(t: TimeData, data: PolyData, model: ps.SINDy) -> None:
    print(rf"Time runs from {t.min()} to {t.max()}, $\delta t$={t[1]-t[0]}", flush=True)
    print("Identified model:")
    model.print()
    print(flush=True)