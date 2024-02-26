import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import pysindy as ps

from .types import TimeData, PolyData

def plot_smoothing_step(
    t: TimeData, data: PolyData, model: ps.SINDy, feature_names: list[str]
) -> Figure:
    fig = plt.figure(figsize=(12, 4))
    for i, feat in enumerate(feature_names):
        ax = fig.add_subplot(1, 3, 1+i)
        ax.plot(t, data[:, i], label=f"{feat} data")
        ax.plot(
            t,
            model.differentiation_method.smoothed_x_[:, i],
            label=f"Smoothed {feat}"
        )
        ax.set_xlabel("Time")
        ax.legend()
    fig.suptitle("Did smoothing work at all?")
    fig.tight_layout()
    return fig


def print_diagnostics(t: TimeData, data: PolyData, model: ps.SINDy) -> None:
    print(rf"Time runs from {t.min()} to {t.max()}, $\delta t$={t[1]-t[0]}", flush=True)
    print("Identified model:")
    model.print(precision=5)
    print(flush=True)