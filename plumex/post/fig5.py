# %%
import sys
from pathlib import Path
from typing import Any
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from mitosis import _load_trial_params
from mitosis import load_trial_data
from rom_plumes.typing import Float2D
from rom_plumes.typing import Frame
from rom_plumes.typing import PlumePoints

from plumex.plotting import CMAP
from plumex.regression_pipeline import _add_regression_to_plot
from plumex.regression_pipeline import _plot_frame_points
from plumex.regression_pipeline import regress_centerline
from plumex.regression_pipeline import REGRESSION_METHODS
from plumex.video_digest import _load_video
from plumex.video_digest import _plot_frame
from plumex.video_digest import _plot_learn_path


def run():
    frame = Frame(800)
    trials_folder = Path(__file__).absolute().parents[2] / "trials"
    # trail hash result data should ideally line upwith the inputs used by
    # center.py regression step.  But as this is a demonstration of method in
    # application, no validation data is reserved.
    example_trial = "cb6956"
    best_meth = "poly_inv_pin"
    points_trials_folder = trials_folder / "center"

    tdata = load_trial_data(example_trial, trials_folder=points_trials_folder)
    targs = _load_trial_params(
        example_trial, step=0, trials_folder=points_trials_folder
    )
    center = list(filter(lambda row: row[0] == frame, tdata[0]["data"]["center"]))[0][1]
    raw_vid, origin_fc = _load_video(targs["filename"])
    raw_frame = raw_vid[frame]

    def _mini_multireg_center(center: PlumePoints) -> Any:
        meth_coeffs: dict[str, Float2D] = {}
        for method in REGRESSION_METHODS:
            meth_coeffs[method] = regress_centerline(
                {"center": [(frame, center)]},
                r_split=sys.maxsize,
                regression_method=method,
                poly_deg=2,
                display=False,
            )["data"]
        return meth_coeffs

    meth_coeffs = _mini_multireg_center(center)
    best_c = meth_coeffs[best_meth]

    # %%
    fig5 = plt.figure(figsize=(15, 4))
    gs = fig5.add_gridspec(1, 3, width_ratios=[1, 0.69, 1])
    fig5.suptitle("Best regression method for center path", size="x-large")
    no_points = cast(PlumePoints, np.empty(shape=(0, 3)))
    ax0 = fig5.add_subplot(gs[0])
    _plot_learn_path(ax0, raw_frame, center, no_points, no_points)
    # hack to make colors line up with second plot
    ax0.get_lines()[0].set_color(CMAP[0])
    ax0.set_xlabel("(a)")
    ax1 = fig5.add_subplot(gs[1])
    y_max, x_max = raw_frame.shape
    origin_pc = (origin_fc[0], float(y_max - origin_fc[1]))
    _plot_frame_points(ax1, center, y_max, None, origin_fc)
    for meth_ind, (meth, coeffs) in enumerate(meth_coeffs.items()):
        color = CMAP[2 + meth_ind]
        _add_regression_to_plot(
            ax1, coeffs.flatten(), meth, center, y_max, origin_fc, color
        )
    # ax1.set_xlim(ax0.get_xlim())
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect(ax0.get_aspect())
    ax1.set_ylim(ax0.get_ylim()[1], ax0.get_ylim()[0])
    ax1.set_xlabel("(b)")
    ax2 = fig5.add_subplot(gs[2])
    _plot_frame(ax2, raw_frame)
    _add_regression_to_plot(
        ax2, best_c.flatten(), best_meth, center, y_max, origin_fc, CMAP[4]
    )
    ax1.legend()
    # hack because _add_regression_to_plot meant for plot coordinate axis (positive up)
    for line in ax2.get_lines():
        line.set_ydata(y_max - line.get_ydata())
    ax2.set_xlabel("(c)")
    fig5.tight_layout()
    fig5.subplots_adjust(top=0.95)

    # %%


if __name__ == "__main__":
    run()
