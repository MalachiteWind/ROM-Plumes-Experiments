from typing import Callable

import numpy as np
from ara_plumes.typing import Float2D
from ara_plumes.typing import GrayImage
from ara_plumes.typing import GrayVideo
from ara_plumes.typing import PlumePoints
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from plumex.regress_edge import create_lin_func
from plumex.regress_edge import create_sin_func
from plumex.regression_pipeline import _construct_rxy_f


def apply_theta_shift(
    t,
    r,
    x,
    y,
    flat_regress_func: Callable[[float, float], float],
    positive: bool = True,
) -> tuple[float, float]:
    d = flat_regress_func(t, r)
    theta_1 = np.arctan2(d, r)
    theta_2 = np.arctan2(y, x)

    def pos_sign(positive: bool):
        if positive:
            return 1
        return -1

    theta = theta_2 + pos_sign(positive) * theta_1

    return r * np.cos(theta), r * np.sin(theta)


def create_edge_func(coeffs: tuple, method: str) -> Callable[[tuple, tuple], tuple]:
    if method == "linear":
        return create_lin_func(coeffs)
    if method == "sinusoid":
        return create_sin_func(coeffs)
    else:
        raise ValueError(f"{method} is not an accepted method.")


def _in_frame(rxy_points: Float2D, frame: GrayImage, orig_center_fc) -> Float2D:
    y_range, x_range = frame.shape
    mask = rxy_points[:, 1:] >= 0
    mask = mask[:, 0] & mask[:, 1]
    less_than_x = rxy_points[:, 1] <= x_range + orig_center_fc[0]
    mask = mask & less_than_x
    less_than_y = rxy_points[:, 2] <= y_range + orig_center_fc[1]
    mask = mask & less_than_y

    return rxy_points[mask]


def _visualize_fits(
    video: GrayVideo,
    n_frames: int,
    center_coef: Float2D,
    center_func_method: str,
    center_plume_points: PlumePoints,
    bottom_plume_points: PlumePoints,
    top_plume_points: PlumePoints,
    top_edge_func: Callable[[float], float],
    bot_edge_func: Callable[[float, float], float],
    orig_center_fc: tuple[float, float],
    start_frame: int = 0,
) -> Figure:
    """
    plot center regression and unflattened edge regression on frames.
    """

    frameskip = len(video) / n_frames

    frame_ids = [int(frameskip * i) for i in range(n_frames)]
    grid_size = int(np.ceil(np.sqrt(n_frames)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(frame_ids):
        ax = axes[i]
        frame_t = video[idx]
        ax.imshow(frame_t, cmap="gray")
        ax.set_title(f"frame {idx+start_frame}")
        ax.set_xticks([])
        ax.set_yticks([])

        center_fit_func = _construct_rxy_f(center_coef[idx], center_func_method)
        # interpolate points?
        raw_center_points = center_plume_points[idx][1]
        raw_bot_points = bottom_plume_points[idx][1]
        raw_top_points = top_plume_points[idx][1]

        ax.scatter(raw_center_points[:, 1], raw_center_points[:, 2], marker=".", c="r")
        ax.scatter(raw_bot_points[:, 1], raw_bot_points[:, 2], marker=".", c="g")
        ax.scatter(raw_top_points[:, 1], raw_top_points[:, 2], marker=".", c="b")

        raw_center_points[:, 1:] -= orig_center_fc

        fit_centerpoints_dc = center_fit_func(raw_center_points)
        fit_centerpoints_dc[:, 1:] += orig_center_fc

        fit_center_points_in_frame = _in_frame(
            fit_centerpoints_dc, frame_t, orig_center_fc
        )
        ax.plot(
            fit_center_points_in_frame[:, 1], fit_center_points_in_frame[:, 2], c="r"
        )
        time = start_frame + idx
        top_points = []
        bot_points = []
        for rad, x_fc, y_fc in raw_center_points:
            top_points.append(
                apply_theta_shift(
                    time,
                    rad,
                    x_fc - orig_center_fc[0],
                    y_fc - orig_center_fc[1],
                    top_edge_func,
                    positive=True,
                )
            )
            bot_points.append(
                apply_theta_shift(
                    time,
                    rad,
                    x_fc - orig_center_fc[0],
                    y_fc - orig_center_fc[1],
                    bot_edge_func,
                    positive=False,
                )
            )
        top_points = np.array(top_points) + orig_center_fc
        bot_points = np.array(bot_points) + orig_center_fc
        ax.plot(top_points[:, 0], top_points[:, 1], c="g")
        ax.plot(bot_points[:, 0], bot_points[:, 1], c="b")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Video Frames", fontsize=16)

    # Tighter layout settings
    fig.tight_layout(pad=0.5)  # Reduce padding around the plot elements
    plt.subplots_adjust(
        left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1, wspace=0.1
    )
    return fig
