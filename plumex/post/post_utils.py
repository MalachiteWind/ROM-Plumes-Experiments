from typing import Callable
from typing import List
from typing import Tuple
from typing import TypedDict

import numpy as np
from ara_plumes.typing import Float1D
from ara_plumes.typing import Float2D
from ara_plumes.typing import GrayImage
from ara_plumes.typing import GrayVideo
from ara_plumes.typing import PlumePoints
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from plumex.regress_edge import create_lin_func
from plumex.regress_edge import create_sin_func
from plumex.regression_pipeline import _construct_rxy_f


class RegressionData(TypedDict):
    video: GrayVideo
    center_coef: Float2D
    center_func_method: str
    center_plume_points: PlumePoints
    top_plume_points: PlumePoints
    bottom_plume_points: PlumePoints
    top_edge_func: Callable[[float], float]
    bot_edge_func: Callable[[float], float]
    start_frame: int
    orig_center_fc: tuple[float, float]


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
    plot_on_raw_points: bool = True,
    plot_center_points: bool = True,
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
        y_lim, x_lim = frame_t.shape
        ax.imshow(frame_t, cmap="gray")
        ax.set_title(f"frame {idx+start_frame}")
        ax.set_xticks([])
        ax.set_yticks([])

        center_fit_func = _construct_rxy_f(center_coef[idx], center_func_method)
        # interpolate points?
        raw_center_points = center_plume_points[idx][1]
        raw_bot_points = bottom_plume_points[idx][1]
        raw_top_points = top_plume_points[idx][1]

        if plot_center_points:
            ax.scatter(
                raw_center_points[:, 1], raw_center_points[:, 2], marker=".", c="r"
            )
        ax.scatter(raw_bot_points[:, 1], raw_bot_points[:, 2], marker=".", c="g")
        ax.scatter(raw_top_points[:, 1], raw_top_points[:, 2], marker=".", c="b")

        raw_center_points[:, 1:] -= orig_center_fc

        fit_centerpoints_dc = center_fit_func(raw_center_points)
        fit_centerpoints_dc[:, 1:] += orig_center_fc

        raw_center_points[:, 1:] += orig_center_fc
        if plot_center_points:
            ax.plot(fit_centerpoints_dc[:, 1], fit_centerpoints_dc[:, 2], c="r")
        time = start_frame + idx
        top_points = []
        bot_points = []

        flag = 1
        if plot_on_raw_points:
            anchor_points = raw_center_points
        else:
            anchor_points = fit_centerpoints_dc
        for rad, x_fc, y_fc in anchor_points:
            top_points.append(
                apply_theta_shift(
                    time,
                    rad,
                    x_fc - orig_center_fc[0] * flag,
                    y_fc - orig_center_fc[1] * flag,
                    top_edge_func,
                    positive=True,
                )
            )
            bot_points.append(
                apply_theta_shift(
                    time,
                    rad,
                    x_fc - orig_center_fc[0] * flag,
                    y_fc - orig_center_fc[1] * flag,
                    bot_edge_func,
                    positive=False,
                )
            )
        top_points = np.array(top_points) + orig_center_fc
        bot_points = np.array(bot_points) + orig_center_fc

        ax.plot(top_points[:, 0], top_points[:, 1], c="g")
        ax.plot(bot_points[:, 0], bot_points[:, 1], c="b")
        ax.set_xlim([0, x_lim])
        ax.set_ylim([y_lim, 0])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Video Frames", fontsize=16)

    # Tighter layout settings
    fig.tight_layout(pad=0.5)  # Reduce padding around the plot elements
    plt.subplots_adjust(
        left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1, wspace=0.1
    )


def plot_raw_frames(video, n_frames, n_rows, n_cols):
    # Calculate the number of frames to skip
    frameskip = len(video) / n_frames

    # Generate frame indices
    frame_ids = [int(frameskip * i) for i in range(n_frames)]

    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axs = axs.flatten()  # Flatten to easily index the axes

    for i, idx in enumerate(frame_ids):
        frame_t = video[idx]
        axs[i].imshow(frame_t)  # Display the frame
        axs[i].axis("off")  # Turn off axis
        axs[i].set_title(f"t = {idx}", fontsize=20)  # Set title with time point

    # Hide any remaining empty subplots if n_frames < n_rows * n_cols
    for j in range(i + 1, n_rows * n_cols):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()
    return


def _visualize_multi_edge_fits(
    video_data: List[RegressionData],
    frame_ids: List[int],
    title: str,
    subtitles: List[str],
    figsize: Tuple[int, int],
    plot_on_raw_points: bool = True,
):
    axis_fontsize = 15
    title_fontsize = 25

    fig, axes = plt.subplots(
        nrows=len(video_data), ncols=len(frame_ids), figsize=figsize
    )
    axes = axes.flatten()
    idx = 0
    for vid in video_data:
        for frame_id in frame_ids:
            ax = axes[idx]
            frame_t = vid["video"][frame_id]
            y_lim, x_lim = frame_t.shape
            # plot frame
            ax.imshow(frame_t, cmap="gray")
            if idx < len(frame_ids):
                ax.set_title(f"frame {frame_id}", fontsize=axis_fontsize)
            if idx % len(frame_ids) == 0:
                ax.set_ylabel(subtitles[idx // len(frame_ids)], fontsize=axis_fontsize)

            # plot edge points
            raw_bot_points = vid["bottom_plume_points"][frame_id][1]
            raw_top_points = vid["top_plume_points"][frame_id][1]

            ax.scatter(raw_bot_points[:, 1], raw_bot_points[:, 2], marker=".", c="g")
            ax.scatter(raw_top_points[:, 1], raw_top_points[:, 2], marker=".", c="b")
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            # ax.axis("off")

            # plot center points
            raw_center_points = vid["center_plume_points"][frame_id][1]
            orig_center_fc = vid["orig_center_fc"]
            if plot_on_raw_points:
                anchor_points = raw_center_points
            else:
                center_fit_method = vid["center_func_method"]
                center_fit_func = _construct_rxy_f(
                    vid["center_coef"][frame_id], center_fit_method
                )
                raw_center_points[:, 1:] -= orig_center_fc
                fit_centerpoints_dc = center_fit_func(raw_center_points)
                raw_center_points[:, 1:] += orig_center_fc
                fit_centerpoints_dc[:, 1:] += orig_center_fc
                anchor_points = fit_centerpoints_dc
            top_points = []
            bot_points = []
            for rad, x_fc, y_fc in anchor_points:
                top_points.append(
                    apply_theta_shift(
                        frame_id,
                        rad,
                        x_fc - orig_center_fc[0],
                        y_fc - orig_center_fc[1],
                        vid["top_edge_func"],
                        positive=True,
                    )
                )
                bot_points.append(
                    apply_theta_shift(
                        frame_id,
                        rad,
                        x_fc - orig_center_fc[0],
                        y_fc - orig_center_fc[1],
                        vid["bot_edge_func"],
                        positive=False,
                    )
                )

            top_points = np.array(top_points) + orig_center_fc
            bot_points = np.array(bot_points) + orig_center_fc

            ax.plot(top_points[:, 0], top_points[:, 1], c="g")
            ax.plot(bot_points[:, 0], bot_points[:, 1], c="b")
            ax.set_xlim([0, x_lim])
            ax.set_ylim([y_lim, 0])

            idx += 1
    fig.suptitle(title, fontsize=title_fontsize)
    fig.tight_layout(pad=0.5)
    return
