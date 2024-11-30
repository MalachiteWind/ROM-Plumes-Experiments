import inspect
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypedDict

import numpy as np
from matplotlib import pyplot as plt
from rom_plumes.typing import Float2D
from rom_plumes.typing import GrayVideo
from rom_plumes.typing import PlumePoints

from plumex.regress_edge import create_lin_func
from plumex.regress_edge import create_sin_func
from plumex.regression_pipeline import _construct_rxy_f
from plumex.types import Float1D


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


def _construct_rxy_from_center_fit(
    indep_data: Float1D,
    center_fit_func: Callable[[float, float, float], float],
    regression_method: str,
) -> Float2D:
    """
    Allows for more interpolating points
    """
    rxy = np.array([indep_data] * 3).T
    if regression_method == "poly_para":
        return center_fit_func(rxy)
    else:
        rxy = center_fit_func(rxy)
        r_vals = np.linalg.norm(rxy[:, 1:], axis=1)
        rxy[:, 0] = r_vals
        return rxy


def _visualize_multi_edge_fits(
    video_data: List[RegressionData],
    frame_ids: List[int],
    subtitles: List[str],
    figsize: Tuple[int, int],
    title: Optional[str] = None,
    plot_edge_points: bool = True,
    plot_center_points: bool = True,
    plot_edge_regression: bool = True,
    plot_center_regression: bool = True,
    plot_on_raw_points: bool = True,
):
    axis_fontsize = 15
    title_fontsize = 25

    fig, axes = plt.subplots(
        nrows=len(video_data), ncols=len(frame_ids), figsize=figsize
    )
    try:
        axes = axes.flatten()
    except AttributeError:
        pass
    idx = 0

    for vid in video_data:
        for frame_id in frame_ids:
            try:
                ax = axes[idx]
            except TypeError:
                ax = axes

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

            if plot_edge_points:
                ax.scatter(
                    raw_bot_points[:, 1], raw_bot_points[:, 2], marker=".", c="g"
                )
                ax.scatter(
                    raw_top_points[:, 1], raw_top_points[:, 2], marker=".", c="b"
                )

            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            # ax.axis("off")

            # plot center points

            raw_center_points = vid["center_plume_points"][frame_id][1]
            orig_center_fc = vid["orig_center_fc"]
            center_fit_method = vid["center_func_method"]
            center_fit_func = _construct_rxy_f(
                vid["center_coef"][frame_id], center_fit_method
            )
            raw_center_points[:, 1:] -= orig_center_fc
            if center_fit_method == "poly_para":
                r_min = np.min(raw_center_points[:, 0]) * 1.5
                r_max = np.max(raw_center_points[:, 0]) * 1.5
                r_vals = np.linspace(r_min, r_max, 101)
                fit_centerpoints_dc = _construct_rxy_from_center_fit(
                    r_vals, center_fit_func, center_fit_method
                )
            else:
                x_min = np.min(raw_center_points[:, 1]) * 2
                x_max = np.max(raw_center_points[:, 1]) * 1.5
                x_vals = np.linspace(x_min, x_max, 101)
                fit_centerpoints_dc = _construct_rxy_from_center_fit(
                    x_vals, center_fit_func, center_fit_method
                )
            raw_center_points[:, 1:] += orig_center_fc
            fit_centerpoints_dc[:, 1:] += orig_center_fc

            if plot_center_points:
                ax.scatter(
                    raw_center_points[:, 1], raw_center_points[:, 2], marker=".", c="r"
                )

            if plot_center_regression:
                ax.plot(fit_centerpoints_dc[:, 1], fit_centerpoints_dc[:, 2], c="r")

            if plot_on_raw_points:
                anchor_points = raw_center_points
            else:
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
            if plot_edge_regression:
                ax.plot(top_points[:, 0], top_points[:, 1], c="g")
                ax.plot(bot_points[:, 0], bot_points[:, 1], c="b")
            ax.set_xlim([0, x_lim])
            ax.set_ylim([y_lim, 0])

            idx += 1
    if title:
        fig.suptitle(title, fontsize=title_fontsize)
    fig.tight_layout(pad=0.5)
    return fig


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
