# Additional figures and plots for AGU24
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from plumex.post.edge_figs import _unpack_data
from plumex.post.edge_figs import trial_lookup_key
from plumex.post.post_utils import _construct_rxy_from_center_fit
from plumex.post.post_utils import apply_theta_shift
from plumex.post.post_utils import RegressionData
from plumex.regression_pipeline import _construct_rxy_f


def run():
    # med 914
    ...


video_data = _unpack_data(*trial_lookup_key["med 914"]["default"])


def _create_frames(
    video_data: RegressionData,
    frames: List[int],
) -> List[Figure]:
    rom_frames = []

    for idx in frames:
        # Create a new figure
        fig, ax = plt.subplots()

        frame_t = video_data["video"][idx]
        y_lim, x_lim = frame_t.shape

        # Display the video frame
        ax.imshow(frame_t, cmap="gray")

        # Plot center points and fit
        raw_center_points = video_data["center_plume_points"][idx][1]
        center_fit_method = video_data["center_func_method"]
        orig_center_fc = video_data["orig_center_fc"]
        center_fit_func = _construct_rxy_f(
            coef=video_data["center_coef"][idx], regression_method=center_fit_method
        )
        raw_center_points[:, 1:] -= orig_center_fc
        if center_fit_method == "poly_para":
            r_min = np.min(raw_center_points[:, 0]) * 1.5
            r_max = np.max(raw_center_points[:, 0]) * 1.5
            r_vals = np.linspace(r_min, r_max, 101)
            fit_centerpoints_dc = _construct_rxy_from_center_fit(
                indep_data=r_vals,
                center_fit_func=center_fit_func,
                regression_method=center_fit_method,
            )
        else:
            x_min = np.min(raw_center_points[:, 1]) * 2
            x_max = np.max(raw_center_points[:, 1]) * 1.5
            x_vals = np.linspace(x_min, x_max, 101)
            fit_centerpoints_dc = _construct_rxy_from_center_fit(
                indep_data=x_vals,
                center_fit_func=center_fit_func,
                regression_method=center_fit_method,
            )
        raw_center_points[:, 1:] += orig_center_fc
        fit_centerpoints_dc[:, 1:] += orig_center_fc

        # Plot the fitted center points
        ax.plot(fit_centerpoints_dc[:, 1], fit_centerpoints_dc[:, 2], c="r")

        # Compute and plot top and bottom points
        anchor_points = fit_centerpoints_dc
        top_points = []
        bot_points = []
        for rad, x_fc, y_fc in anchor_points:
            top_points.append(
                apply_theta_shift(
                    t=idx,
                    r=rad,
                    x=x_fc - orig_center_fc,
                    y=y_fc - orig_center_fc,
                    flat_regress_func=video_data["top_edge_func"],
                    positive=True,
                )
            )
            bot_points.append(
                apply_theta_shift(
                    t=idx,
                    r=rad,
                    x=x_fc - orig_center_fc,
                    y=y_fc - orig_center_fc,
                    flat_regress_func=video_data["bot_edge_func"],
                    positive=False,
                )
            )

        top_points = np.array(top_points) + orig_center_fc
        bot_points = np.array(bot_points) + orig_center_fc

        # Plot the top and bottom edges
        ax.plot(top_points[:, 0], top_points[:, 1], c="g")
        ax.plot(bot_points[:, 0], bot_points[:, 1], c="b")

        # Append the figure to rom_frames
        rom_frames.append(fig)

        # Close the figure to avoid display overlap
        plt.close(fig)

    return rom_frames


if __name__ == "__main__":
    run()
