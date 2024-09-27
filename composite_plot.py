import mitosis
import numpy as np
from ara_plumes.typing import GrayVideo
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from plumex.config import data_lookup
from plumex.video_digest import _load_video

# from plumex.regression_pipeline import _construct_rxy_f

center_step = mitosis.load_trial_data("5507db", trials_folder="trials/center-regress")
center_points = center_step[0]["data"]["center"]
center_coeff_dc = center_step[1]["regressions"]["poly_inv_pin"]["data"]
video, orig_center_fc = _load_video(data_lookup["filename"]["low-866"])

start_frame = center_points[0][0]
end_frame = center_points[-1][0]


def frame_to_ind(fr: int) -> int:
    return fr - start_frame


# fix
# hardcode to demonstrate
# frame_id = 500
# frame_ind = frame_to_ind(frame_id)
# fit_func = _construct_rxy_f(center_coeff_dc[frame_ind], "poly_inv_pin")
# raw_centerpoints = center_points[frame_ind][1]
# raw_frame = video[frame_id]
# fit_centerpoints_dc = fit_func(raw_centerpoints)

# plt.imshow(raw_frame)
# plt.plot(raw_centerpoints, marker=".")
# plt.plot(fit_centerpoints_dc)


def _visualize_fits(video: GrayVideo, n_frames: int, start_frame:int=0) -> Figure:
    """
    plot center regression and unflattened edge regression on frames.
    """

    frameskip = len(video) / n_frames

    frame_ids = [int(frameskip * i) for i in range(n_frames)]
    grid_size = int(np.ceil(np.sqrt(n_frames)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 9))
    axes = axes.flatten()

    for i, idx in enumerate(frame_ids):
        ax = axes[i]
        frame_t = video[idx]
        ax.imshow(frame_t, cmap="gray")
        ax.set_title(f"frame {idx+start_frame}")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Video Frames", fontsize = 16)

    # Tighter layout settings
    fig.tight_layout(pad=0.5)  # Reduce padding around the plot elements
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1, wspace=0.1)


_visualize_fits(video[start_frame: end_frame+1], n_frames=9, start_frame=start_frame)
