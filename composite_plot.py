import mitosis
import numpy as np
from ara_plumes.typing import Float2D
from ara_plumes.typing import GrayImage
from ara_plumes.typing import GrayVideo
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from typing import cast, List
from ara_plumes.typing import PlumePoints

from plumex.config import data_lookup
from plumex.regression_pipeline import _construct_rxy_f
from plumex.video_digest import _load_video
from plumex.data import load_centerpoints



# loading center regression results, come from 862 points instead of 866
# ignore 862 and 864
center_regress_hash = "85c44b"
center_step = mitosis.load_trial_data(center_regress_hash, trials_folder="trials/center-regress")
center_points = center_step[0]["data"]["center"]
center_fit_method = center_step[1]["main"]
center_coeff_dc = center_step[1]["regressions"][center_fit_method]["data"]
video, orig_center_fc = _load_video(data_lookup["filename"]["low-862"])

edge_points = load_centerpoints("/home/grisal/github/ARA-Plumes-Experiments/plume_videos/step1/390cee.dill")["data"]
# mitosis.load_trial_data()

# center = cast(List[tuple[int, PlumePoints]], edge_points["center"])
bot = cast(List[tuple[int, PlumePoints]], edge_points["bottom"])
top = cast(List[tuple[int, PlumePoints]], edge_points["top"])


edge_step = mitosis.load_trial_data("a52e31", trials_folder="trials/edge-regress")

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

def _in_frame(rxy_points:Float2D, frame:GrayImage)-> Float2D:
    y_range, x_range = frame.shape
    mask = rxy_points[:,1:] >=0
    mask = mask[:,0] & mask[:,1]
    less_than_x = rxy_points[:,1] <= x_range + orig_center_fc[0]
    mask = mask & less_than_x
    less_than_y = rxy_points[:,2] <= y_range+orig_center_fc[1]
    mask = mask & less_than_y

    return rxy_points[mask]
    


def _visualize_fits(
    video: GrayVideo,
    n_frames: int,
    center_coef: Float2D,
    center_func_method: str,
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
        raw_center_points = center_points[idx][1]
        raw_bot_points = bot[idx][1]
        raw_top_points = top[idx][1]

        ax.scatter(raw_center_points[:,1], raw_center_points[:,2], marker=".",c='r')
        ax.scatter(raw_bot_points[:,1], raw_bot_points[:,2], marker=".",c='g')
        ax.scatter(raw_top_points[:,1], raw_top_points[:,2], marker=".",c='b')

        raw_center_points[:,1:] -= orig_center_fc
    
        fit_centerpoints_dc = center_fit_func(raw_center_points)
        fit_centerpoints_dc[:,1:] += orig_center_fc

        fit_center_points_in_frame = _in_frame(fit_centerpoints_dc,frame_t)
        ax.plot(fit_center_points_in_frame[:,1], fit_center_points_in_frame[:,2], c='r')
        

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Video Frames", fontsize=16)

    # Tighter layout settings
    fig.tight_layout(pad=0.5)  # Reduce padding around the plot elements
    plt.subplots_adjust(
        left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1, wspace=0.1
    )


_visualize_fits(
    video[start_frame : end_frame + 1],
    n_frames=9,
    center_coef=center_coeff_dc,
    center_func_method=center_fit_method,
    start_frame=start_frame,
)
