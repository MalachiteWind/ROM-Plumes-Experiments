import pickle
from typing import Any, cast

from ara_plumes import PLUME
from ara_plumes.typing import GrayVideo, Frame, PlumePoints
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .data import pickle_path
from .types import PolyData


def create_centerline(
    filename: str,
    img_range: list[int]=[0, -1],
    fixed_range: list[int]=[0, -1],
    gauss_kw: dict[str, Any]=None,
    circle_kw: dict[str, Any]=None,
    contour_kws: dict[str, Any]=None,
) -> dict[str, PolyData]:
    """Calculate the centerline path from a moviefile

    Args:
        filename: path from plume_videos/ to movie file.  Must have adjacent
            centerpoint file
        remainder of args are from ara_plumes.PLUME initialization
    """
    if gauss_kw is None:
        gauss_kw = {}
    if circle_kw is None:
        circle_kw = {}
    if contour_kws is None:
        contour_kws = {}

    origin_filename = filename + "_ctr.pkl"
    with open(pickle_path / origin_filename, "rb") as fh:
        origin = pickle.load(fh)
    plume = PLUME()
    np_filename = filename[:-3]+ "pkl" # replace mov with pkl
    with open(pickle_path / np_filename, "rb") as fh:
        plume.numpy_frames = pickle.load(fh)
    plume.orig_center = tuple(int(coord) for coord in origin)
    center, bottom, top = plume.train(
        img_range=img_range,
        fixed_range=fixed_range,
        concentric_circle_kws=circle_kw,
        get_contour_kws=contour_kws,
        **gauss_kw
    )
    visualize_points(plume.numpy_frames, center, bottom, top, n_plots=15)
    return {"main": None, "data": {"center": center, "bottom": bottom, "top": top}}


def visualize_points(
    vid: GrayVideo,
    center: list[tuple[Frame, PlumePoints]],
    bottom: list[tuple[Frame, PlumePoints]],
    top: list[tuple[Frame, PlumePoints]],
    n_plots: int=9
) -> Figure:
    n_plots = 9
    min_frame_t = center[0][0]
    max_frame_t = center[-1][0]
    plot_frameskip = (max_frame_t - min_frame_t) / n_plots
    frame_ids = [int(plot_frameskip * i) for i in range(n_plots)]
    n_rows = (n_plots + 2) // 3
    y_px, x_px = vid.shape[1:]
    vid_aspect = x_px/y_px
    fig, axes = plt.subplots(n_rows, 3, figsize=[vid_aspect * 9, 3 * n_rows])
    for frame_id, ax in zip(frame_ids, axes.flatten()):
        frame_t, frame_center = center[frame_id]
        _, frame_bottom = bottom[frame_id]
        _, frame_top = top[frame_id]
        ax = cast(Axes, ax)
        ax.imshow(vid[frame_t])
        ax.plot(frame_center[:, 1], frame_center[:, 2], "r.")
        ax.plot(frame_bottom[:, 1], frame_bottom[:, 2], "b.")
        ax.plot(frame_top[:, 1], frame_top[:, 2], "g.")
        ax.set_title(f"Frame {frame_t}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig
