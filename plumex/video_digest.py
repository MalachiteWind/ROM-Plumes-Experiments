import pickle
from logging import getLogger
from typing import Any
from typing import cast
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from rom_plumes import PLUME
from rom_plumes.models import get_contour
from rom_plumes.typing import Contour_List
from rom_plumes.typing import Frame
from rom_plumes.typing import GrayImage
from rom_plumes.typing import GrayVideo
from rom_plumes.typing import PlumePoints

from .data import PICKLE_PATH
from .plotting import CEST
from .plotting import CMAP


logger = getLogger(__name__)


def create_plumepoints(
    filename: str,
    img_range: tuple[int, int] = (0, -1),
    fixed_range: tuple[int, int] = (0, -1),
    gauss_space_kws: Optional[dict[str, Any]] = None,
    gauss_time_kws: Optional[dict[str, Any]] = None,
    circle_kw: Optional[dict[str, Any]] = None,
    contour_kws: Optional[dict[str, Any]] = None,
) -> dict[str, None | dict[str, list[tuple[Frame, PlumePoints]]]]:
    """Calculate the centerline path from a moviefile

    Args:
        filename: path from plume_videos/ to movie file.  Must have adjacent
            centerpoint file
        remainder of args are from rom_plumes.PLUME initialization
    """
    if gauss_space_kws is None:
        gauss_space_kws = {}
    if gauss_time_kws is None:
        gauss_time_kws = {}
    if circle_kw is None:
        circle_kw = {}
    if contour_kws is None:
        contour_kws = {}

    raw_vid, orig_center = _load_video(filename)
    clean_vid = PLUME.clean_video(
        raw_vid,
        fixed_range,
        gauss_space_blur=True,
        gauss_time_blur=True,
        gauss_space_kws=gauss_space_kws,
        gauss_time_kws=gauss_time_kws,
    )
    center, bottom, top = PLUME.video_to_ROM(
        clean_vid,
        orig_center,
        img_range,
        concentric_circle_kws=circle_kw,
        get_contour_kws=contour_kws,
    )
    visualize_points(
        raw_vid, clean_vid, orig_center, center, bottom, top, 15, contour_kws
    )
    return {"main": None, "data": {"center": center, "bottom": bottom, "top": top}}


def _load_video(filename: str) -> tuple[GrayVideo, tuple[int, int]]:
    origin_filename = filename + "_ctr.pkl"
    with open(PICKLE_PATH / origin_filename, "rb") as fh:
        origin = pickle.load(fh)
    np_filename = filename[:-3] + "pkl"  # replace mov with pkl
    with open(PICKLE_PATH / np_filename, "rb") as fh:
        raw_vid = pickle.load(fh)
    orig_center = (int(origin[0]), int(origin[1]))
    return raw_vid, orig_center


def visualize_points(
    raw_vid: GrayVideo,
    clean_vid: GrayVideo,
    origin: tuple[float, float],
    center: list[tuple[Frame, PlumePoints]],
    bottom: list[tuple[Frame, PlumePoints]],
    top: list[tuple[Frame, PlumePoints]],
    n_frames: int = 9,
    contour_kws: Optional[dict[str, Any]] = None,
) -> Figure:
    if contour_kws is None:
        contour_kws = {}
    min_frame_t = center[0][0]
    max_frame_t = center[-1][0]
    plot_frameskip = (max_frame_t - min_frame_t) / n_frames
    frame_ids = [int(plot_frameskip * i) for i in range(n_frames)]
    n_cols = 5
    y_px, x_px = raw_vid.shape[1:]
    vid_aspect = x_px / y_px
    fig, axes = plt.subplots(n_frames, n_cols, figsize=[vid_aspect * 8, 2 * n_frames])
    axes = np.reshape(axes, (n_frames, n_cols))  # enforce 2D array even if n_frames=1
    for frame_id, ax_row in zip(frame_ids, axes):
        frame_t, frame_center = center[frame_id]
        _, frame_bottom = bottom[frame_id]
        _, frame_top = top[frame_id]
        raw_im = raw_vid[frame_t]
        cln_im = clean_vid[frame_t]
        ax_row = cast(list[Axes], ax_row)
        ax_row[0].set_ylabel(f"Frame {frame_t}")
        _plot_frame(ax_row[0], raw_im)
        contours = get_contour(cln_im, **contour_kws)
        _plot_frame(ax_row[1], cln_im)
        _plot_contours(ax_row[2], cln_im, origin, contours)
        _plot_learn_path(ax_row[3], cln_im, frame_center, frame_top, frame_bottom)
        _plot_learn_path(ax_row[4], raw_im, frame_center, frame_top, frame_bottom)
    fig.tight_layout()
    return fig


def _plot_frame(ax: Axes, image: GrayImage):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_contours(
    ax: Axes, image: GrayImage, origin: tuple[float, float], contours: Contour_List
):
    _plot_frame(ax, image)
    for contour in contours:
        if len(contour) < 3:
            logger.warn("Skipping plot of contour < 3 points")
            continue
        cpath = Path(contour.reshape((-1, 2)), closed=True)
        cpatch = PathPatch(cpath, alpha=0.5, edgecolor=CEST, facecolor=CEST)
        ax.add_patch(cpatch)
    radii = 300
    num_circs = 6
    for radius in range(radii, num_circs * radii, radii):
        ax.add_patch(Circle(origin, radius, color=CMAP[4], fill=False))


def _plot_learn_path(
    ax: Axes,
    image: GrayImage,
    frame_center: PlumePoints,
    frame_top: PlumePoints,
    frame_bottom: PlumePoints,
    marker_size: Optional[int] = None

):
    _plot_frame(ax, image)
    if marker_size:
        ax.plot(frame_center[:, 1], frame_center[:, 2], "r.", markersize=marker_size)
        ax.plot(frame_bottom[:, 1], frame_bottom[:, 2], "b.",markersize=marker_size)
        ax.plot(frame_top[:, 1], frame_top[:, 2], "g.", markersize=marker_size)  
    else:
        ax.plot(frame_center[:, 1], frame_center[:, 2], "r.")
        ax.plot(frame_bottom[:, 1], frame_bottom[:, 2], "b.")
        ax.plot(frame_top[:, 1], frame_top[:, 2], "g.")
