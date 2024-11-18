# %%
from collections import namedtuple
from pathlib import Path
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
from mitosis import _load_trial_params
from rom_plumes import PLUME
from rom_plumes.models import get_contour

from plumex.video_digest import _load_video
from plumex.video_digest import _plot_contours

def run():

    # %%
    frame = 1000
    hexstr = "cb6956"
    trials_folder = Path(__file__).absolute().parents[2] / "trials"
    points_trials_folder = trials_folder / "center"
    tparams = _load_trial_params(hexstr, step=0, trials_folder=points_trials_folder)


    # %%
    def _single_img_range(kwargs: dict[str, Any], frame: int) -> dict[str, Any]:
        kwargs.pop("img_range", None)
        kwargs["frame"] = frame
        # deprecated 'mean_smoothing'
        if ckw := kwargs.get("circle_kw", False):
            ckw.pop("mean_smoothing")
        return kwargs


    _PlotData = namedtuple(
        "_PlotData",
        ["orig_center", "raw_im", "clean_im", "contour_kws", "center", "bottom", "top"],
    )


    def _mini_video_digest(
        filename: str,
        frame: int,
        fixed_range: tuple[int, int] = (0, -1),
        gauss_space_kws: Optional[dict[str, Any]] = None,
        gauss_time_kws: Optional[dict[str, Any]] = None,
        circle_kw: Optional[dict[str, Any]] = None,
        contour_kws: Optional[dict[str, Any]] = None,
    ) -> _PlotData:
        raw_vid, orig_center = _load_video(filename)
        clean_vid = PLUME.clean_video(
            raw_vid,
            fixed_range,
            gauss_space_blur=True,
            gauss_time_blur=True,
            gauss_space_kws=gauss_space_kws,
            gauss_time_kws=gauss_time_kws,
        )
        img_range = (frame, frame + 1)
        center, bottom, top = PLUME.video_to_ROM(
            clean_vid,
            orig_center,
            img_range,
            concentric_circle_kws=circle_kw,
            get_contour_kws=contour_kws,
        )
        return _PlotData(
            orig_center,
            raw_vid[frame],
            clean_vid[frame],
            contour_kws or {},
            center,
            bottom,
            top,
        )


    # %%
    # The first cell of fig 1, to be manually manipulated
    fig1a = plt.figure()
    example_data = _mini_video_digest(**_single_img_range(tparams, frame))
    contours = get_contour(example_data.clean_im, **example_data.contour_kws)
    _plot_contours(
        fig1a.add_subplot(1, 1, 1),
        example_data.clean_im,
        example_data.orig_center,
        contours,
    )


if __name__ == "__main__":
    run()
