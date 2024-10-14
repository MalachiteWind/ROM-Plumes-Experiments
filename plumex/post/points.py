# %%
from collections import namedtuple
from pathlib import Path
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
from ara_plumes import PLUME
from ara_plumes.models import get_contour
from matplotlib.figure import Figure
from mitosis import _load_trial_params
from mitosis import load_trial_data

from plumex.video_digest import _load_video
from plumex.video_digest import _plot_contours
from plumex.video_digest import _plot_frame
from plumex.video_digest import _plot_learn_path


frame = 838
trials = {
    "good": "cb6956",
    "extra blur": "430130",
    "one contour": "76b9db",
}
trials_folder = Path(__file__).absolute().parents[2] / "trials"
points_trials_folder = trials_folder / "center"

trial_info = {
    trial_tag: (
        load_trial_data(hexstr, trials_folder=points_trials_folder),
        _load_trial_params(hexstr, step=0, trials_folder=points_trials_folder),
    )
    for trial_tag, hexstr in trials.items()
}


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
def _single_img_range(kwargs: dict[str, Any], frame: int) -> dict[str, Any]:
    kwargs.pop("img_range", None)
    kwargs["frame"] = frame
    # deprecated 'mean_smoothing'
    if ckw := kwargs.get("circle_kw", False):
        ckw.pop("mean_smoothing")
    return kwargs


# %%

trial_info = {
    trial_tag: (data, _single_img_range(kwargs, frame))
    for trial_tag, (data, kwargs) in trial_info.items()
}
trial_plot_data = {
    trial_tag: _mini_video_digest(**kwargs)
    for trial_tag, (_, kwargs) in trial_info.items()
}


# %%
def _compare_step1_plot(trial_plot_data: dict[str, _PlotData]) -> Figure:
    n_variants = len(trial_plot_data)
    superfig = plt.figure(figsize=(10, 1.3 * n_variants))
    gs = superfig.add_gridspec(n_variants, 5)
    for superrow, (variant_name, exptup) in enumerate(trial_plot_data.items()):
        contours = get_contour(exptup.clean_im, **exptup.contour_kws)
        ax0 = superfig.add_subplot(gs[superrow, 0])
        ax0.set_ylabel(variant_name)
        _plot_frame(ax0, exptup.raw_im)
        _plot_frame(superfig.add_subplot(gs[superrow, 1]), exptup.clean_im)
        _plot_contours(
            superfig.add_subplot(gs[superrow, 2]),
            exptup.clean_im,
            exptup.orig_center,
            contours,
        )
        _plot_learn_path(
            superfig.add_subplot(gs[superrow, 3]),
            exptup.clean_im,
            exptup.center[0][1],
            exptup.top[0][1],
            exptup.bottom[0][1],
        )
        _plot_learn_path(
            superfig.add_subplot(gs[superrow, 4]),
            exptup.raw_im,
            exptup.center[0][1],
            exptup.top[0][1],
            exptup.bottom[0][1],
        )

    return superfig


# %% Figure 2
fig2 = _compare_step1_plot({k: v for k, v in trial_plot_data.items() if k == "good"})
fig2.suptitle("Data reduction process for plume film")
fig2.tight_layout()
fig2.subplots_adjust(top=0.80)
fig2.axes[0].set_ylabel("")
fig2.axes[0].set_xlabel("(a)")
fig2.axes[1].set_xlabel("(b)")
fig2.axes[2].set_xlabel("(c)")
fig2.axes[3].set_xlabel("(d)")
fig2.axes[4].set_xlabel("(e)")
pass
# %% Figure 4
fig4 = _compare_step1_plot({k: v for k, v in trial_plot_data.items()})
fig4.tight_layout()
fig4.subplots_adjust(top=0.95)
fig4.suptitle("How sensitive are the plume points to the extraction parameters?")
pass

# %%
