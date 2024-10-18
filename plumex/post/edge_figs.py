from pathlib import Path
from typing import cast
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import mitosis
import numpy as np
from ara_plumes.models import flatten_edge_points
from ara_plumes.typing import PlumePoints

from plumex.post.post_utils import _visualize_multi_edge_fits
from plumex.post.post_utils import create_edge_func
from plumex.post.post_utils import RegressionData
from plumex.config import data_lookup
from plumex.regress_edge import create_sin_func
from plumex.video_digest import _load_video


trials_folder = Path(__file__).absolute().parents[2] / "trials"

# ignore 862 and 864
# (center_regress_hash, video_lookup_word, edge_regress_hash)
trial_lookup_key = {
    "low 862": {"default": ("85c44b", "low-862", "a52e31")},
    "low 865": {"default": ("aedee1", "low-865", "264935")},
    "low 866": {
        "default": ("5507db", "low-866", "741931"),
    },
    "low 867": {"default": ("98c0dc", "low-867", "baa666")},
    "low 868": {"default": ("ea68e7", "low-868", "e90c88")},
    "low 869": {"default": ("7314c2", "low-869", "db4a6c")},
    "low 913": {"default": ("bc1c70", "low-913", "21a901")},
    "med 914": {"default": ("0d3391", "med-914", "8dd223")},
    "med 916": {"default": ("c41675", "med-916", "776714")},
    "med 917": {"default": ("9be0a9", "med-917", "e8e683")},
    "med 871": {"default": ("a42cc9", "med-871", "7cab4b")},
    "hi 919": {"default": ("1e0610", "hi-919", "2557c9")},
    "hi 920": {"default": ("c0ab39", "hi-920", "485d19")},
    "no wind": {"default": ("db4841", "blender-nowind", "d7598b")},
    "wind": {"default": ("61645e", "blender-wind", "a559a7")},
    "low 1": {"default": ("677028", "low-1", "ea256d")},
    "hi 1": {"default": ("729d03", "hi-1", "487513")},
    "hi 2": {"default": ("56a332", "hi-2", "3115a5")},
}


def _unpack_data(
    center_regress_hash: str,
    video_lookup_keyword: str,
    edge_regress_hash: str,
) -> RegressionData:

    video, orig_center_fc = _load_video(data_lookup["filename"][video_lookup_keyword])
    center_mitosis_step = mitosis.load_trial_data(
        hexstr=center_regress_hash, trials_folder=trials_folder / "center-regress"
    )
    edge_mitosis_step = mitosis.load_trial_data(
        hexstr=edge_regress_hash, trials_folder=trials_folder / "edge-regress"
    )

    center_fit_method = center_mitosis_step[1]["main"]
    center_coeff_dc = center_mitosis_step[1]["regressions"][center_fit_method]["data"]

    edge_plume_points = edge_mitosis_step[0]["data"]

    center = cast(List[tuple[int, PlumePoints]], edge_plume_points["center"])
    bot = cast(List[tuple[int, PlumePoints]], edge_plume_points["bottom"])
    top = cast(List[tuple[int, PlumePoints]], edge_plume_points["top"])

    top_edge_method, bot_edge_method = edge_mitosis_step[1]["main"]

    edge_coefs_top = np.nanmean(
        edge_mitosis_step[1]["accs"]["top"][top_edge_method]["coeffs"], axis=0
    )
    edge_coefs_bot = np.nanmean(
        edge_mitosis_step[1]["accs"]["bot"][bot_edge_method]["coeffs"], axis=0
    )

    top_func = create_edge_func(edge_coefs_top, top_edge_method)
    bot_func = create_edge_func(edge_coefs_bot, bot_edge_method)

    start_frame = center[0][0]
    end_frame = center[-1][0]

    return RegressionData(
        video=video[start_frame : end_frame + 1],
        center_coef=center_coeff_dc,
        center_func_method=center_fit_method,
        center_plume_points=center,
        top_plume_points=top,
        bottom_plume_points=bot,
        top_edge_func=top_func,
        bot_edge_func=bot_func,
        start_frame=start_frame,
        orig_center_fc=orig_center_fc,
    )


vid_names = ["low 869", "med 914", "hi 920"]
frame_ids = [250, 750, 1000, 1200]


video_data = [_unpack_data(*trial_lookup_key[name]["default"]) for name in vid_names]


def _create_fig2b_raw():

    figsize = (20, 10)
    what_to_plot = {
        "plot_edge_points": True,
        "plot_center_points": False,
        "plot_edge_regression": True,
        "plot_center_regression": False,
    }

    _visualize_multi_edge_fits(
        video_data=video_data,
        frame_ids=frame_ids,
        title="Edge Regression (Raw Points)",
        subtitles=vid_names,
        figsize=figsize,
        plot_on_raw_points=True,
        **what_to_plot
    )


def _create_fig2b_regression():

    figsize = (20, 10)
    what_to_plot = {
        "plot_edge_points": False,
        "plot_center_points": False,
        "plot_edge_regression": True,
        "plot_center_regression": True,
    }

    _visualize_multi_edge_fits(
        video_data=video_data,
        frame_ids=frame_ids,
        title="Edge Regression (Attached Center)",
        subtitles=vid_names,
        figsize=figsize,
        plot_on_raw_points=False,
        **what_to_plot
    )


def _create_fig1d():
    what_to_plot = {
        "plot_edge_points": False,
        "plot_center_points": False,
        "plot_edge_regression": True,
        "plot_center_regression": True,
    }

    _visualize_multi_edge_fits(
        video_data=[video_data[-1]],
        frame_ids=[frame_ids[0]],
        subtitles=[vid_names[-1]],
        figsize=(10, 5),
        plot_on_raw_points=False,
        **what_to_plot
    )


def _create_fig1c():
    edge_regress_hi920_hash = "485d19"
    frame_id = 250

    edge_data = mitosis.load_trial_data(
        hexstr=edge_regress_hi920_hash, trials_folder=trials_folder / "edge-regress"
    )

    center_plumepoints = cast(
        List[Tuple[int, PlumePoints]], edge_data[0]["data"]["center"]
    )
    bot_plumepoints = cast(
        List[Tuple[int, PlumePoints]], edge_data[0]["data"]["bottom"]
    )

    rad_dist = flatten_edge_points(
        mean_points=center_plumepoints[frame_id][1],
        vari_points=bot_plumepoints[frame_id][1],
    )

    r_max = np.max(rad_dist[:, 0])
    r_lin = np.linspace(0, r_max, 101)
    time = center_plumepoints[frame_id][0]
    t_lin = np.ones(len(r_lin)) * time

    bot_coef_sin = np.nanmean(edge_data[1]["accs"]["bot"]["sinusoid"]["coeffs"], axis=0)

    bot_sin_func = create_sin_func(bot_coef_sin)

    bot_sin_vals = bot_sin_func(t_lin, r_lin)

    fig, ax = plt.subplots()
    ax.plot(r_lin, bot_sin_vals, c="g")
    ax.scatter(rad_dist[:, 0], rad_dist[:, 1], c="k")
    ax.set_title(r"$d(r,t)=A \sin( \omega r - \gamma t + B) + C + r D$", fontsize=18)
    ax.hlines(
        y=0, xmin=np.min(rad_dist[:, 0]), xmax=np.max(rad_dist[:, 0]) - 10.0, colors="r"
    )
    ax.set_ylim(top=850)
    ax.set_xlabel("r", fontsize=18)
    ax.set_ylabel("d", fontsize=18)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    return fig


_create_fig1c()
_create_fig1d()
_create_fig2b_raw()
_create_fig2b_regression()
