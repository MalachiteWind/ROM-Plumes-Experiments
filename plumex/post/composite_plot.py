from pathlib import Path
from typing import Callable
from typing import cast
from typing import List
from typing import TypedDict

import mitosis
import numpy as np
from ara_plumes.typing import GrayVideo
from ara_plumes.typing import PlumePoints

from plumex.config import data_lookup
from plumex.post.post_utils import _visualize_fits
from plumex.post.post_utils import _visualize_multi_edge_fits
from plumex.post.post_utils import create_edge_func
from plumex.post.post_utils import plot_raw_frames
from plumex.post.post_utils import RegressionData
from plumex.types import Float2D
from plumex.video_digest import _load_video


trials_folder = Path(__file__).absolute().parents[2] / "trials"

# ignore 862 and 864
# (center_regress_hash, video_lookup_word, edge_regress_hash)
trial_lookup_key = {
    "862": {"default": ("85c44b", "low-862", "a52e31")},
    "865": {"default": ("aedee1", "low-865", "264935")},
    "866": {
        "default": ("5507db", "low-866", "741931"),
    },
    "867": {"default": ("98c0dc", "low-867", "baa666")},
    "868": {"default": ("ea68e7","low-868","e90c88")},
    "869": {"default": ("7314c2","low-869","db4a6c")},
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



vid_names = ["866", "867", "868", "869"]
frame_ids = [0, 100, 350, 450]


video_data = [
    _unpack_data(*trial_lookup_key[name]["default"]) for name in vid_names
]

lenghts = [len(vid["video"]) for vid in video_data]

_visualize_multi_edge_fits(
    video_data=video_data,
    frame_ids=frame_ids,
    title="Edge Regression",
    subtitles=vid_names,
    figsize=(12, 8),
    plot_on_raw_points=True
)

print()
