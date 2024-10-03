from typing import cast
from typing import List

import mitosis
import numpy as np
from ara_plumes.typing import PlumePoints

from plumex.config import data_lookup
from plumex.video_digest import _load_video
from post_utils import _visualize_fits
from post_utils import create_edge_func


# ignore 862 and 864
center_regress_hash = "85c44b"
video_lookup_word = "low-862"
edge_regress_hash = "a52e31"

center_mitosis_step = mitosis.load_trial_data(
    center_regress_hash, trials_folder="trials/center-regress"
)

# This is the same as center
# center_points = center_mitosis_step[0]["data"]["center"]
center_fit_method = center_mitosis_step[1]["main"]
center_coeff_dc = center_mitosis_step[1]["regressions"][center_fit_method]["data"]

video, orig_center_fc = _load_video(data_lookup["filename"][video_lookup_word])

edge_mitosis_step = mitosis.load_trial_data(
    edge_regress_hash, trials_folder="trials/edge-regress"
)
edge_plume_points = edge_mitosis_step[0]["data"]

# This is the same as center_points
center = cast(List[tuple[int, PlumePoints]], edge_plume_points["center"])
bot = cast(List[tuple[int, PlumePoints]], edge_plume_points["bottom"])
top = cast(List[tuple[int, PlumePoints]], edge_plume_points["top"])

top_method, bot_method = edge_mitosis_step[1]["main"]

edge_coefs_top = np.nanmean(
    edge_mitosis_step[1]["accs"]["top"][top_method]["coeffs"], axis=0
)
edge_coefs_bot = np.nanmean(
    edge_mitosis_step[1]["accs"]["bot"][bot_method]["coeffs"], axis=0
)

top_func = create_edge_func(edge_coefs_top, top_method)
bot_func = create_edge_func(edge_coefs_bot, bot_method)

start_frame = center[0][0]
end_frame = center[-1][0]

_visualize_fits(
    video[start_frame : end_frame + 1],
    n_frames=9,
    center_coef=center_coeff_dc,
    center_func_method=center_fit_method,
    center_plume_points=center,
    bottom_plume_points=bot,
    top_plume_points=top,
    top_edge_func=top_func,
    bot_edge_func=bot_func,
    start_frame=start_frame,
    orig_center_fc=orig_center_fc,
    plot_on_raw_points=True
)
# print("test change")
_visualize_fits(
    video[start_frame : end_frame + 1],
    n_frames=9,
    center_coef=center_coeff_dc,
    center_func_method=center_fit_method,
    center_plume_points=center,
    bottom_plume_points=bot,
    top_plume_points=top,
    top_edge_func=top_func,
    bot_edge_func=bot_func,
    start_frame=start_frame,
    orig_center_fc=orig_center_fc,
    plot_on_raw_points=False
)