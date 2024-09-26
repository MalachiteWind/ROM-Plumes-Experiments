import mitosis
from matplotlib import pyplot as plt

from plumex.config import data_lookup
from plumex.regression_pipeline import _construct_rxy_f
from plumex.video_digest import _load_video

center_step = mitosis.load_trial_data("5507db", trials_folder="trials/center-regress")
center_points = center_step[0]["data"]["center"]
center_coeff_dc = center_step[1]["regressions"]["poly_inv_pin"]["data"]
video, orig_center_fc = _load_video(data_lookup["filename"]["low-866"])

start_frame = center_points[0][0]


def frame_to_ind(fr: int) -> int:
    return fr - start_frame

# fix
# hardcode to demonstrate
frame_id = 500
frame_ind = frame_to_ind(frame_id)
fit_func = _construct_rxy_f(center_coeff_dc[frame_ind], "poly_inv_pin")
raw_centerpoints = center_points[frame_ind][1]
raw_frame = video[frame_id]
fit_centerpoints_dc = fit_func(raw_centerpoints)

plt.imshow(raw_frame)
plt.plot(raw_centerpoints, marker=".")
plt.plot(fit_centerpoints_dc)
