from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mitosis import load_trial_data

from plumex.regression_pipeline import MultiRegressionResults
from plumex.regression_pipeline import RegressionResults
from plumex.types import Float1D

center_trial_keys = {
    "lo1": "677028",
    "865": "b867ae",
    "866": "5507db",
    "867": "98c0dc",
    "868": "ea68e7",
    "869": "7314c2",
    "871": "a42cc9",
    "913": "bc1c70",
    "914": "0d3391",
    "916": "c41675",
    "917": "9be0a9",
    "919": "1e0610",
    "920": "c0ab39",
    "hi1": "729d03",
    "hi2": "56a332",
}
train_set = ("lo1", "865", "866", "867", "871", "913", "914", "917", "919", "hi1")
test_set = ("868", "869", "916", "920", "hi2")
low_se = ("lo1", "865", "866", "867", "868")
trials_folder = Path(__file__).absolute().parents[2] / "trials"
center_trials_dir = trials_folder / "center-regress"
center_trial_data: dict[str, MultiRegressionResults] = {
    vidkey: load_trial_data(trialkey, trials_folder=center_trials_dir)[1]
    for vidkey, trialkey in center_trial_keys.items()
}


def get_scores(res: MultiRegressionResults) -> dict[str, Float1D]:
    """Get the scores from each frame, for each regression method"""

    def _get_non_nan_scores(res: RegressionResults):
        return res["val_acc"][res["non_nan_inds"]]

    return {
        "lin": _get_non_nan_scores(res["regressions"]["linear"]),
        "poly": _get_non_nan_scores(res["regressions"]["poly"]),
        "poly_para": _get_non_nan_scores(res["regressions"]["poly_para"]),
        "poly_inv": _get_non_nan_scores(res["regressions"]["poly_inv_pin"]),
    }


trial_scores = {vidkey: get_scores(res) for vidkey, res in center_trial_data.items()}
train_vid_scores = {vidkey: trial_scores[vidkey] for vidkey in train_set}
test_vid_scores = {vidkey: trial_scores[vidkey] for vidkey in test_set}


def concat_video_scores(
    vid_scores: dict[str, dict[str, Float1D]]
) -> dict[str, Float1D]:
    return {
        meth: np.concatenate([scores[meth] for scores in vid_scores.values()], axis=0)
        for meth in ("lin", "poly", "poly_para", "poly_inv")
    }


train_scores = concat_video_scores(train_vid_scores)
test_scores = concat_video_scores(test_vid_scores)
low_scores = concat_video_scores()  # fill this in here

[
    plt.hist(scores, label=method)
    for method, scores in train_scores.items()
    if method != "poly"
]
plt.legend()
