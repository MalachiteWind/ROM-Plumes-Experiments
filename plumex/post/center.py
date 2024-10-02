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
low_set = ("lo1", "865", "866", "867", "868")
med_set = ("914", "916", "917", "871")
hi_set = ("hi1", "hi2", "919", "920")
full_methods = {
    "lin": "Linear",
    "poly": "Polynomial",
    "poly_para": "Parametric Quadratic",
    "poly_inv_pin": "Square-Root",
}


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
        "poly_inv_pin": _get_non_nan_scores(res["regressions"]["poly_inv_pin"]),
    }


trial_scores = {vidkey: get_scores(res) for vidkey, res in center_trial_data.items()}


def concat_video_scores(
    vid_scores: dict[str, dict[str, Float1D]]
) -> dict[str, Float1D]:
    return {
        meth: np.concatenate([scores[meth] for scores in vid_scores.values()], axis=0)
        for meth in ("poly_inv_pin", "lin", "poly_para", "poly")
    }


train_scores = concat_video_scores(
    {vidkey: trial_scores[vidkey] for vidkey in train_set}
)
test_scores = concat_video_scores({vidkey: trial_scores[vidkey] for vidkey in test_set})
low_scores = concat_video_scores({vidkey: trial_scores[vidkey] for vidkey in low_set})
med_scores = concat_video_scores({vidkey: trial_scores[vidkey] for vidkey in med_set})
hi_scores = concat_video_scores({vidkey: trial_scores[vidkey] for vidkey in hi_set})
all_min = min(min(accs) for accs in test_scores.values())
all_max = max(max(accs) for accs in test_scores.values())


# %%
def compare_histogram(concat_scores):
    plt.title("Plume-Center Regressions Performance on Test Set")
    bins = np.linspace(0, 1.5, 10)
    plt.hist(
        [-score for score in concat_scores.values()],
        label=[full_methods[method] for method in concat_scores.keys()],
        bins=list(bins),
    )
    plt.ylabel("Number of frames")
    plt.xlabel("Relative Extrapolation Error")
    plt.legend()


# %% Figure 6
compare_histogram(test_scores)
print(
    r"""Caption: The square-root regression best extrapolated
      the plume's drift away from the point source,
      followed closely by linear regression (95% confidence level)."""
)

# %% Optional investigation figures - combined test/train
compare_histogram(low_scores)
compare_histogram(med_scores)
compare_histogram(hi_scores)
