"""Experiment and utilities to regress the center of a plume video.

Developer's note:
    The abbreviations "fc", "dc", and "pc" refer to frame coordinates (in which
    origin is the top left, y axis pointing down), decentered coordinates (in
    which origin is the plume source, origin pointing down), and plot
    coordinates (in which origin is the bottom left, y axis pointing up)
"""
from collections.abc import Sequence
from logging import getLogger
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional
from typing import TypedDict
from warnings import warn

import numpy as np
from ara_plumes.models import PLUME
from ara_plumes.typing import Bool1D
from ara_plumes.typing import Frame
from ara_plumes.typing import PlumePoints
from ara_plumes.typing import X_pos
from ara_plumes.typing import Y_pos
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from tqdm import tqdm

from .plotting import CMAP
from .plotting import CMEAS
from .types import Float1D
from .types import Float2D
from .types import Int1D
from .types import NpFlt

LOGGER = getLogger(__name__)
REGRESSION_METHODS = {
    "linear": ("y=ax + b", "ab"),
    "poly": ("y=ax^2 + bx + c", "abc"),
    "poly_inv_pin": ("x=ay^2 + by + c", "abc"),
    "poly_para": ("x=ar^2 + br + c;  y=dr^2 + er + f", "abcdef"),
}


class RegressionResults(TypedDict):
    main: float
    non_nan_inds: Bool1D
    train_acc: Float1D
    n_train: Int1D
    n_val: Int1D
    val_acc: Float1D
    data: Float2D
    n_frames: int


class MultiRegressionResults(TypedDict):
    main: str
    data: Float2D
    n_frames: int
    regressions: dict[str, RegressionResults]


def multi_regress_centerline(
    data: dict[str, list[tuple[Frame, PlumePoints]]],
    r_split: int,
    poly_deg: int = 2,
) -> MultiRegressionResults:
    """Mitosis experiment to compare regress_centerline across methods

    Arguments same as regress_centerline, without 'regression_method'

    Returns:
        Experiment results as regress_centerline, but main metric is name of
        best method, "data" is data from best method, and "accs" is dictionary
        of method name to its validation and train accuracies by frame
    """
    meth_results: dict[str, RegressionResults] = {}
    main_accs: list[tuple[str, float]] = []
    coeffs = []
    decenter = cast(tuple[X_pos, Y_pos], data["center"][0][1][0, 1:])
    for method in REGRESSION_METHODS:
        meth_results[method] = regress_centerline(
            data, r_split, method, poly_deg, display=False
        )
        main_accs.append((method, meth_results[method]["main"]))
        coeffs.append(meth_results[method]["data"])

    main_accs.sort(key=lambda tup: tup[1])
    best_method = main_accs[-1][0]
    best_data = meth_results[best_method]["data"]
    n_frames = -1
    for result in meth_results.values():
        result["data"]
        n_frames = result["n_frames"]

    # Plot validation distribution
    methods_ascending = [metric for metric, _ in main_accs]
    nested_best_n_results = _nest_best_n_results(methods_ascending, meth_results)

    fig = _plot_acc_hist_by_tranche(nested_best_n_results)
    fig.suptitle(f"Validation Accuracy over {n_frames} frames")

    # Plot each method's points
    _visualize_points(
        data["center"],
        coeffs,
        list(REGRESSION_METHODS.keys()),
        r_split=r_split,
        n_plots=15,
        origin_fc=decenter,
    )

    # Plot accuracy distribution by number of data points
    fig = _scatter_accuracy_npoints_by_tranche(nested_best_n_results)
    fig.suptitle("Validation Accuracy by number of train/validation points")

    # Plot coefficient distributions
    for method, res in meth_results.items():
        n_frames = res["n_frames"]
        fig = _plot_coef_dist(res["data"], *REGRESSION_METHODS[method])
        fig.suptitle(fig.get_suptitle() + f" ({n_frames} frames)")
    return MultiRegressionResults(
        main=best_method, data=best_data, n_frames=n_frames, regressions=meth_results
    )


def regress_centerline(
    data: dict[str, list[tuple[Frame, PlumePoints]]],
    r_split: int,
    regression_method: str,
    poly_deg: int = 2,
    display: bool = True,
) -> RegressionResults:
    """Mitosis experiment to fit mean path of plume points

    Args:
        data: output of video_digest step, a dictionary of center, top, and
            bottom points for each frame of a video
        r_split: the pixel coordinate of boundary between validation (left)
            and training (right) data
        regression_method: method for drawing curve through points as
            understood by PLUME.regress_multiframe_mean
        poly_deg: polynomial degree of curve used in regression_method

    Returns:
        Experiment results of train and validation accuracy.  "data" key
        is of shape (number of timepoints, number of coefficients in
        regression_method). "main" metric is average validation accuracy
    """
    mean_points = data["center"]
    train_set_fc, val_set_fc = _split_into_train_val(mean_points, r_split)
    n_train = cast(Int1D, np.array([len(points) for _, points in train_set_fc]))
    n_val = cast(Int1D, np.array([len(points) for _, points in val_set_fc]))
    origin_fc = cast(tuple[X_pos, Y_pos], data["center"][0][1][0, 1:])
    coef_time_series_dc = PLUME.regress_multiframe_mean(
        mean_points=train_set_fc,
        regression_method=regression_method,
        poly_deg=poly_deg,
        decenter=origin_fc,
    )

    train_set_dc = [(frame, points - (0, *origin_fc)) for frame, points in train_set_fc]
    val_set_dc = [(frame, points - (0, *origin_fc)) for frame, points in val_set_fc]
    train_acc = get_coef_acc(coef_time_series_dc, train_set_dc, regression_method)
    val_acc = get_coef_acc(coef_time_series_dc, val_set_dc, regression_method)
    non_nan_inds = ~np.isnan(val_acc)
    non_nan_val_acc = val_acc[non_nan_inds]

    n_frames = len(mean_points)
    if display:
        plt.hist((train_acc, val_acc), label=("Train", "Validation"))
        plt.legend()
        plt.title(f"Accuracy over {n_frames} frames")
        plt.legend()

        _visualize_points(
            mean_points,
            [coef_time_series_dc],
            [regression_method],
            r_split=r_split,
            n_plots=15,
            origin_fc=origin_fc,
        )
        _plot_acc_dist(
            val_acc[non_nan_inds], n_val[non_nan_inds], n_train[non_nan_inds]
        )
        _plot_coef_dist(coef_time_series_dc, *REGRESSION_METHODS[regression_method])
    if len(non_nan_val_acc) == 0:
        warn(
            "No frames have any points in the validation set.  "
            "Try decreasing r_split to allow more points in validation set.",
            RuntimeWarning,
            stacklevel=2,
        )
    if len(non_nan_val_acc) < len(val_acc):
        warn(
            "Some frames do not have any points in the validation set",
            RuntimeWarning,
            stacklevel=2,
        )

    return RegressionResults(
        main=non_nan_val_acc.mean(),
        non_nan_inds=non_nan_inds,
        train_acc=train_acc,
        n_train=n_train,
        n_val=n_val,
        val_acc=val_acc,
        data=coef_time_series_dc,
        n_frames=n_frames,
    )


def _split_into_train_val(
    mean_points: list[tuple[Frame, PlumePoints]], r_split: int
) -> tuple[list[tuple[Frame, PlumePoints]], list[tuple[Frame, PlumePoints]]]:
    """
    Splits mean_points into training and validation sets based
    on coordinate values along the x-axis.

    Parameters:
    ----------
    mean_points:
        Mean points returned from PLUME.train().

    r_split:
        Value to determine where the split of data occurs on the frame.
        Distance along radial axis.

    Returns:
    -------
    train_set:
        Frame points associated with x-coordinate values greater than
        or equal to r_split.

    val_set:
        Frame points associated with x-coordinate values less than
        r_split.
    """
    train_set = []
    val_set = []
    for (t, frame_points) in tqdm(mean_points):
        mask = frame_points[:, 0] <= r_split

        train_set.append((t, frame_points[mask]))
        val_set.append((t, frame_points[~mask]))

    return train_set, val_set


def _construct_rxy_f(
    coef: Float1D, regression_method: str
) -> Callable[[PlumePoints], PlumePoints]:
    """Create a function from regression that maps r,x,y to the inputs

    Note that functions fit on decentered coordinates should only act on
    decentered coordinates.

    Parameters:
        coef:
            array of poly coefficients in descending degree order.

    Returns:
        Callable function that takes r, x, and y as the -1 axis of the
        argument and returns in the same shape
    """
    if regression_method == "poly_para":
        mid_index = len(coef) // 2
        f1 = np.polynomial.Polynomial(coef[:mid_index][::-1])
        f2 = np.polynomial.Polynomial(coef[mid_index:][::-1])

        def f(rxy):  # type: ignore
            r = rxy[..., 0]
            x = f1(r)
            y = f2(r)
            return np.stack([r, x, y], axis=-1)

    elif regression_method in ("poly_inv", "poly_inv_pin"):
        # if x = ay^2 + by + c, then y = sqrt((x-c)/a + b^2/(4a^2)) - b/(2a)
        a, b, c = coef

        def f(rxy):  # type: ignore
            r = rxy[..., 0]
            x = rxy[..., 1]
            y = -np.sqrt((x - c) / a + b**2 / (4 * a**2)) - b / (2 * a)
            return np.stack([r, x, y], axis=-1)

    elif regression_method in ("linear", "poly"):

        def f(rxy):
            f_y_of_x = np.polynomial.Polynomial(coef[::-1])  # type: ignore
            r = rxy[..., 0]
            x = rxy[..., 1]
            y = f_y_of_x(x)
            return np.stack([r, x, y], axis=-1)

    else:
        raise ValueError("Unrecognized regression method")
    return f


def get_coef_acc(
    coef_time_series_dc: np.ndarray[Any, NpFlt],
    eval_set_dc: list[tuple[Frame, PlumePoints]],
    regression_method: str,
) -> Float1D:
    """Get the L2 accuracy of learned coefficients on PlumePoints"""
    if len(coef_time_series_dc) != len(eval_set_dc):
        raise TypeError("length of arrays must match")

    n_frames = len(coef_time_series_dc)
    accs = cast(Float1D, np.zeros(n_frames))
    for i in range(n_frames):
        coef_i = coef_time_series_dc[i]

        pred_dc = _construct_rxy_f(coef_i, regression_method)
        _, rxy_true_dc = eval_set_dc[i]

        rxy_pred_dc = pred_dc(rxy_true_dc)
        if len(rxy_true_dc) == 0:
            accs[i] = np.nan
        else:
            accs[i] = -np.linalg.norm(
                rxy_true_dc[:, 1:] - rxy_pred_dc[:, 1:]
            ) / np.linalg.norm(rxy_true_dc[:, 1:])

    return accs


def _plot_acc_dist(
    val_acc: Float1D,
    n_val: Int1D,
    n_train: Int1D,
    ax: Axes | None = None,
    label: str | None = None,
    title_on: bool = True,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1)
        ax = cast(Axes, ax)
    n_frames = len(val_acc)
    ax.scatter(val_acc, n_val + n_train, label=label)
    if title_on:
        ax.set_title(f"Accuracy Distribution across {n_frames} frames.")
    ax.set_xlabel("Validation accuracy")
    ax.set_ylabel("points in frame")
    return ax


def _plot_coef_dist(
    coef_time_series: Float2D,
    expression: Optional[str] = None,
    terms: Optional[Sequence[str]] = None,
) -> Figure:
    n_coeffs = coef_time_series.shape[1]
    fig, axs = plt.subplots(1, n_coeffs, figsize=[3 * n_coeffs, 3])
    if not terms:
        terms = [chr(coef_ind) for coef_ind in range(n_coeffs)]
    for ax, coef_ind, term_symbol in zip(axs, range(n_coeffs), terms, strict=True):
        ax = cast(Axes, ax)
        ax.hist(coef_time_series[:, coef_ind])
        ax.set_xlabel(term_symbol)
    suptitle = "Distribution of coefficients (decentered frame coordinates)"
    if expression:
        suptitle += f": {expression}"
    fig.suptitle(suptitle)
    return fig


def _visualize_points(
    mean_points: list[tuple[Frame, PlumePoints]],
    coef_time_series_dc: Sequence[Float2D],
    regression_methods: Sequence[str],
    origin_fc: tuple[X_pos, Y_pos],
    r_split: None | float = None,
    n_plots: int = 9,
) -> Figure:
    min_frame_t = mean_points[0][0]
    max_frame_t = mean_points[-1][0]
    plot_frameskip = (max_frame_t - min_frame_t) / n_plots
    frame_ids = [int(plot_frameskip * i) for i in range(n_plots)]
    n_rows = (n_plots + 2) // 3
    x_max = max(np.max(points[1][:, 1]) for points in mean_points)
    y_max = max(np.max(points[1][:, 2]) for points in mean_points)
    # in frame coordinates, y is pixels below top, but in plot coordinates
    # it's above x-axis
    origin_pc = (origin_fc[0], float(y_max - origin_fc[1]))
    fig, axes = plt.subplots(n_rows, 3, figsize=[1.5 * 9, 3 * n_rows])
    fig.suptitle("How well does regression work?")
    for frame_id, ax in zip(frame_ids, axes.flatten()):
        frame_t, frame_points = mean_points[frame_id]
        xy_true_fc = frame_points[:, 1:]
        ax.plot(
            xy_true_fc[:, 0],
            y_max - xy_true_fc[:, 1],
            ".",
            color=CMEAS,
            label="centerpoints",
        )
        if r_split:
            ax.add_patch(Circle(origin_pc, r_split, color="k", fill=False))
        for meth_ind, (coeff_meth, method) in enumerate(
            zip(coef_time_series_dc, regression_methods)
        ):
            coeffs = coeff_meth[frame_id]
            predict_dc = _construct_rxy_f(coeffs, method)
            frame_points_dc = np.hstack(
                (frame_points[:, :1], frame_points[:, 1:] - origin_fc)
            )
            xy_pred_dc = predict_dc(frame_points_dc)[:, 1:]
            xy_pred_fc = xy_pred_dc + origin_fc
            LOGGER.debug(f"Moving origin from {xy_pred_dc[0]} to f{xy_pred_fc[0]}")
            ax.plot(
                xy_pred_fc[:, 0],
                y_max - xy_pred_fc[:, 1],
                "x",
                color=CMAP[2 + meth_ind],
            )
            rxy_max_dc = frame_points.max(axis=0) - (0, *origin_fc)
            rxy_min_dc = frame_points.min(axis=0) - (0, *origin_fc)
            interp_points_dc = np.linspace(rxy_min_dc, rxy_max_dc, 20)
            xy_interp_dc = predict_dc(interp_points_dc)[:, 1:]
            xy_interp_fc = xy_interp_dc + origin_fc
            ax.plot(
                xy_interp_fc[:, 0],
                y_max - xy_interp_fc[:, 1],
                "--",
                color=CMAP[2 + meth_ind],
                label=f"{method} regression",
            )
        ax.set_title(f"Frame {frame_t}")
        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])

    ax.legend()
    fig.tight_layout()
    return fig


def _plot_acc_hist_by_tranche(
    nested_best_n_results: Sequence[dict[str, RegressionResults]]
) -> Figure:
    n_sub = len(nested_best_n_results)
    fig, axes = plt.subplots(1, n_sub, figsize=[3 * n_sub, 3])
    axes = cast(list[Axes], axes)
    for ax, tranche in zip(axes, nested_best_n_results):
        non_nan_data = []
        non_nan_methods = []
        for method, res in tranche.items():
            non_nans = res["non_nan_inds"]
            if len(res["val_acc"][non_nans]) > 0:
                non_nan_data.append(res["val_acc"])
                non_nan_methods.append(method)
        ax.hist(non_nan_data, label=non_nan_methods)
    axes[-1].legend()
    return fig


def _scatter_accuracy_npoints_by_tranche(
    nested_best_n_results: Sequence[dict[str, RegressionResults]]
) -> Figure:
    n_sub = len(nested_best_n_results)
    fig, axes = plt.subplots(1, n_sub, figsize=[3 * n_sub, 3])
    axes = cast(list[Axes], axes)
    for ax, tranche in zip(axes, nested_best_n_results):
        for method, res in tranche.items():
            non_nans = res["non_nan_inds"]
            _plot_acc_dist(
                res["val_acc"][non_nans],
                res["n_val"][non_nans],
                res["n_train"][non_nans],
                ax,
                label=method,
                title_on=False,
            )
    axes[-1].legend()
    return fig


def _nest_best_n_results(
    methods_ascending: Sequence[str], meth_results: dict[str, RegressionResults]
) -> list[dict[str, RegressionResults]]:
    """Repeat a dictionary of regression results, trimming one each time"""
    n_methods = len(methods_ascending)
    methods_descending = methods_ascending[::-1]
    return [
        {method: meth_results[method] for method in methods_descending[:n]}
        for n in range(1, n_methods + 1)
    ]
