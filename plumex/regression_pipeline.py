from ara_plumes.models import PLUME
from tqdm import tqdm

import numpy as np

from typing import List
from typing import Any
from typing import Optional
from typing import Callable
from typing import Literal
from .types import Frame
from .types import PlumePoints
from .types import NpFlt
from .types import Float1D
from .types import Float2D


def multi_regress_centerline(
    data: dict[List[tuple[Frame, PlumePoints]]],
    x_split: int,
    poly_deg: int = 2,
    decenter: Optional[tuple[int,int]] = None
) -> dict[str, Any]:
    """Mitosis experiment to compare regress_centerline across methods

    Arguments same as regress_centerline, without 'regression_method'

    Returns:
        Experiment results as regress_centerline, but main metric is name of
        best method, "data" is data from best method, and "accs" is dictionary
        of method name to its validation and train accuracies by frame
    """
    regression_methods = ("linear", "poly", "poly_inv", "poly_para")
    meth_results = {}
    for method in regression_methods:
        meth_results[method] = regress_centerline(
            data, x_split, method, poly_deg, decenter
        )
    val_accs = [(method, result.pop("main")) for method, result in meth_results.items()]
    val_accs.sort(key=lambda tup: tup[1])
    best_method = val_accs[-1][0]
    best_data = meth_results[best_method]["data"],
    n_frames = -1
    for result in meth_results:
        result.pop("data")
        n_frames = result.pop("n_frames")

    return {
        "main": best_method,
        "data": best_data,
        "n_frames": n_frames,
        "accs": meth_results
    }

def regress_centerline(
    data: dict[List[tuple[Frame, PlumePoints]]],
    x_split: int,
    regression_method: str,
    poly_deg: int = 2,
    decenter: Optional[tuple[int,int]] = None
) -> dict[str, Any]:
    """Mitosis experiment to fit mean path of plume points
    
    Args:
        data: output of video_digest step, a dictionary of center, top, and
            bottom points for each frame of a video
        x_split: the pixel coordinate of boundary between validation (left)
            and training (right) data
        regression_method: method for drawing curve through points as
            understood by PLUME.regress_multiframe_mean
        poly_deg: polynomial degree of curve used in regression_method
        decenter: Shift to apply to plume coordinates before fitting curve

    Returns:
        Experiment results of train and validation accuracy.  "data" key
        is of shape (number of timepoints, number of coefficients in
        regression_method). "main" metric is average validation accuracy
    """
    mean_points = data["center"]
    train_set, val_set = _split_into_train_val(mean_points, x_split)

    coef_time_series = PLUME.regress_multiframe_mean(
        mean_points=train_set,
        regression_method=regression_method,
        poly_deg=poly_deg,
        decenter=decenter
    )
    train_acc = get_coef_acc(coef_time_series, train_set, regression_method)
    val_acc = get_coef_acc(coef_time_series, val_set, regression_method)

    return {
        "main": val_acc.mean(),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "data": coef_time_series,
        "n_frames": len(mean_points)
    }


def _split_into_train_val(
        mean_points: List[tuple[Frame, PlumePoints]],
        x_split: int
) -> tuple[
    List[tuple[Frame, PlumePoints]],
    List[tuple[Frame, PlumePoints]]
]:
    """
    Splits mean_points into training and validation sets based
    on coordinate values along the x-axis.

    Parameters:
    ----------
    mean_points:
        Mean points returned from PLUME.train().
    
    x_split:
        Value to determine where the split of data occurs on the frame.
    
    Returns:
    -------
    train_set:
        Frame points associated with x-coordinate values greater than 
        or equal to x_split.
        
    val_set:
        Frame points associated with x-coordinate values less than 
        x_split.
    """
    train_set = []
    val_set = []
    for (t, frame_points) in tqdm(mean_points):

        mask = frame_points[:,1] >= x_split

        train_set.append((t, frame_points[mask]))
        val_set.append((t, frame_points[~mask]))

    return train_set, val_set


def _construct_f(coef: Float1D, regression_method:Optional[str]=None) -> Callable[[float],float] | Callable[[float],Float1D]:
    """construct function f based on coefficients and regression_method

    Parameters:
    ----------
    coef:
        array of poly coefficients in descending degree order.
    
    regression_method:
        makes function parametric if regression_method = "poly_para". 
    
    Returns:
    --------
        f:
            Callable function that takes float as argument.
    """
    if regression_method == "poly_para":
        mid_index = len(coef)//2
        f1 = np.polynomial.Polynomial(coef[:mid_index][::-1])
        f2 = np.polynomial.Polynomial(coef[mid_index:][::-1])
        def f(x):
            return np.array([f1(x), f2(x)])
    else: 
        f = np.polynomial.Polynomial(coef[::-1])
    return f

def _get_true_pred(
        func: Callable[[float],float] | Callable[[float],Float1D], 
        r_x_y: PlumePoints, 
        regression_method: str
) -> tuple[Float2D,Float2D]:
    """
    Vectorize approximation function ``func``, map correct inputs from `r_x_y`, and
    extract true values from `r_x_y` based on regression_method used.
    """
    if regression_method == "poly" or regression_method == "linear":
        y_pred = func(r_x_y[:,1])
        xy_pred = np.vstack((r_x_y[:,1],y_pred)).T
        xy_true = r_x_y[:,1:]

    if regression_method == "poly_inv":
        y_true = r_x_y[:,2]
        x_pred = func(y_true)
        xy_pred = np.vstack((x_pred,y_true)).T
        xy_true = r_x_y[:,1:]

    if regression_method == "poly_para":
        xy_pred = func(r_x_y[:,0]).T
        xy_true = r_x_y[:,1:]
    
    return xy_true, xy_pred



def get_coef_acc(
        coef_time_series:np.ndarray[Any, NpFlt],
        train_val_set: List[tuple[Frame,PlumePoints]], 
        regression_method: str
) -> Float1D:
    """Get the L2 accuracy of learned coefficients on PlumePoints"""
    if len(coef_time_series) != len(train_val_set):
        raise TypeError(f"length of arrays must match")
    
    n_frames = len(coef_time_series)
    accs = np.zeros(n_frames)
    for i in range(n_frames):
        coef_i = coef_time_series[i]

        f = _construct_f(coef_i,regression_method)
        _, r_x_y = train_val_set[i]

        xy_true, xy_pred = _get_true_pred(f,r_x_y, regression_method)
        accs[i] = 1 - np.linalg.norm(xy_true-xy_pred)/np.linalg.norm(xy_true)
    
    return accs




