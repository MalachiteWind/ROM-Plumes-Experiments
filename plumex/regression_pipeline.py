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


def regress_video(
        mean_points:  List[tuple[Frame, PlumePoints]],
        x_split: int,
        regression_method: str,
        poly_deg: int = 2,
        decenter: Optional[tuple[int,int]] = None
):
    """
    Apply regression test on train and val data.
    """
    train_set, val_set = split_into_train_val(mean_points, x_split)

    coef_time_series = PLUME.regress_multiframe_mean(
        mean_points=train_set,
        regression_method=regression_method,
        poly_deg=poly_deg,
        decenter=decenter
    )
    train_acc = get_coef_acc(coef_time_series, train_set, regression_method)
    val_acc = get_coef_acc(coef_time_series, val_set, regression_method)

    return {"main": val_acc, "train_acc": train_acc, "val_acc": val_acc}


def split_into_train_val(
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


def _get_L2_acc(X_true:np.ndarray[Any,NpFlt], X_pred:np.ndarray[Any,NpFlt]) -> float:
    """
    Get L2 accuracy between two arrays
    """
    err = np.linalg.norm(X_true-X_pred)/np.linalg.norm(X_true)
    return 1-err

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
) -> tuple[Float1D,Float1D] | tuple[Float2D,Float2D]:
    """
    Acquire true coordinates from Frame points and predicted points from 
    learned function based on regression_method used.
    """
    if regression_method == "poly" or regression_method == "linear":
        y_pred = func(r_x_y[:,1])
        y_true = r_x_y[:,2]

    if regression_method == "poly_inv":
        y_pred = func(r_x_y[:,2])
        y_true = r_x_y[:,1]

    if regression_method == "poly_para":
        y_pred = func(r_x_y[:,0]).T
        y_true = r_x_y[:,1:]
    
    return y_true, y_pred



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

        y_true, y_pred = _get_true_pred(f,r_x_y, regression_method)
        accs[i] = _get_L2_acc(y_true,y_pred) 
    
    return accs




