from ara_plumes.models import PLUME
from tqdm import tqdm

import numpy as np

from typing import List
from typing import Any
from .types import Frame
from .types import PlumePoints
from .types import NpFlt
from .types import Float1D
from .types import Float2D


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


def get_L2_acc(X_true:np.ndarray[Any,NpFlt], X_pred:np.ndarray[Any,NpFlt]) -> float:
    """
    Get L2 accuracy between two arrays
    """
    err = np.linalg.norm(X_true-X_pred)/np.linalg.norm(X_true)
    return 1-err