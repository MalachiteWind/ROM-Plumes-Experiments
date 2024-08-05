from ara_plumes.models import flatten_edge_points
from scipy.optimize import curve_fit
from scipy.linalg import lstsq

from typing import cast
from typing import List
from .types import PlumePoints
from .types import Float2D
from .types import Float1D

import numpy as np


# run after video_digest
def regress_edge(data:dict,
                 train_len: float,
                 randomize: bool = True,
                 seed: int = 1234
):
    """
    Arguments:
    ----------
    data: 
        dictionary contain top, bottom, and center plumepoints
    
    train_len:
        float value raning from 0 to 1 indicating what percentage of data to 
        to be used for training. Remaing is used for test set. 
    
    randomize:
        If True training data is selected at random. If False, first sequential frames
        is used as training data. Remaining frames is test data.
    
    """
    # set seed
    np.random.seed(seed=seed)
    center = cast(List[tuple[int,PlumePoints]],data["center"])
    bot = cast(List[tuple[int,PlumePoints]],data["bottom"])
    top = cast(List[tuple[int,PlumePoints]],data["top"])

    assert len(center) == len(top)
    assert len(top) == len(bot)

    bot_flattened = []
    top_flattened = []
    for (t,center_pp), (t,bot_pp), (t,top_pp) in zip(center, bot, top):
        
        rad_dist_bot = flatten_edge_points(center_pp,bot_pp)
        rad_dist_top = flatten_edge_points(center_pp,top_pp)

        t_rad_dist_bot = np.hstack((t*np.ones(len(rad_dist_bot),1),rad_dist_bot))
        t_rad_dist_top = np.hstack((t*np.ones(len(rad_dist_top),1),rad_dist_top))

        bot_flattened.append(t_rad_dist_bot)
        top_flattened.append(t_rad_dist_top)
    
    top_flattened = np.concatenate(top_flattened,axis=0)
    bot_flattened = np.concatenate(bot_flattened,axis=0)
    
    # create training data
    indices_top = np.arange(len(top_flattened))
    indices_bot = np.arange(len(bot_flattened))

    if randomize:
        np.random.shuffle(indices_top)
        np.random.shuffle(indices_bot)
    
    top_train_idx = int(len(top_flattened)*train_len)
    bot_train_idx = int(len(bot_flattened)*train_len)

    top_train = cast(Float2D, top_flattened[indices_top[:top_train_idx]])
    bot_train = cast(Float2D,top_flattened[indices_bot[:bot_train_idx]])

    top_test = cast(Float2D,top_flattened[indices_top[top_train_idx:]])
    bot_test = cast(Float2D,bot_flattened[indices_bot[bot_train_idx:]])



def do_sinusoid_regression(
    X: Float2D,
    Y: Float1D,
    initial_guess: tuple[float, float, float, float],
) -> Float1D:
    """
    Return regressed sinusoid coefficients (a,w,g,t) to function
    d(t,r) = a*sin(w*r - g*t) + b*r
    """

    def sinusoid_func(X, A, w, gamma, B):
        t, r = X
        return A * np.sin(w * r - gamma * t) + B * r

    coef, _ = curve_fit(sinusoid_func, (X[:, 0], X[:, 1]), Y, initial_guess)
    return coef

def do_lstsq_regression(X: Float2D, Y: Float1D) -> Float1D:
    "Calculate multivariate lienar regression. Bias term is first returned term"
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    coef, _, _, _ = lstsq(X, Y)
    return coef


n_trials=1000

def ensemble(fit_func,n_trails,):
    ...
