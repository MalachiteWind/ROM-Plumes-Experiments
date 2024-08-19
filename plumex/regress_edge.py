from ara_plumes.models import flatten_edge_points
from scipy.optimize import curve_fit
from scipy.linalg import lstsq

from typing import cast
from typing import List
from typing import Optional
from typing import Callable
from typing import Any
from .types import PlumePoints
from .types import Float2D
from .types import Float1D

import numpy as np
import matplotlib.pyplot as plt


# run after video_digest
def regress_edge(data:dict,
                 train_len: float,
                 n_trials: int,
                 intial_guess:tuple[float,float,float,float],
                 randomize: bool = True,
                 replace: bool = True,
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
    
    n_trials:
        number of trials to run.
    
    initial_guess:
        Initial guess for sinusoid optimizaiton alg.
    
    randomize:
        If True training data is selected at random. If False, first sequential frames
        is used as training data. Remaining frames is test data.
    
    replace:
        Sampling done on with/without replacement.
    
    seed:
        For reproducibility of experiements.
    
    """
    regression_methods = ("linear", "sinusoid")
    meth_results = {
        "top" : {},
        "bot" : {},
    }
    main_accs = []
    # set seed
    np.random.seed(seed=seed)
    center = cast(List[tuple[int,PlumePoints]],data["center"])
    bot = cast(List[tuple[int,PlumePoints]],data["bottom"])
    top = cast(List[tuple[int,PlumePoints]],data["top"])

    top_flat, bot_flat = create_flat_data(center,top,bot)

    ensem_kws = {"trian_len": train_len,
                 "n_trials": n_trials,
                 "method": method,
                 "seed": seed,
                 "replace": replace,
                 "randomize": randomize,
                 "initial_guess": intial_guess,}

    for method in regression_methods:
        meth_results["top"][method] = ensem_regress_edge(
            X=top_flat[:,:2],
            Y=top_flat[:,2],
            **ensem_kws,
        )

        meth_results["bot"][method] = ensem_regress_edge(
            X=bot_flat[:,:2],
            Y=bot_flat[:,2],
            **ensem_kws
        )        
    if method == "sinusoid":
        titles = ["A_opt", "w_opt", "g_opt", "B_opt"]
    elif method == "linear":
        titles = ["bias", "x1", "x2"]
        
    plot_hist(meth_results["top"][method]["coeffs"],titles = titles,big_title="top")
    plot_hist(meth_results["bot"][method]["coeffs"],titles = titles,big_title="bottom")

    return {
        "accs": meth_results
    }

def create_sin_func(awgb):
    A, w, g, B = awgb
    def sin_func(t:float,r:float)->float:
        return A * np.sin(w * r - g * t) + B* r
    return sin_func

def create_lin_func(coef):
    def lin_func(t:float,r:float)->float:
        return coef[0]+t*coef[1]+r*coef[2]
    return lin_func

def create_flat_data(
        center:List[tuple[int,PlumePoints]],
        top:List[tuple[int,PlumePoints]],
        bot:List[tuple[int,PlumePoints]]
)-> tuple[Float2D,Float2D]:
    assert len(center) == len(top)
    assert len(top) == len(bot)

    ## Turn into func: create_flat_data
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

    return top_flattened,bot_flattened


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

def bootstrap(X:Float2D,Y:Float1D,n_trials:int,method:str, seed:int,replace:bool=True,intial_guess:Optional[tuple]=None):
    """
    Apply ensemble bootstrap to data. 

    Parameters:
    ----------
    X: Multivate independent data
    Y: dependent data.
    n_trails: number of trials to run regression
    method: "lstsq" or "sinusoid"
    seed: Reproducability of experiments.
    replace: Sampling with(out) replacement.
    intial_guess: tuple of intiial guess for optimization alg for "sinusoid" method.

    Returns:
    --------
    coef: np.ndarray of learned regression coefficients. 

    """
    np.random.seed(seed=seed)
    coef_data = []
    for _ in range(n_trials):
    
        idxs=np.random.choice(a=len(X),size=len(X),replace=replace)
        X_bootstrap = X[idxs]
        Y_bootstrap = Y[idxs]

        if method == "sinusoid":
            coef = do_sinusoid_regression(X_bootstrap,Y_bootstrap, initial_guess=intial_guess)
        elif method == "lstsq":
            coef = do_lstsq_regression(X_bootstrap, Y_bootstrap)
        coef_data.append(coef)

    
    return np.array(coef_data)


def ensem_regress_edge(
        X:Float2D,
        Y:Float1D,
        train_len:float, 
        n_trials:int,method:str, 
        seed:int,
        replace:bool=True, 
        randomize:bool=True,
        intial_guess:Optional[tuple]=None
)-> dict[str, Any]:
    assert len(X) == len(Y)

    idxs = np.arange(len(X))
    if randomize:
        np.random.shuffle(idxs)
    train_idx, val_idx = idxs[:int(train_len*len(X))], idxs[int(train_len*len(X)):]

    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    coef_bs = bootstrap(
        X=X_train,
        Y=Y_train,
        n_trials=n_trials,
        method=method,
        intial_guess=intial_guess,
        seed=seed,
        replace=replace
    )

    mean_coef = coef_bs.mean(axis=0)

    if method == "linear":
        coef_func = create_lin_func(mean_coef)
    elif method == 'sinusoid':       
        coef_func = create_sin_func(mean_coef) 


    # train acc
    Y_train_pred = coef_func(X_train[:,0],X_train[:,1])
    train_acc = np.linalg.norm(Y_train_pred-Y_train)/np.linalg.norm(Y_train)

    # val_acc
    Y_val_pred = coef_func(X_val[:,0],X_val[:,1])
    val_acc = np.linalg.norm(Y_val_pred-Y_val)/np.linalg.norm(Y_val)

    return {
        "val_acc": val_acc,
        "train_acc": train_acc,
        "coeffs": coef_bs
    }


def plot_hist(param_hist, titles, big_title=None):
    assert len(param_hist.T) == len(titles)

    num_cols = param_hist.shape[1]
    fig, axs = plt.subplots(1, num_cols, figsize=(15, 3))

    param_opt = param_hist.mean(axis=0)

    for i in range(num_cols):
        axs[i].hist(param_hist[:, i], bins=50, density=True, alpha=0.8)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("val")
        axs[i].set_ylabel("Frequency")
        axs[i].axvline(param_opt[i], c="red", linestyle="--")

    if big_title:
        fig.suptitle(big_title, fontsize=16, y=1.05)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  
    
    return fig

