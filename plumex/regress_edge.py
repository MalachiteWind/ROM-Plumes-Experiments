from ara_plumes.models import flatten_edge_points
from scipy.optimize import curve_fit
from scipy.linalg import lstsq

from typing import cast
from typing import List
from typing import Optional
from typing import Callable
from .types import PlumePoints
from .types import Float2D
from .types import Float1D

import numpy as np


# run after video_digest
def regress_edge(data:dict,
                 train_len: float,
                 n_trials: int,
                 intial_guess:tuple[float,float,float,float],
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
    
    n_trials:
        number of trials to run.
    
    initial_guess:
        Initial guess for sinusoid optimizaiton alg.
    
    randomize:
        If True training data is selected at random. If False, first sequential frames
        is used as training data. Remaining frames is test data.
    
    seed:
        For reproducibility of experiements.
    
    """
    regression_methods = ("linear", "sinusoid")
    meth_results = {}
    main_accs = []
    # set seed
    np.random.seed(seed=seed)
    center = cast(List[tuple[int,PlumePoints]],data["center"])
    bot = cast(List[tuple[int,PlumePoints]],data["bottom"])
    top = cast(List[tuple[int,PlumePoints]],data["top"])

    top_flat, bot_flat = create_flat_data(center,top,bot)

    top_train, top_test = train_test_split(top_flat,train_len,randomize)
    bot_train, bot_test = train_test_split(bot_flat,train_len,randomize)

    for method in regression_methods:
        coef_top = ensemble(X = top_train[:,:2],Y=top_train[:,2],n_trials=n_trials,method=method,intial_guess=intial_guess,seed=seed)
        coef_bot = ensemble(X = bot_train[:,:2],Y=bot_train[:,2],n_trials=n_trials,method=method,intial_guess=intial_guess,seed=seed)

        mean_coef_top = coef_top.mean(axis=0)
        mean_coef_bot = coef_bot.mean(axis=0)

        if method == "linear":
            func_top = create_lin_func(mean_coef_top)
            func_bot = create_lin_func(mean_coef_bot)
        elif method == 'sinusoid':       
            func_top = create_sin_func(mean_coef_top)
            func_bot = create_sin_func(mean_coef_bot) 
    
        top_train_acc, top_test_acc = train_test_acc(top_train,top_test,func_top)
        bot_train_acc, bot_test_acc = train_test_acc(bot_train,bot_test,func_bot)

        top = {}


def train_test_acc(top_bot_train,top_bot_test, pred_func):
    X_train = top_bot_train[:,:2]
    Y_train = top_bot_train[:,2]

    X_test = top_bot_test[:,:2]
    Y_test = top_bot_test[:,2]

    Y_train_pred = pred_func(X_train[:,0],X_train[:,1])
    Y_test_pred = pred_func(X_test[:,0], X_test[:,1])

    train_acc = np.linalg.norm(Y_train-Y_train_pred)/np.linalg.norm(Y_train)
    test_acc = np.linalg.norm(Y_test - Y_test_pred)/np.linalg.norm(Y_test)

    return train_acc, test_acc

def create_sin_func(awgb):
    A, w, g, B = awgb
    def sin_func(t:float,r:float)->float:
        return A * np.sin(w * r - g * t) + B* r
    return sin_func

def create_lin_func(coef):
    def lin_func(t:float,r:float)->float:
        return coef[0]+t*coef[1]+r*coef[2]
    return lin_func

def train_test_split(data:Float2D, train_len:float, randomize:bool):
    idxs = np.arange(len(data))
    
    if randomize:
        np.random.shuffle(idxs)
    
    train_idxs = idxs[:int(train_len*len(idxs))]
    test_idxs = idxs[int(train_len*len(idxs)):]

    train_data = data[train_idxs]
    test_data = data[test_idxs]

    return train_data, test_data

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

def ensemble(X:Float2D,Y:Float1D,n_trials:int,method:str, seed:int,replace:bool=True,intial_guess:Optional[tuple]=None):
    """
    Apply ensemble bootstrap to data. 

    Parameters:
    ----------
    X: Multivate independent data
    Y: dependent data.
    n_trails: number of trials to run regression
    method: "lstsq" or "sinusoid"
    seed: Reproducability of experiments.
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


def ensem_regress_edge(X:Float2D,Y:Float1D,n_trials:int,method:str, seed:int,replace:bool=True,intial_guess:Optional[tuple]=None):
    coef = ensemble(X = X,Y=Y,n_trials=n_trials,method=method,intial_guess=intial_guess,seed=seed)

    mean_coef = coef.mean(axis=0)

    if method == "linear":
        coef_func = create_lin_func(mean_coef)
    elif method == 'sinusoid':       
        coef_func = create_sin_func(mean_coef) 

    train_acc, test_acc = train_test_acc(top_train,top_test,func_top)


    ...

