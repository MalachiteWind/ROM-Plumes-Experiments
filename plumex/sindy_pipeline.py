# Inputs ?
# - Data matrix
# - Hyperparameters for Ensemble SINDy?

# Outputs
# - Fitted plots
# - Accurcy Score

# Have data generated from plume-pipeline.py? 

# Try simple example on Jake repo

import re
import pickle
from typing import Any, cast, Literal, Optional

import gen_experiments
import numpy as np
from numpy.typing import NBitBase
import matplotlib.pyplot as plt
import pysindy as ps
from matplotlib.figure import Figure
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp

from .types import PolyData, TimeData
from .plotting import plot_smoothing_step, print_diagnostics

name = "sindy-pipeline"

Kwargs = dict[str, Any]
TrapMode = tuple[Literal["trap"], Optional[Kwargs]]
"""Use trapping mode.  Kwargs to TrappingSR3 optimizer"""
PolyMode = tuple[Literal["poly"], Optional[tuple[Kwargs, ps.BaseOptimizer, float]]]
"""Use stabilizing polynomial terms.  Kwargs to PolynomialLibrary, optimizer,
and stabilizing epsilon.
"""

pickle_path = Path(__file__).parent.resolve() / "../plume_videos/July_20/video_low_1/"

def _load_pickle(filename: str) -> PolyData:
    with open(pickle_path / filename, 'rb') as f:
        return pickle.load(f)["mean"]

lookup_dict = {
    "seed": {"bad_seed": 12},
    "datafile": {"old-default": "gauss_blur_coeff.pkl"},
    "diff_params": {
        "test": {"diffcls": "SmoothedFiniteDifference", "smoother_kws": {"window_length": 4}},
        "smoother": {"diffcls": "SmoothedFiniteDifference", "smoother_kws": {"window_length": 15}},
        "kalman-autoks": {"diffcls": "sindy", "kind": "kalman", "alpha": "gcv"},
    },
    "reg_mode": {
        "trap-test": ("trap", {"eta": 1e-1}),
        "old-default": ("poly", (
            {"degree": 3},
            ps.STLSQ(threshold=.12, alpha=1e-3, max_iter=100),
            1e-5
        ))
    },
    "ens_kwargs": {"old-default": {"n_models": 20, "n_subset": None}},
}



def run(
        seed: int,
        datafile: str,
        ens_kwargs: Optional[Kwargs] = None,
        diff_params: Optional[Kwargs] = None,
        normalize=True,
        stabilizing_eps=1e-5,
        reg_mode: TrapMode | PolyMode = ("poly", None)
    ):
    
    """
    Takes in a 3-dimensional timeseries and applies an Ensemble Sindy
    pipeline for model discovery and forward simulating.

    Parameters:
    -----------
    datafile:
        filename in the pickle folder for data to use.  Must hold an array
        of n_time x 3, the coefficients of the fit polynomial.

    ens_kwargs:
        kwargs to EnsembleOptimizer

    diff_params:
        Arguments to construct differentiation method.  Chooses class from
        "diffcls" key, all other keys are sent as kwargs to constructor

    normalize: bool, optional (default True)
        Normalize time_series data by applying StandardScalar() transform from
        sklearn.preprocessing.
    
    reg_mode:
        Regularization mode, either 'trap' or 'poly' and kwargs.  Controls
        whether stability of discovered equation is enforced by Trapping, or by
        adding odd polynomial terms of highest order with negative coefficients.
        A tuple of the name and kwargs, either for trapping optimizer or as
        a tuple of kwargs for polynomial library and STLSQ optimizer.
    
    Returns:
    --------
    err: float
        normalizing L2 error between time_seris (potentially normalized) and 
        solved learned ODE system.
    
    model: pySINDy model

    X_train: np.ndarray
        time_series (potentially normalized by StandardScalar()) 
    
    X_train_sim: np.ndarray
        learned time_series from solved ODE system
    
    scalar: StandardScalar object
        StandardScalar object (potentially) used to normalized time_series data.
    """
    if diff_params is None:
        diff_params = {"diffcls": "finitedifference"}
    feat_params = {"featcls": "Polynomial"}
    if ens_kwargs is None:
        ens_kwargs = {}
    opt_params = {"optcls": "ensemble", "bagging": True} | ens_kwargs
    if reg_mode[0] == "trap":
        feat_params |= {"degree": 2, "include_bias": False}
        opt_init = {} if reg_mode[1] is None else reg_mode[1]
        opt_params |= {"opt": ps.TrappingSR3(**opt_init)}
    elif reg_mode[0] == "poly":
        if reg_mode[1] is None:
            stabilizing_eps = 1e-5
            opt_params |= {"opt": ps.STLSQ()}
        else:
            feat_params |= cast(Kwargs, reg_mode[1][0])
            opt_params |= {"opt": reg_mode[1][1]}
            stabilizing_eps = reg_mode[1][2]
    else:
        raise ValueError("Regularization mode must be either 'poly' or 'trap'")

    np.random.seed(seed=seed)
    time_series = _load_pickle(datafile)
    
    ############################
    # Apply normalized scaling #
    ############################

    t = np.array(range(len(time_series)))
    scalar = StandardScaler()
    if normalize==True:
        time_series: PolyData = scalar.fit_transform(time_series)

    ########################
    # Apply Ensemble SINDy #
    ########################
    feature_names = ['a', 'b', 'c']

    model = gen_experiments.utils._make_model(feature_names, float(t[1]-t[0]), diff_params, feat_params, opt_params)
    model.fit(time_series, t=t)

    plot_smoothing_step(t, time_series, model, feature_names)
    plt.show()  # flush output

    if reg_mode[0] == "poly":
        stab_order = _stabilize_model(model, time_series, stabilizing_eps)

    print_diagnostics(t, time_series, model)

    integrator_kws = {}
    integrator_kws["method"] = "LSODA"

    X_stable_sim = model.simulate(time_series[0], t, integrator_kws=integrator_kws)
    ################
    # Plot Results #
    ################

    m = min(time_series.shape[0],X_stable_sim[0].shape[0])

    fig, axs = plt.subplots(
        1,
        time_series.shape[1], 
        sharex=True, 
        figsize=(15, 4)
    )
    # fig.suptitle("Learned Normalized Coefficients")  
    # Add this line to set the title

    for i in range(time_series.shape[1]):
        if i == time_series.shape[1]-1:
            axs[i].plot(
                t[:m], 
                time_series[:m, i], 
                "k", 
                label="true normalized data"
            )

            axs[i].plot(
                t[:m], 
                X_stable_sim.T[:m, i], 
                "r--", 
                label="model simulation"
            )

            axs[i].legend(loc="best")
        else:
            axs[i].plot(t[:m], time_series[:m, i], "k")
            axs[i].plot(t[:m], X_stable_sim.T[:m, i], "r--")
        axs[i].set(xlabel="t")
        axs[i].set_title("Coeff {}".format(model.feature_names[i]))
    fig.suptitle(
        f"Stabalized: "
        f"eps={stabilizing_eps}, degree={stab_order}" if reg_mode == "poly" else "trap"
    )
    # plt.show()
        
            

    ########################################
    # Solve ODE system w/ pySINDy simulate #
    ########################################
    X_train = time_series
    t_train = t
    x0 = X_train[0]

    print("Solving SINDy system...")
    error_occured = False
    try:
        X_train_sim = model.simulate(x0,t_train)

        ################
        # Plot Results #
        ################
        m = min(X_train.shape[0],X_train_sim.shape[0])

        fig, axs = plt.subplots(
            1,
            X_train.shape[1], 
            sharex=True, 
            figsize=(15, 4)
        )
        # fig.suptitle("Learned Normalized Coefficients") 
        # Add this line to set the title

        for i in range(X_train.shape[1]):
            if i == X_train.shape[1]-1:
                axs[i].plot(
                    t_train[:m], 
                    X_train[:m, i], 
                    "k", 
                    label="true normalized data"
                )
                axs[i].plot(
                    t_train[:m], 
                    X_train_sim[:m, i], 
                    "r--", 
                    label="model simulation"
                )
                axs[i].legend(loc="best")
            else:
                axs[i].plot(
                    t_train[:m], 
                    X_train[:m, i], 
                    "k"
                )
                axs[i].plot(
                    t_train[:m], 
                    X_train_sim[:m, i], 
                    "r--"
                )
            axs[i].set(xlabel="t")
            axs[i].set_title("Coeff {}".format(feature_names[i]))
        fig.suptitle("SINDy simulate")
        plt.show(block=True)
    
    except Exception as e:
        print(f"Numerical solver unstable. Error {e}")
        error_occured = True

    ######################
    # Compute accuracies #
    ######################
    if error_occured is False:
        def L2_error(x_true, x_approx):
                return np.linalg.norm(x_true-x_approx)/np.linalg.norm(x_true)
        
        err = L2_error(X_train[:m].reshape(-1), X_train_sim[:m].reshape(-1))
        print("accuracy: ",1-err)
        print("error: ", err,"\n")
        results = {
            "main": err,
            "error": err,
            "model": model, 
            "X_train": X_train, 
            "X_train_sim": X_train_sim,
            "scalar_transform": scalar
        }
        return results


def get_func_from_SINDy(model, precision=10):
    """
    Takes in learned SINDy model and returns list of learned ode
    equations (str) in correct format to be converted into lambda
    functions.

    Parameters:
    -----------
    model: Trained pySINDy model
    precision: int, optional (default 10)
            Number of decimcal points to include for each
            coefficient in displayed ode equations

    Returns:
    --------
    funcs: list of strings
        List of string itt of RHS of learned ODE system in format
        to be formed into lambda functions.


    """
    # model.feature_library.transform()
    eqns = model.equations(precision=precision)
    funcs = []
    for eqn in eqns:
        # print(eqn)
        # Replace each feature with multiplication "a"-> "*a"
        for feature in model.feature_names:
            eqn = eqn.replace(feature, "*" + feature)

        # Get rid of 1 coefficient for constant term
        eqn = eqn.replace(" 1 ", "")
        eqn = eqn.replace("^", "**")

        # Get rid of all white space
        eqn = re.sub(r"\s+", "", eqn)

        # Return string in correct formatting
        funcs.append(eqn)
    return funcs


def _stabilize_model(
    model: ps.SINDy,
    time_series: PolyData,
    stabilizing_eps: float
) -> int:
    """Mutate fitted polynomial SINDy model to apply stabilization

    A model represents an equation of the form (e.g. 1-d)

    .. math::
        \dot x = ax + b x^2 + \dots + z x^n

    This function mutates it to:

    .. math::
        \dot x = ax + b x^2 + \dots + z x^n - \eps x^p

    where :math:`p` is the next even integer greater than :math:`n`.  This
    helps guarantee that initial value problems of the model will not blow up.
    It works best when :math:`eps` is chosen so that the final term is less than
    :math:`1` over the range of observed data, e.g. :math:`\eps=(\max x)^{-p}`.

    Warning:
        Other properties of the model may be inconsistent, e.g.
        ``model.optimizer.history_``.  But it should predict and simulate
        correctly.

    Args:
        model:
            The SINDy model to modify.  Must have a polynomial library and be
            fitted
        time_series: the data used to fit the model
        stabilizing_eps: Coefficient for stabilizing polynomial terms.

    Returns:
        stabilizing polynomial order
    """
    n_coord = time_series.shape[-1]
    poly_lib = model.feature_library
    poly_degree = poly_lib.degree
    if poly_degree == 0:
        stab_order = poly_degree + 1
    else:
        stab_order = poly_degree + 2

    stabilizing_lib = ps.CustomLibrary(
        [lambda x: x**stab_order],
        [lambda x: f"{x}^{stab_order}"],)
    total_lib = ps.GeneralizedLibrary([poly_lib, stabilizing_lib])
    total_lib.fit(time_series)
    dummy_coef = -stabilizing_eps * np.eye(n_coord)
    model.optimizer.coef_ = np.concatenate(
        (model.optimizer.coef_, dummy_coef), axis=1
    )
    model.feature_library = total_lib
    # SINDy.model is a sklearn Pipeline, whose first step is the feature lib
    model.model.steps[0] = ("features", total_lib)
    return stab_order
