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
from warnings import warn

import gen_experiments
import numpy as np
from numpy.typing import NBitBase
import matplotlib.pyplot as plt
import pysindy as ps
from matplotlib.figure import Figure
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp

from .types import PolyData, Float1D
from .plotting import plot_smoothing_step, print_diagnostics, plot_predictions, plot_simulation

name = "sindy-pipeline"

Kwargs = dict[str, Any]
TrapMode = tuple[Literal["trap"], Optional[Kwargs]]
"""Use trapping mode.  Kwargs to TrappingSR3 optimizer"""
PolyMode = tuple[Literal["poly"], Optional[tuple[Kwargs, ps.BaseOptimizer, float]]]
"""Use stabilizing polynomial terms.  Kwargs to PolynomialLibrary, optimizer,
and stabilizing epsilon.
"""

pickle_path = Path(__file__).parent.resolve() / "../plume_videos/"

def _load_pickle(filename: str) -> PolyData:
    with open(pickle_path / filename, 'rb') as f:
        data_file = pickle.load(f)
        if isinstance(data_file,np.ndarray) is True:
            return data_file
        elif isinstance(data_file,dict) is True:   
            return data_file["mean"]
        else:
            raise(ValueError("Datafile must be an array or dict with key 'mean'."))

lookup_dict = {
    "seed": {"bad_seed": 12},
    "datafile": {"old-default": "July_20_2023/video_low_1/gauss_blur_coeff.pkl",
                 "jan-10-v1":"Jan_10_2024/high/mean_poly_coeff_plume_jan_10_2024"\
                             "_high_fixed_range_200_img_range_280_2235_orig_center"\
                             "_1573_1073_gauss_time_window_21_gauss_time_sigma_6_seed_1234.pkl",
                 "jan-10-v2":"Jan_10_2024/high/mean_poly_coeff_plume_jan_10_2024"\
                             "_high_fixed_range_200_img_range_415_2235_orig_center"\
                             "_1573_1073_gauss_time_window_21_gauss_time_sigma_6_seed_1234.pkl",
                  "jan-8-v1":"Jan_8_2024/med/mean_poly_coeff_plume_jan_8_2024_"\
                             "med_img_0871_fixed_range_90_img_range_200_2200_"\
                             "orig_center_1572_1078_seed_1234.pkl",
                  "jan-8-v2":"Jan_8_2024/med/mean_poly_coeff_plume_jan_8_2024_"\
                             "med_img_0871_fixed_range_90_img_range_200_2200_"\
                             "orig_center_1572_1078_num_of_contours_2_seed_1234.pkl",
                  "jan-8-v3":"Jan_8_2024/med/mean_poly_coeff_plume_jan_8_2024_"\
                             "med_img_0871_fixed_range_90_img_range_200_2200_"\
                             "orig_center_1572_1078_num_of_contours_3_seed_1234.pkl",
          "jan-8-v3-trimmed":"Jan_8_2024/med/mean_poly_coeff_600_1000_plume_jan_8_2024_"\
                             "med_img_0871_fixed_range_90_img_range_200_2200_orig_center_"\
                             "1572_1078_num_of_contours_3_seed_1234.pkl"},
    "diff_params": {
        "test": {"diffcls": "SmoothedFiniteDifference", "smoother_kws": {"window_length": 4}},
        "smoother": {"diffcls": "SmoothedFiniteDifference", "smoother_kws": {"window_length": 15}},
        "x-smooth": {"diffcls": "SmoothedFiniteDifference", "smoother_kws": {"window_length": 45}},
        "xx-smooth": {"diffcls": "SmoothedFiniteDifference", "smoother_kws": {"window_length": 100, "polyorder": 2}},
        "xxx-smooth": {"diffcls": "SmoothedFiniteDifference", "smoother_kws": {"window_length": 500, "polyorder": 2}},
        "tv": {"diffcls": "sindy", "kind": "trend_filtered", "alpha": 1, "order": 0},
        "kalman-autoks": {"diffcls": "sindy", "kind": "kalman", "alpha": "gcv"},
        "kalman": {"diffcls": "sindy", "kind": "kalman", "alpha": 1e-4},
    },
    "reg_mode": {
        "trap-test": ("trap", {"eta": 1e-1}),
        "old-default": ("poly", (
            {"degree": 3},
            ps.STLSQ(threshold=.12, alpha=1e-3, max_iter=100),
            1e-5
        )),
        "choosy-poly": ("poly", (
            {"degree": 3},
            ps.STLSQ(threshold=.12, alpha=1e-3, max_iter=100),
            None
        )),
        "choosy-sparser": ("poly", (
            {"degree": 3},
            ps.STLSQ(threshold=.12, alpha=1e-1, max_iter=100),
            None
        )),
        "poly-semisparse": ("poly", (
            {"degree": 3},
            ps.STLSQ(threshold=.12, alpha=1e-2, max_iter=100),
            None
        )),
        "quad-default": ("poly", (
            {"degree": 2},
            ps.STLSQ(threshold=.05, alpha=1e-2, max_iter=100),
            None
        )),
        "c-sparserer": ("poly", (
            {"degree": 3},
            ps.STLSQ(threshold=.12, alpha=1e0, max_iter=100),
            None
        )),
    },
    "ens_kwargs": {"old-default": {"n_models": 20, "n_subset": None}},
    "feat_params":{
        "poly": {"featcls": "Polynomial"},
        "fourier": {"featcls": "Fourier"},
        "weak-pde": {"featcls": "Weak"},
        "pde": {"featcls": "PDE"}
    }
}



def run(
        seed: int,
        datafile: str,
        ens_kwargs: Optional[Kwargs] = None,
        diff_params: Optional[Kwargs] = None,
        feat_params: Optional[Kwargs] = None,
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
    if feat_params is None:
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
    scaler = StandardScaler()
    if normalize==True:
        time_series: PolyData = scaler.fit_transform(time_series)

    metrics = {}
    data_collinearity = np.linalg.cond(time_series)
    print("Collinearity (condition number) of data: ", data_collinearity)
    metrics["raw-collinearity"] = data_collinearity

    ########################
    # Apply Ensemble SINDy #
    ########################
    feature_names = ['a', 'b', 'c']

    model = gen_experiments.utils.make_model(
        feature_names, float(t[1]-t[0]), diff_params, feat_params, opt_params
    )
    model.fit(time_series, t=t)
    x_smooth = model.differentiation_method.smoothed_x_
    plot_smoothing_step(t, time_series, x_smooth, feature_names)
    plt.show()  # flush output

    x_dot_est = model.differentiation_method(time_series, t)
    smooth_inf_norm = np.linalg.norm(x_dot_est, float("inf"), axis=0)
    smooth_2_norm = np.linalg.norm(x_dot_est, 2, axis=0)
    print(r"∞-norms of estimated ẋ: ", smooth_inf_norm, flush=True)
    print(r"2-norms of estimated ẋ: ", smooth_2_norm, flush=True)
    smooth_collinearity = np.linalg.cond(x_smooth)
    print("Collinearity (condition number) of smoothed data: ", smooth_collinearity)
    metrics["smooth-collinearity"] = smooth_collinearity
    metrics |= {"smooth-inf-norm": smooth_inf_norm, "smooth-2-norm": smooth_2_norm}

    print_diagnostics(t, model, precision=8)
    x_dot_pred = model.predict(x_smooth)
    plot_predictions(t, np.asarray(x_dot_est), np.asarray(x_dot_pred), feature_names)
    plt.show()
    metrics["pred-accuracy"] = model.score(x_smooth, t, x_dot_est)
    print("Prediction Accuracy: ", metrics["pred-accuracy"])
    print("Prediction Relative Error: ", 1-metrics["pred-accuracy"])

    results = {
        "main": metrics["pred-accuracy"],
        "metrics": metrics,
        "model": model,
        "X_train": time_series,
        "scalar_transform": scaler
    }

    ################
    # Integration  #
    ################
    if reg_mode[0] == "poly":
        stab_order = _stabilize_model(model, time_series, stabilizing_eps)
        print("Stabilized model:")
        model.print(precision=8)
    integrator_kws = {}
    integrator_kws["method"] = "LSODA"
    try:
        X_stable_sim = model.simulate(
            x_smooth[0], t, integrator_kws=integrator_kws, integrator='odeint'
        )
        integration_success = True
    except Exception as exc:
        warn(f"Simulation error: {exc.args[0]}", RuntimeWarning)
        integration_success = False
    if integration_success:
        X_stable_sim = cast(PolyData, X_stable_sim)  # type: ignore
        ind_sim = min(x_smooth.shape[0], X_stable_sim.shape[0])
        if reg_mode[0] == "poly":
            title = f"Stabilized: Poly, eps={stabilizing_eps}, degree={stab_order}"  # type: ignore
        else:
            title = "Stabilized: Trapping"
        plot_simulation(
            t, x_smooth, X_stable_sim, feat_names=model.feature_names, title=title  # type: ignore
        )
        plt.show()
        def L2_error(x_true, x_approx):
                return np.linalg.norm(x_true-x_approx)/np.linalg.norm(x_true)
        sim_err = L2_error(x_smooth[:ind_sim], X_stable_sim[:ind_sim])
        metrics["sim-err"] = sim_err
        results["X_train_sim"] = X_stable_sim
        print(f"Simulation Accuracy: {1-sim_err}")
        print(f"Simulation Relative Error: {sim_err}")


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
    stabilizing_eps: Optional[float]
) -> int:
    r"""Mutate fitted polynomial SINDy model to apply stabilization

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
        stabilizing_eps: Coefficient for stabilizing polynomial terms.  If None,
            calcuate it as 1 / (2*max), where max is calculated along each axis.
            This means that the stabilizing term won't activate except when
            substantially beyond the data range.

    Returns:
        stabilizing polynomial order
    """
    n_coord = time_series.shape[-1]
    poly_lib = model.feature_library
    poly_degree = cast(int, poly_lib.degree)
    if poly_degree % 2 == 0:
        stab_order = poly_degree + 1
    else:
        stab_order = poly_degree + 2

    stabilizing_lib = ps.CustomLibrary(
        [lambda x: x**stab_order],
        [lambda x: f"{x}^{stab_order}"],)
    total_lib = ps.GeneralizedLibrary([poly_lib, stabilizing_lib])
    total_lib.fit(time_series)
    if stabilizing_eps is None:
        coef: Float1D = 1 / (2 * np.max(time_series, axis=0))
    else:
        coef: Float1D = stabilizing_eps * np.ones(time_series.shape[1])
    dummy_coef = -coef * np.eye(n_coord)
    model.optimizer.coef_ = np.concatenate(
        (model.optimizer.coef_, dummy_coef), axis=1
    )
    model.feature_library = total_lib
    # SINDy.model is a sklearn Pipeline, whose first step is the feature lib
    model.model.steps[0] = ("features", total_lib)
    return stab_order
