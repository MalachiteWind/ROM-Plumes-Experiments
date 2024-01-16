# Inputs ?
# - Data matrix
# - Hyperparameters for Ensemble SINDy?

# Outputs
# - Fitted plots
# - Accurcy Score

# Have data generated from plume-pipeline.py? 

# Try simple example on Jake repo

import re
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pysindy as ps
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp

name = "sindy-pipeline"


pickle_path = Path(__file__).parent.resolve() / "../plume_videos/July_20/video_low_1/gauss_blur_coeff.pkl"

with open(pickle_path, 'rb') as f:
    loaded_arrays = pickle.load(f)

lookup_dict = {
    "seed": {"bad_seed": 12},
    "time_series": {"july_20_low_1": loaded_arrays["mean"]},
    "window_length": {"window_length": 4},
    "ensem_thresh": {"ensem_thresh": 0.12},
    "ensem_alpha": {"ensem_alpha": 1e-3},
    "ensem_max_iter": {"ensem_max_iter": 100},
    "poly_degree": {"poly_degree": 3},
    "ensem_num_models": {"ensem_num_models": 40} ,
    "ensem_time_points": {"ensem_time_points": 100}
}



def run(
        seed,
        time_series,
        window_length,
        ensem_thresh,
        ensem_alpha,
        ensem_max_iter,
        ensem_num_models=20,
        ensem_time_points=None,
        normalize=True,
        poly_degree=2,
        stabalzing_eps=1e-5,
    ):
    
    """
    Takes in a 3-dimensional timeseries and applies an Ensemble Sindy
    pipeline for model discovery and forward simulating.

    Parameters:
    -----------
    times_series: np.ndarray 
        Timeseries of coefficients (a,b,c) learned from concentric circle
        pipeline.
    
    window_length: int
        window length used in pySINDy.SmoothedFiniteDifference() method---used 
        to smooth and predict derivatives of time_series data.
    
    ensem_thresh: float
        Thresholding applied to model discovery in ensembling method for pySINDy.
    
    ensem_alpha: float
    
    ensem_max_iter: int
        number of interation to use before stoping base_optimizer (STLSQs).
    
    ensem_n_models : int, optional (default 20)
        Number of models to generate via ensemble_optimizer. 

    ensem_n_subset : int, optional (default None is 0.6*len(time base), if bagging=True)
        Number of time points to use for ensemble_optimizer.    
        NOTE: If bagging=False in ensemble_optimizer, then default is len(time base).

    normalize: bool, optional (default True)
        Normalize time_series data by applying StandardScalar() transform from
        sklearn.preprocessing.
    
    poly_degree: int, optional (default 2)
        Maximum polynomial order ot serach over in class of functions for 
        PolynomailLibrary in pySINDy.
    
    stabalizing_eps: float, optional (default 1e-5)
        Coefficent of higher order (odd) term added into discovered ODE system 
        from pySINDy. Adds stability to numerical integrator of system
        (default scipy.integrate.solve_ivp)
    
    
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

    np.random.seed(seed=seed)

    # Check if time_series is numpy array
    if isinstance(time_series, list):
        print("time_series is list!")
        time_series = np.array(time_series)
    
    ############################
    # Apply normalized scaling #
    ############################

    t = np.array([i for i in range(len(time_series))])
    scalar = StandardScaler()
    if normalize==True:
        # print("normalize:", normalize)
        time_series = scalar.fit_transform(time_series)

    ########################
    # Apply Ensemble SINDy #
    ########################
    feature_names = ['a', 'b', 'c']
    
    smoothed_fd = ps.SmoothedFiniteDifference(
        smoother_kws={'window_length': window_length}
    )

    base_optimizer = ps.STLSQ(
        threshold=ensem_thresh,
        alpha=ensem_alpha,
        max_iter=ensem_max_iter
    )

        
    ensemble_optimizer=ps.optimizers.base.EnsembleOptimizer(
        base_optimizer,
        bagging=True,
        n_models=ensem_num_models,
        n_subset=ensem_time_points   
    )

    poly_lib = ps.PolynomialLibrary(degree=poly_degree)
    model = ps.SINDy(
        feature_names=feature_names,
        optimizer=ensemble_optimizer,
        differentiation_method=smoothed_fd,
        feature_library=poly_lib
    )

    model.fit(time_series, t=t)

    print(
        "window_length: {}, thresh: {}, alpha: {},  max iter: {}, stabalzing eps: {}"
        .format(
            window_length, 
            ensem_thresh, 
            ensem_alpha, 
            ensem_max_iter, 
            stabalzing_eps
        )
    )

    model.print()

    #######################################
    # Solve ODE system w/ Stabalzing Term #
    #######################################
    # Get function in correct format to be evaluated 
    funcs = get_func_from_SINDy(model=model)

    # Define Right-hand side of ODE system 
    a_dot = lambda a,b,c: eval(funcs[0])
    b_dot = lambda a,b,c: eval(funcs[1])
    c_dot = lambda a,b,c: eval(funcs[2])

    # Get degree of feature library
    poly_degree = model.feature_library.get_params()['degree']
    stabalizing_degree = poly_degree+1

    # Ensure stabalizing degree is odd
    if stabalizing_degree % 2 == 0:
        stabalizing_degree += 1 

    # Define new ODE system 
    def ode_sys(t, y,
                a_dot=a_dot,
                b_dot=b_dot,
                c_dot=c_dot, 
                stabalizing_deg=stabalizing_degree,
                eps = 1e-5):
        a,b,c = y
        da = a_dot(a,b,c) - eps*a**stabalizing_deg
        db = b_dot(a,b,c) - eps*b**stabalizing_deg
        dc = c_dot(a,b,c) - eps*c**stabalizing_deg
        rhs = [da,db,dc]
        return rhs

    # Initialize integrator keywords for solve_ivp to
    # replicate the odeint defaults
    integrator_keywords = {}
    integrator_keywords["rtol"] = 1e-12 # 1e-6
    integrator_keywords["method"] = "LSODA"
    integrator_keywords["method"] = "RK45" # Try as opposed to LSODA
    integrator_keywords["atol"] = 1e-12 # 1e-6

    t_solve = t

    y0 = time_series[0]

    params = (a_dot,b_dot,c_dot,stabalizing_degree,stabalzing_eps)

    print(f"Solving SINDy system with eps = {stabalzing_eps}...")

    error_occured = False
    try:
        X_solved = solve_ivp(
            ode_sys,
            t_span=(t_solve[0],t_solve[-1]),
            y0=y0, 
            t_eval=t_solve,
            args=params,
            # **integrator_keywords
        )
    except Exception as e:
        print(f"Numerical Solver unstable. Error: {e}")
        error_occured = True
    
    # if error_occured is True:

    
    ################
    # Plot Results #
    ################
    if error_occured is False:
        X_stable_sim = X_solved.y

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
                    t_solve[:m], 
                    time_series[:m, i], 
                    "k", 
                    label="true normalized data"
                )

                axs[i].plot(
                    t_solve[:m], 
                    X_stable_sim.T[:m, i], 
                    "r--", 
                    label="model simulation"
                )

                axs[i].legend(loc="best")
            else:
                axs[i].plot(t_solve[:m], time_series[:m, i], "k")
                axs[i].plot(t_solve[:m], X_stable_sim.T[:m, i], "r--")
            axs[i].set(xlabel="t")
            axs[i].set_title("Coeff {}".format(model.feature_names[i]))
        fig.suptitle(
            f"Stabalized: eps={stabalzing_eps}, degree={stabalizing_degree}"
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
