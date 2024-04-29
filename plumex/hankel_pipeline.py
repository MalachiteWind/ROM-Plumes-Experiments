# Run Analysis on time-series 

# TO DO:
# - Fix plotting to include variance capture for first columnbs of V [Done]
# - Fix mitosis bug [DONE]
# - Add plotting of orginal function [Done]
# - Load dill files and smoothing
# - Add data smoothing option
# - Add check that svd worked well
import numpy as  np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Any, Optional
import gen_experiments

from .types import PolyData, Float2D
from .plotting import plot_hankel_variance, plot_dominate_hankel_modes, plot_time_series

Kwargs = dict[str, Any]

def run(
        data: Float2D,
        hankel_kwargs: dict,
        variance: list,
        diff_params: Optional[Kwargs],
        normalize: bool=True,
        whitening: bool=False
):
    """
    """
    #################
    # Preprocessing #
    #################

    # Need this in order to not throw issue bug in mitosis
    # print("My Flag!")
    scalar = StandardScaler()
    if normalize:
        data = scalar.fit_transform(data)
    
    if whitening:
        eigvals, eigvecs = np.linalg.eigh(np.cov(data.T))
        data: PolyData = (data@eigvecs)/np.sqrt(eigvals) 

    # Construct Hankel Matrix
    H = _construct_hankel(data, **hankel_kwargs)

    # Compute SVD
    U,S,Vh = np.linalg.svd(H)

    # Obtain number of modes
    num_of_modes, vars_captured = _get_variances_modes(
        S, vars=variance
    )

    # Get Smooth data
    data_smooth = _smooth_data(
        data,
        diff_params
    )

    H_smooth = _construct_hankel(data_smooth,**hankel_kwargs)
    _,S_smooth,Vh_smooth = np.linalg.svd(H_smooth)
    num_of_modes_smooth, vars_captured_smooth = _get_variances_modes(
        S_smooth, vars=variance
    )


    ############
    # Plotting #
    ############
    # plot time_series
    t = np.arange(len(data))
    feature_names = ['a','b','c']
    plot_time_series(t, data=data, smooth_data=data_smooth, feature_names=feature_names)
    plt.show()

    # Plot singular values of H
    S_norm = S/np.sum(S)
    plot_hankel_variance(S_norm, locs=num_of_modes, vars=vars_captured)
    plt.show()

    S_smooth_norm = S_smooth/np.sum(S_smooth)
    plot_hankel_variance(S_smooth_norm,locs=num_of_modes_smooth,
                         vars=vars_captured_smooth)
    plt.show()

    # Plot dominate modes
    plot_dominate_hankel_modes(Vh.T, num_of_modes=num_of_modes[0], variance=vars_captured[0])
    plt.show()

    fig4=plot_dominate_hankel_modes(Vh_smooth.T, num_of_modes=num_of_modes_smooth[0], variance=vars_captured_smooth[0])
    plt.show()

    results = {
        "main": 1,
        "metrics": {"results": 1}
    }

    return results


# Helper Functions
def _construct_hankel(
        A: np.ndarray,
        k: int=10,
        window: float=0.8,
        dt: int=1
        
):
    """
    Construct a Hankel Matrix from a given array A (nxd)

    Parameters:
    ----------
    A: np.ndarray
        Array to apply time delay embedding too.

    k: int (default 10)
        Number of time delay embeddings to apply 

    window: float in [0,1] (default 0.8)
        Percentage of data to create window size of embedding. 

    dt: int (default 1)
        time delay time size. I.e. how much to shift data by
        for each row of Hankel.
        
    Returns:
    -------
    H: np.ndarray
        Time delay embedding matrix. (`d * k` x int(n*window))
    """

    n,d = A.shape
    m=int(n*window)
    
    H = []
    l=0
    for _ in range(k):
        H += list(A[l:m+l].T)
        l+=dt
    H=np.array(H)

    
    return H

def _get_variances_modes(S, vars = [.9,.95,.99]):
    """
    Return number of modes (descending) that captures desired variance 
    of data.
    """


    S[::-1].sort()
    var_sums = np.array(
        [np.sum(S[:i]) for i in range(len(S))]
    ) / np.sum(S)

    num_of_modes = []
    vars_captured = []
    for var_i in vars:
        loc_i = np.min(
            np.argwhere(var_sums>=var_i)
        )
        num_of_modes.append(loc_i)
        vars_captured.append(var_sums[loc_i])


    return num_of_modes, vars_captured

def _smooth_data(
        A: np.ndarray,
        diff_params: dict[Kwargs]
):
    """
    Return smoothed data when using selected diff_params from lookup_dict.
    """
    t = np.arange(len(A))
    model = gen_experiments.utils.make_model(
        input_features=None,
        diff_params=diff_params,
        feat_params={"featcls":"polynomial"},
        opt_params={"optcls":"stlsq"},
        dt=1
    )
    model.fit(A,t)
    smooth_A = model.differentiation_method.smoothed_x_

    return smooth_A