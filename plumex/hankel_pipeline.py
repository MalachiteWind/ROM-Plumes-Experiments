# Run Analysis on time-series 

import numpy as  np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Any, Optional
import gen_experiments
from pydmd import DMD


from .types import PolyData, Float1D, Float2D
from .plotting import plot_hankel_variance, plot_dominate_hankel_modes, plot_time_series, plot_data_and_dmd

Kwargs = dict[str, Any]

def run(
        data: Float2D,
        hankel_kwargs: dict,
        variance: list,
        diff_params: Optional[Kwargs]=None,
        normalize: bool=True,
        whitening: bool=False
):
    """
    """
    #################
    # Preprocessing #
    #################
    scalar = StandardScaler()
    if normalize:
        data = scalar.fit_transform(data)
    
    if whitening:
        eigvals, eigvecs = np.linalg.eigh(np.cov(data.T))
        data: PolyData = (data@eigvecs)/np.sqrt(eigvals) 

    metrics = {}
    data_collinearity = np.linalg.cond(data)
    metrics["raw-collinearity"]=data_collinearity

    # Construct Hankel Matrix
    H = _construct_hankel(data, **hankel_kwargs)

    # Compute SVD
    U,S,Vh = np.linalg.svd(H)

    # Check SVD Decomposition
    H_SVD = (U*S)@Vh[:len(S),:]
    if not np.allclose(H,H_SVD):
        print("SVD decomposition issue")

    # Obtain number of modes
    num_of_modes, vars_captured = _get_variances_modes(
        S, vars=variance
    )

    # Get Smooth data
    time_series_kws = {}
    hankel_variance_kws={}
    dominate_modes_kws={}
    if diff_params:
        data_smooth = _smooth_data(
            data,
            diff_params
        )

        H_smooth = _construct_hankel(data_smooth,**hankel_kwargs)
        _,S_smooth,Vh_smooth = np.linalg.svd(H_smooth)
        num_of_modes_smooth, vars_captured_smooth = _get_variances_modes(
            S_smooth, vars=variance
        )

        time_series_kws["smooth_data"] = data_smooth

        hankel_variance_kws["S_norm_smooth"] = S_smooth/np.sum(S_smooth)
        hankel_variance_kws["locs_smooth"] = num_of_modes_smooth
        hankel_variance_kws["vars_smooth"] = vars_captured_smooth

        dominate_modes_kws["V_smooth"] = Vh_smooth.T
        dominate_modes_kws["num_of_modes_smooth"] = num_of_modes_smooth[0]
        dominate_modes_kws["variance_smooth"] = vars_captured_smooth[0]

    #############
    # Exact DMD #
    #############
    dmd = DMD(svd_rank=num_of_modes[0]+1)
    dmd.fit(H)
    H_dmd = dmd.reconstructed_data.real

    if 'window' in hankel_kwargs:
        hankel_kwargs.pop('window')

    data_dmd = _dehankel(H_dmd,**hankel_kwargs)

    data_and_dmd_kws = {}
    if diff_params:
        dmd_smooth = DMD(svd_rank=num_of_modes_smooth[0]+1)
        dmd_smooth.fit(H_smooth)
        H_dmd_smooth = dmd_smooth.reconstructed_data.real
        data_dmd_smooth = _dehankel(H_dmd_smooth,**hankel_kwargs)

        t = range(len(data_dmd_smooth.T))
        data_and_dmd_kws["smooth_data"] = data_smooth[:len(t),:]
        data_and_dmd_kws["smooth_dmd_data"]=data_dmd_smooth.T
        data_and_dmd_kws["var_smooth"] = vars_captured_smooth[0]
        data_and_dmd_kws["svd_rank_smooth"] = num_of_modes_smooth[0]+1

    ############
    # Plotting #
    ############
    # plot time_series
    t = np.arange(len(data))
    feature_names = ['a','b','c']
    plot_time_series(
        t, 
        data=data, 
        feature_names=feature_names,
        **time_series_kws
    )
    plt.show()

    # Plot singular values of H
    S_norm = S/np.sum(S)
    plot_hankel_variance(
        S_norm, 
        locs=num_of_modes, 
        vars=vars_captured,
        **hankel_variance_kws
    )
    plt.show()

    # Plot dominate modes
    plot_dominate_hankel_modes(
        Vh.T, 
        num_of_modes=num_of_modes[0], 
        variance=vars_captured[0],
        **dominate_modes_kws
    )
    plt.show()

    # Plot DMD reconstruction
    t=range(len(data_dmd.T))
    plot_data_and_dmd(
        t=t, data=data[:len(t),:],dmd_data=data_dmd.T,
        feature_names=feature_names,
        var=vars_captured[0],
        svd_rank=num_of_modes[0]+1,
        **data_and_dmd_kws
    )
    plt.show()


    results = {
        "main": 1,
    }

    return results


# Helper Functions
def _construct_hankel(
        A: PolyData,
        k: int=10,
        window: float=0.8,
        dt: int=1  
) -> Float2D:
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
        for each row of Hankel matrix.
        
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

def _dehankel(H,k=10,dt=1):
    """
    Given a Hankel matrix H as a 2-D numpy.ndarray, uses the `delays`
    and `lag` attributes to unravel the data in the Hankel matrix.

    :param H: 2-D Hankel matrix of data.
    :type H: numpy.ndarray
    :return: de-Hankeled (m,) or (n, m) array of data.
    :rtype: numpy.ndarray
    """
    if not isinstance(H, np.ndarray) or H.ndim != 2:
        raise ValueError("Data must be a 2-D numpy array.")

    Hn, Hm = H.shape
    n = int(Hn / k)
    m = int(Hm + ((k - 1) * dt))
    X = np.empty((n, m))
    for i in range(k):
        X[:, i * dt : i * dt + Hm] = H[i * n : (i + 1) * n]

    return np.squeeze(X)


def _get_variances_modes(
        S: Float1D, 
        vars: list[float] = [.9,.95,.99]
) -> tuple[list[int],list[float]]:
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
        A: PolyData,
        diff_params: dict[Kwargs]
) -> PolyData:
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