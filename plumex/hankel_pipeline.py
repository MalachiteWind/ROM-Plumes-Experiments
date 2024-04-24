# Run Analysis on time-series 

# TO DO:
# - Fix plotting to include variance capture for first columnbs of V
# - Fix mitosis bug [DONE]
# - Add plotting of orginal function [Done]
# - Load dill files and smoothing
import numpy as  np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .types import PolyData, Float1D
from .plotting import plot_hankel_variance, plot_dominate_hankel_modes, plot_time_series

name = "hankel-pipeline"

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
    "datafile": {
         "jan-8-v3-trimmed":"Jan_8_2024/med/mean_poly_coeff_600_1000_plume_jan_8_2024_"\
                             "med_img_0871_fixed_range_90_img_range_200_2200_orig_center_"\
                             "1572_1078_num_of_contours_3_seed_1234.pkl"
    },

    "hankel_kwargs": {
        "default": {"k":10,"window":0.8},
        "large_time": {"k":50}
    },

    "variance": {
        "default": [0.9,0.95,0.99]
    }

}

def run(
        datafile: str,
        hankel_kwargs: dict,
        variance: list,
        normalize: bool=True
):
    # Load Data
    time_series = _load_pickle(datafile)

    scalar = StandardScaler()
    if normalize:
        time_series = scalar.fit_transform(time_series)

    # Construct Hankel Matrix
    H = _construct_hankel(time_series, **hankel_kwargs)

    # Compute SVD
    U,S,Vh = np.linalg.svd(H)

    # Obtain number of modes
    num_of_modes, vars_captured = _get_variances_modes(
        S, vars=variance
    )

    ############
    # Plotting #
    ############
    # plot time_series
    t = np.arange(len(time_series))
    feature_names = ['a','b','c']
    plot_time_series(t, data=time_series, feature_names=feature_names)
    plt.show()

    # Plot singular values of H
    S_norm = S/np.sum(S)
    plot_hankel_variance(S_norm, locs=num_of_modes, vars=vars_captured)
    plt.show()

    # Plot dominate modes
    plot_dominate_hankel_modes(Vh.T, num_of_modes=num_of_modes[0], variance=vars_captured[0])
    plt.show()

    results = {"main": 1}

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
