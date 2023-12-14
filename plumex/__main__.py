import pickle
import numpy as np
import mitosis
from pathlib import Path
from plumex import sindy_pipeline 

folder = Path(__file__).parent.resolve()/"good_seed_exp"


# Load data 
pickle_path = Path(__file__).parent.resolve() / "../plume_videos/July_20/video_low_1/gauss_blur_coeff.pkl"

with open(pickle_path, 'rb') as f:
    loaded_arrays = pickle.load(f)

# Params
time_series = loaded_arrays["mean"]
window_length = 4
ensem_thresh = 0.12
ensem_alpha=1e-3
ensem_max_iter=100
poly_degree=3
ensem_num_models=40
ensem_time_points = 100 # ensem_time_points = 100
seed =12

params = [
    mitosis.Parameter("bad_seed", "seed", seed),
    mitosis.Parameter("july_20_low_1", "time_series", time_series, [np]),
    mitosis.Parameter("window_length","window_length",window_length),
    mitosis.Parameter("ensem_thresh","ensem_thresh", ensem_thresh), 
    mitosis.Parameter("ensem_alpha", "ensem_alpha", ensem_alpha),
    mitosis.Parameter("ensem_max_iter", "ensem_max_iter", ensem_max_iter),
    mitosis.Parameter("poly_degree", "poly_degree", 3),
    mitosis.Parameter("ensem_num_models", "ensem_num_models", 40),
    mitosis.Parameter("ensem_time_points", "ensem_time_points", ensem_time_points)
]

# print(type(time_series))
# print(list(time_series))
# print(np.array(list(time_series)))
# print(time_series)cl
# if isinstance(list(time_series), list):
#     print("gotcha!")

# print(time_series.tolist())
# print(np.array(time_series.tolist()))

# # Run experiment
mitosis.run(
    sindy_pipeline,
    params=params,
    debug=True,
    trials_folder=folder
)

# sindy_pipeline.run(
#     time_series=time_series,
#     window_length=window_length,
#     ensem_thresh=ensem_thresh,
#     ensem_alpha=ensem_alpha,
#     ensem_max_iter=ensem_max_iter,
#     ensem_num_models=ensem_num_models,
#     ensem_time_points=ensem_time_points,
#     poly_degree=poly_degree,
#     seed=seed
# )

# Good seeds
# -12
#   - ensem_time_points = 100
#   - crashes with ensem_time_poitns = None

# Not Great seeds
# - 31415

# Seeds that Crash
# - 1234
# - 123
# - 12
#   - crashes when ensem_time_pints = None