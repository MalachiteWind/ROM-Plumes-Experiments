import sindy_pipeline 
import pickle
import numpy as np

# Load data 
pickle_path = "plume_videos/July_20/video_low_1/gauss_blur_coeff.pkl"

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
# ensem_time_points = 100
ensem_time_points = None

seed =123

# Run experiment
sindy_pipeline.run(
    time_series=time_series,
    window_length=window_length,
    ensem_thresh=ensem_thresh,
    ensem_alpha=ensem_alpha,
    ensem_max_iter=ensem_max_iter,
    ensem_num_models=ensem_num_models,
    ensem_time_points=ensem_time_points,
    poly_degree=poly_degree,
    seed=seed
)

# Good seeds
# -12
#   - ensem_time_points = 100
#   - crashes with ensem_time_poitns = None

# Not Great seeds
# - 31415

# Seeds that Crash
# - 1234
# - 123