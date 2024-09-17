import cv2
import pysindy as ps

data_lookup = {
    "filename": {
        "old-default": "July_20_2023/video_low_1/gauss_blur_coeff.pkl",
        "jan-10-v1": "Jan_10_2024/high/mean_poly_coeff_plume_jan_10_2024"
        "_high_fixed_range_200_img_range_280_2235_orig_center"
        "_1573_1073_gauss_time_window_21_gauss_time_sigma_6_seed_1234.pkl",
        "jan-10-v2": "Jan_10_2024/high/mean_poly_coeff_plume_jan_10_2024"
        "_high_fixed_range_200_img_range_415_2235_orig_center"
        "_1573_1073_gauss_time_window_21_gauss_time_sigma_6_seed_1234.pkl",
        "jan-8-v1": "Jan_8_2024/med/mean_poly_coeff_plume_jan_8_2024_"
        "med_img_0871_fixed_range_90_img_range_200_2200_"
        "orig_center_1572_1078_seed_1234.pkl",
        "jan-8-v2": "Jan_8_2024/med/mean_poly_coeff_plume_jan_8_2024_"
        "med_img_0871_fixed_range_90_img_range_200_2200_"
        "orig_center_1572_1078_num_of_contours_2_seed_1234.pkl",
        "jan-8-v3": "Jan_8_2024/med/mean_poly_coeff_plume_jan_8_2024_"
        "med_img_0871_fixed_range_90_img_range_200_2200_"
        "orig_center_1572_1078_num_of_contours_3_seed_1234.pkl",
        "jan-8-v3-trimmed": "Jan_8_2024/med/mean_poly_coeff_600_1000_plume_jan_8_2024_"
        "med_img_0871_fixed_range_90_img_range_200_2200_orig_center_"
        "1572_1078_num_of_contours_3_seed_1234.pkl",
        "test-fullbuild": "Jan_8_2024/low/IMG_0866.MOV",
        "test-centerpoints": "test_0865_points.pkl",
        "low-862": "Jan_8_2024/low/IMG_0862.MOV",
        "low-864": "Jan_8_2024/low/IMG_0864.MOV",
        "low-865": "Jan_8_2024/low/IMG_0865.MOV",
        "low-866": "Jan_8_2024/low/IMG_0866.MOV",
        "low-867": "Jan_8_2024/low/IMG_0867.MOV",
        "low-868": "Jan_8_2024/low/IMG_0868.MOV",
        "low-869": "Jan_8_2024/low/IMG_0869.MOV",
        "low-913": "Jan_10_2024/low/IMG_0913.MOV",
        "med-914": "Jan_10_2024/med/IMG_0914.MOV",
        "med-916": "Jan_10_2024/med/IMG_0916.MOV",
        "med-917": "Jan_10_2024/med/IMG_0917.MOV",
        "med-871": "Jan_8_2024/med/IMG_0871.MOV",
        "hi-919": "Jan_10_2024/high/IMG_0919.MOV",
        "hi-920": "Jan_10_2024/high/IMG_0920.MOV",
        "blender-nowind": "blender/basic_plume.mp4",
        "blender-wind": "blender/plume_wind.mp4",
        "hi-921": "Jan_10_2024/high/IMG_0921.MOV",
        "low-1": "July_20_2023/low_1.MP4",
        "low-2": "July_20_2023/low_2.MP4",
        "hi-1": "July_20_2023/high_1.MP4",
        "hi-2": "July_20_2023/high_2.MP4",
        "med-0": "July_20_2023/med.MP4",
        "862pts": "step1/390cee.dill",
        "mock": "step1/mock_edges.pkl",
    },
    "fixed_range": {
        "test": 200,
        "862": (0, 240),
        "864": (0, 240),
        "865": (0, 240),
        "866": (0, 380),
        "867": (0, 200),
        "868": (0, 100),
        "869": (0, 150),
        "871": (0, 50),
        "913": (0, 125),
        "914": (0, 200),
        "916": (0, 130),
        "917": (0, 250),
        "919": (0, 200),
        "920": (0, 150),
        "nowind": (0, 1),
        "wind": (0, 1),
        "lo1": (0, 200),
        "hi1": (0, 200),
        "hi2": (0, 200),
    },
    "img_range": {
        "862": (1000, 1450),
        "864": (500, 1650),
        "865": (379, 1650),
        "866": (480, 1250),
        "867": (340, -1),
        "868": (450, 930),
        "869": (450, 2238),
        "871": (300, 2203),
        "913": (250, 1790),
        "914": (410, 2874),
        "916": (400, 1940),
        "917": (400, 1679),
        "919": (500, 2453),
        "920": (300, 2094),
        "nowind": (200, 372),
        "wind": (130, 411),
        "lo1": (900, 1400),
        "hi1": (1000, 1500),
        "hi2": (700, 1100),
    },
    "gauss_space_kws": {
        "off": {"kernel_size": (1, 1), "sigma_x": 0.1, "sigma_y": 0.1},
        "low": {"kernel_size": (15, 15), "sigma_x": 5, "sigma_y": 5},
        "med": {"kernel_size": (45, 45), "sigma_x": 15, "sigma_y": 15},
        "high": {"kernel_size": (301, 301), "sigma_x": 100, "sigma_y": 100},
        "shear": {"kernel_size": (301, 31), "sigma_x": 100, "sigma_y": 10},
    },
    "gauss_time_kws": {
        "off": {"kernel_size": 1, "sigma": 1e-1},
        "low": {"kernel_size": 3, "sigma": 1},
        "med": {"kernel_size": 9, "sigma": 3},
        "high": {"kernel_size": 27, "sigma": 9},
    },
    "contour_kws": {
        "2c": {"num_of_contours": 2},
        "3c": {"num_of_contours": 3},
        "4c": {"num_of_contours": 4},
        "2c-chain": {
            "num_of_contours": 2,
            "find_contour_method": cv2.CHAIN_APPROX_SIMPLE,
        },
        "2c-nothresh": {"num_of_contours": 2, "threshold_method": None},
        "2c-smooth": {"num_of_contours": 2, "contour_smoothing": True},
    },
    "circle_kw": {
        "noscale": {"interior_scale": 1.0},
        "pixeltol": {"atol": 1},
        "no-smooth": {"mean_smoothing": False},
    },
}

sindy_lookup = {
    "diff_params": {
        "test": {
            "diffcls": "SmoothedFiniteDifference",
            "smoother_kws": {"window_length": 4},
        },
        "smoother": {
            "diffcls": "SmoothedFiniteDifference",
            "smoother_kws": {"window_length": 15},
        },
        "x-smooth": {
            "diffcls": "SmoothedFiniteDifference",
            "smoother_kws": {"window_length": 45},
        },
        "xx-smooth": {
            "diffcls": "SmoothedFiniteDifference",
            "smoother_kws": {"window_length": 100, "polyorder": 2},
        },
        "xxx-smooth": {
            "diffcls": "SmoothedFiniteDifference",
            "smoother_kws": {"window_length": 500, "polyorder": 2},
        },
        "tv": {"diffcls": "sindy", "kind": "trend_filtered", "alpha": 1, "order": 0},
        "kalman-autoks": {"diffcls": "sindy", "kind": "kalman", "alpha": "gcv"},
        "kalman": {"diffcls": "sindy", "kind": "kalman", "alpha": 1e-4},
    },
    "reg_mode": {
        "trap-test": ("trap", {"eta": 1e-1}),
        "trap-merge": ("trap", {"eta": 1e-1, "_n_tgts": 3}),
        "trap-merge-dindy": ("trap", {"eta": 1e-1, "_n_tgts": 3, "threshold": 0}),
        "old-default": (
            "poly",
            ({"degree": 3}, ps.STLSQ(threshold=0.12, alpha=1e-3, max_iter=100), 1e-5),
        ),
        "choosy-poly": (
            "poly",
            ({"degree": 3}, ps.STLSQ(threshold=0.12, alpha=1e-3, max_iter=100), None),
        ),
        "choosy-sparser": (
            "poly",
            ({"degree": 3}, ps.STLSQ(threshold=0.12, alpha=1e-1, max_iter=100), None),
        ),
        "poly-semisparse": (
            "poly",
            ({"degree": 3}, ps.STLSQ(threshold=0.12, alpha=1e-2, max_iter=100), None),
        ),
        "quad-default": (
            "poly",
            ({"degree": 2}, ps.STLSQ(threshold=0.05, alpha=1e-2, max_iter=100), None),
        ),
        "c-sparserer": (
            "poly",
            ({"degree": 3}, ps.STLSQ(threshold=0.12, alpha=1e0, max_iter=100), None),
        ),
    },
    "ens_kwargs": {"old-default": {"n_models": 20, "n_subset": None}},
}


hankel_lookup = {
    "hankel_kwargs": {
        "default": {"k": 10},
        "default_80": {"k": 10, "window": 0.8},
        "large_k": {"k": 50},
        "small_k": {"k": 5},
        "very_small": {"k": 5, "window": 0.7},
    },
    "variance": {"default": [0.9, 0.95, 0.99], "95": [0.95]},
    "diff_params": sindy_lookup["diff_params"],
}

regress_lookup = {"r_split": {"default": 50 * 15}}

regress_edge_lookup = {
    "train_len": {"default": 0.8},
    "n_bags": {"default": 1000},
    "initial_guess": {"default": (1, 1, 1, 1)},
}
