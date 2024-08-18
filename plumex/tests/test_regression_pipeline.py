import numpy as np
import pytest
from ara_plumes.typing import Frame

from ..regression_pipeline import _construct_rxy_f
from ..regression_pipeline import _split_into_train_val
from ..regression_pipeline import get_coef_acc


def test_get_coef_acc():
    coef_time_series = np.array([[1, 2, 3], [3, 2, 1]])
    R = np.array([0, 1, 2])
    f1 = lambda x: x**2 + 2 * x + 3
    f2 = lambda x: 3 * x**2 + 2 * x + 1
    r_x_y_1 = np.vstack((R, R, f1(R))).T
    r_x_y_2 = np.vstack((R, R, f2(R))).T

    train_val_set = [(Frame(1), r_x_y_1), (Frame(2), r_x_y_2)]

    result = get_coef_acc(
        coef_time_series, eval_set_dc=train_val_set, regression_method="poly"
    )
    expected = np.array([1, 1])

    np.testing.assert_array_almost_equal(expected, result)

    # empty
    train_val_set = [(1, r_x_y_1), (2, r_x_y_2[np.array([False for _ in range(3)])])]
    expected = np.array([1, np.nan])
    result = get_coef_acc(coef_time_series, train_val_set, regression_method="poly")
    np.testing.assert_array_almost_equal(expected, result)


def test_split_into_train_val():
    train_i = np.arange(6) + 4
    frame_train = np.vstack((train_i, train_i, train_i)).T

    expected_train = [(1, frame_train), (2, frame_train)]

    val_i = np.arange(4)
    frame_val = np.vstack((val_i, val_i, val_i)).T
    expected_val = [(1, frame_val), (2, frame_val)]

    expected_train, expected_val = expected_val, expected_train

    R = np.arange(10)
    frame_points = np.vstack((R, R, R)).T
    mean_points = [(1, frame_points), (2, frame_points)]
    result_train, result_val = _split_into_train_val(mean_points, r_split=3)

    for i, (t, frame) in enumerate(result_train):
        np.testing.assert_equal(t, expected_train[i][0])
        np.testing.assert_array_almost_equal(frame, expected_train[i][1])

    for i, (t, frame) in enumerate(result_val):
        np.testing.assert_equal(t, expected_val[i][0])
        np.testing.assert_array_almost_equal(frame, expected_val[i][1])


@pytest.mark.parametrize(
    ["coef", "regression_method", "expected"],
    [
        ((1, 2), "linear", np.array([2, 3, 1 * 3 + 2])),
        ((1, 2, 3), "poly", np.array([2, 3, 18])),
        (
            (1, 2, 3),
            "poly_inv_pin",
            # note the inverse form of quadratic, lower branch
            np.array([2, 3, -np.sqrt((3 - 3) / 1 + 2**2 / (4 * 1)) - 2 / (2 * 1)]),
        ),
        (
            (1, 2, 3, 3, 2, 1),
            "poly_para",
            np.array([2, 2**2 + 2 * 2 + 3, 3 * 2**2 + 2 * 2 + 1]),
        ),
    ],
    ids=["linear", "poly", "poly_inv_pin", "poly_para"],
)
def test_construct_f(coef, regression_method, expected):
    rxy = np.array([2, 3, 5])
    predict_dc = _construct_rxy_f(coef, regression_method)
    result = predict_dc(rxy)
    np.testing.assert_array_almost_equal(expected, result)
