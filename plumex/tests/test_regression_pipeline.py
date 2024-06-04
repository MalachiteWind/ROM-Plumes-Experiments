from ..regression_pipeline import get_coef_acc
from ..regression_pipeline import _split_into_train_val
from ..regression_pipeline import _get_L2_acc
from ..regression_pipeline import _construct_f
from ..regression_pipeline import _get_true_pred

import numpy as np

def test_get_coef_acc():
    coef_time_series = np.array([
        [1,2,3],
        [3,2,1]
    ])
    R = np.array([0,1,2])
    f1 = lambda x: x**2 + 2*x + 3
    f2 = lambda x: 3*x**2 + 2*x + 1
    r_x_y_1 = np.vstack((R,R,f1(R))).T
    r_x_y_2 = np.vstack((R,R,f2(R))).T

    train_val_set = [
        (1,r_x_y_1),
        (2,r_x_y_2)
    ]

    result = get_coef_acc(coef_time_series,train_val_set=train_val_set,regression_method='poly')
    expected = np.array([1,1])

    np.testing.assert_array_almost_equal(expected,result)

def test_split_into_train_val():
    train_i = np.arange(6)+4
    frame_train = np.vstack((train_i, train_i, train_i)).T

    expected_train = [(1,frame_train), (2,frame_train)]

    val_i = np.arange(4)
    frame_val = np.vstack((val_i,val_i,val_i)).T
    expected_val = [(1,frame_val),(2,frame_val)]

    R = np.arange(10)
    frame_points = np.vstack((R,R,R)).T
    mean_points = [
        (1,frame_points),
        (2,frame_points)
    ]
    result_train, result_val=_split_into_train_val(
        mean_points,x_split=4
    )

    for i,(t, frame) in enumerate(result_train):
        np.testing.assert_equal(t,expected_train[i][0])
        np.testing.assert_array_almost_equal(frame,expected_train[i][1])
    
    for i,(t, frame) in enumerate(result_val):
        np.testing.assert_equal(t,expected_val[i][0])
        np.testing.assert_array_almost_equal(frame,expected_val[i][1])


def test_get_L2_acc():
    x_true = np.array([1,0])
    x_pred = np.array([1,1])

    expected = 0
    result = _get_L2_acc(x_true, x_pred)
    np.testing.assert_almost_equal(expected,result)

def test_construct_f():
    coef = (1,2,3)
    f = _construct_f(coef)

    expected = np.array([3,6,11])

    result = f(np.array([0,1,2]))

    np.testing.assert_array_almost_equal(expected,result)

    # Test Parametric
    coef = np.array([1,2,3,3,2,1])
    f = _construct_f(coef,regression_method='poly_para')
    expected = np.array([
        [3,1],
        [6,6],
        [11,17]]
    )
    result = f(np.array([0,1,2])).T

    np.testing.assert_array_almost_equal(expected,result)
    

def test_get_true_pred():
    # Test "linear"
    R = np.array([0,1,2])
    f = lambda x: 2*x + 3
    r_x_y = np.vstack((R,R,f(R))).T

    result_true, result_pred = _get_true_pred(f,r_x_y,regression_method='linear')

    expected = np.array([3,5,7])

    np.testing.assert_array_almost_equal(expected,result_true)
    np.testing.assert_array_almost_equal(expected,result_pred)

    # Test "poly" and "linear"
    R = np.array([0,1,2])
    f = lambda x: x**2 + 2*x + 3
    r_x_y = np.vstack((R,R,f(R))).T

    result_true, result_pred = _get_true_pred(f,r_x_y,regression_method='poly')

    expected = np.array([3,6,11])

    np.testing.assert_array_almost_equal(expected,result_true)
    np.testing.assert_array_almost_equal(expected,result_pred)

    # Test "poly_inv"
    R = np.array([0,1,2])
    f = lambda x: x**2 + 2*x + 3
    r_x_y = np.vstack((R,f(R),R)).T

    result_true, result_pred = _get_true_pred(f,r_x_y,regression_method='poly_inv')

    expected = np.array([3,6,11])

    np.testing.assert_array_almost_equal(expected,result_true)
    np.testing.assert_array_almost_equal(expected,result_pred)
    
    # Test "poly_para"
    R = np.array([0,1,2])
    f = _construct_f(np.array([1,2,3,3,2,1]),regression_method='poly_para')
    r_x_y = np.vstack((R,f(R))).T

    result_true, result_pred = _get_true_pred(f,r_x_y,regression_method='poly_para')

    expected = np.array([
        [3,1],
        [6,6],
        [11,17]]
    )

    np.testing.assert_array_almost_equal(expected,result_true)
    np.testing.assert_array_almost_equal(expected,result_pred)





    

    