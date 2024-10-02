import numpy as np
from plumex.regress_edge import create_lin_func, create_sin_func
from typing import Callable

def apply_theta_shift(
        t,
        r,
        x,
        y,
        flat_regress_func:Callable[[float,float],float],
        positive:bool=True
)->tuple[float,float]:
    d = flat_regress_func(t,r)
    theta_1 = np.arctan2(d,r)
    theta_2 = np.arctan2(y,x)

    def pos_sign(positive:bool):
        if positive:
            return 1
        return -1
    
    theta = theta_2+pos_sign(positive)*theta_1

    return r*np.cos(theta), r*np.sin(theta)

def create_edge_func(coeffs:tuple, method:str)->Callable[[tuple,tuple], tuple]:
    if method == "linear":
        return create_lin_func(coeffs)
    if method == "sinusoid":
        return create_sin_func(coeffs)
    else:
        raise ValueError(f"{method} is not an accepted method.")



