import numpy as np

name = "sine-exp"
lookup_dict = {
    "amplitude": {"low":0.1, "high":10}
}
def run(seed, amplitude):
    """
    Deterimne if the maximum value of the sine function equals ``amplitude``
    """
    x = np.arange(0, 10, .05)
    y = amplitude * np.sin(x)
    err = np.abs(max(y) - amplitude)
    results = {"main": err}
    return results
