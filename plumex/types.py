from typing import Literal

import numpy as np
from numpy.typing import NBitBase

PolyData = np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating[NBitBase]]]
TimeData = np.ndarray[tuple[int], np.dtype[np.floating[NBitBase]]]
