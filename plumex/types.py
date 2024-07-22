from typing import Literal

import numpy as np
from numpy.typing import NBitBase

NpFlt = np.dtype[np.floating[NBitBase]]

PolyData = np.ndarray[tuple[int, Literal[3]], NpFlt]
Float1D = np.ndarray[tuple[int], NpFlt]
Float2D = np.ndarray[tuple[int, int], NpFlt]
Int1D = np.ndarray[tuple[int], np.dtype[np.int_]]
