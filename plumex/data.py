import pickle
from pathlib import Path

import numpy as np

from .types import PolyData



PICKLE_PATH = Path(__file__).parent.resolve() / "../plume_videos/"

def load_centerpoints(filename: str) -> dict[str, PolyData]:
    with open(PICKLE_PATH / filename, 'rb') as f:
        data_file = pickle.load(f)
        if isinstance(data_file,np.ndarray) is True:
            return {"data": data_file}
        elif isinstance(data_file,dict) is True:   
            return {"data": data_file["center"]}
        else:
            raise(ValueError("Datafile must be an array or dict with key 'center'."))
