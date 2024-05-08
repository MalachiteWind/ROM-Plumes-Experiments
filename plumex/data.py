import pickle
from pathlib import Path

import numpy as np

from .types import PolyData



pickle_path = Path(__file__).parent.resolve() / "../plume_videos/"

def load_pickle(filename: str) -> PolyData:
    with open(pickle_path / filename, 'rb') as f:
        data_file = pickle.load(f)
        if isinstance(data_file,np.ndarray) is True:
            return {"data": data_file}
        elif isinstance(data_file,dict) is True:   
            return {"data": data_file["mean"]}
        else:
            raise(ValueError("Datafile must be an array or dict with key 'mean'."))
