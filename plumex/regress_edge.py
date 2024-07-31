from ara_plumes.models import flatten_edge_points
from typing import cast
from typing import List
from .types import PlumePoints

import numpy as np


# run after video_digest
def regress_edge(data:dict,
                 training: float,
                 randomize: bool = True,
                 seed: int = 1234
):
    np.random.seed(seed=seed)
    center = cast(List[tuple[int,PlumePoints]],data["center"])
    bot = cast(List[tuple[int,PlumePoints]],data["bottom"])
    top = cast(List[tuple[int,PlumePoints]],data["top"])

    assert len(center) == len(top)
    assert len(top) == len(bot)

    bot_flattened = []
    top_flattened = []
    for (t,center_pp), (t,bot_pp), (t,top_pp) in zip(center, bot, top):
        
        rad_dist_bot = flatten_edge_points(center_pp,bot_pp)
        rad_dist_top = flatten_edge_points(center_pp,top_pp)

        t_rad_dist_bot = np.hstack((t*np.ones(len(rad_dist_bot),1),rad_dist_bot))
        t_rad_dist_top = np.hstack((t*np.ones(len(rad_dist_top),1),rad_dist_top))

        bot_flattened.append(t_rad_dist_bot)
        top_flattened.append(t_rad_dist_top)
    
    top_flattened = np.concatenate(top_flattened,axis=0)
    bot_flattened = np.concatenate(bot_flattened,axis=0)
    
    # create training data
    indices_top = np.arange(len(top_flattened))
    indices_bot = np.arange(len(bot_flattened))

    if randomize:
        np.random.shuffle(indices_top)
        np.random.shuffle(indices_bot)
    
    top_train_idx = int(len(top_flattened)*training)
    bot_train_idx = int(len(bot_flattened)*training)

    top_train = top_flattened[:top_train_idx,:]
    bot_train = top_flattened[:bot_train_idx,:]

    top_test = top_flattened[top_train_idx:,:]
    bot_test = bot_flattened[bot_train_idx:,:]






        
    
    