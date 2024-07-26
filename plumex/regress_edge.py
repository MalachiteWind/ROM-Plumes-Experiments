from ara_plumes.models import flatten_edge_points
from typing import cast
from typing import List
from .types import PlumePoints

import numpy as np


# run after video_digest
def regress_edge(data:dict):
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

        t_rad_dist_bot = np.hstack((t*np.ones(len(rad_dist_bot).reshape(-1,1),rad_dist_bot)))
        t_rad_dist_top = np.hstack((t*np.ones(len(rad_dist_top).reshape(-1,1),rad_dist_top)))

        bot_flattened.append(t_rad_dist_bot)
        top_flattened.append(t_rad_dist_top)
        
    
    