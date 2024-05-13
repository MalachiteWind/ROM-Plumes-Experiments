import pickle
from typing import Any, Optional

from ara_plumes import PLUME
from .data import pickle_path
from .types import PolyData


def create_centerline(
    filename: str,
    img_range: Optional[list[int]] = None,
    subtraction_method: str="fixed",
    fixed_range: Optional[int]=None,
    gauss_kw: dict[str, Any]=None,
    regression_method: str="sinusoid",
    circle_kw: dict[str, Any]=None,
    regression_kw: dict[str, Any]=None,
) -> dict[str, PolyData]:
    """Calculate the centerline path from a moviefile

    Args:
        filename: path from plume_videos/ to movie file.  Must have adjacent
            centerpoint file
        remainder of args are from ara_plumes.PLUME initialization
    """
    if gauss_kw is None:
        gauss_kw = {}
    if circle_kw is None:
        circle_kw = {}
    if regression_kw is None:
        regression_kw = {}

    origin_filename = filename + "_ctr.pkl"
    with open(pickle_path / origin_filename, "rb") as fh:
        origin = pickle.load(fh)
    plume = PLUME(str(pickle_path / filename))
    plume.orig_center = tuple(int(coord) for coord in origin)
    center, _, _ = plume.train(
        img_range=img_range,
        subtraction_method=subtraction_method,
        fixed_range=fixed_range,
        display_vid=False,
        regression_method=regression_method,
        concentric_circle_kws=circle_kw,
        regression_kws=regression_kw,
        **gauss_kw
    )
    return {"main": None, "data": center}

