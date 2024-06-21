"""
This script opens up the click_coordinates for a file given on the
command line, saving it in creates a <file>_ctr.pkl.

The syntax is:
python -m plumex.choose_center <date folder> <wind folder> <num>

So for plume_videos/Jan_10_2024/med/IMG_0943.MOV, you would call
python -m Jan_10_2024 med 0943
"""

import pickle

from ara_plumes import PLUME, click_coordinates
from pathlib import Path

from plumex.data import PICKLE_PATH as DATA_DIR


def _select_filename(date: str, wind: str, suffix: str, data_dir: Path) -> Path:
    """Calculate the path from the data directory to a movie file"""
    if not (
        (pth := data_dir / date).exists()
        and (pth := pth / wind).exists()
        and (pth := pth / f"IMG_{suffix}.MOV").exists()
    ):  # short-circuits so pth is the first file/folder that doesn't exist
        raise RuntimeError(f"{pth} does not exist")
    return pth


def create_metadata(date: str, wind: str, suffix: str):
    """Create and save the centerpoint metadata for a saved video file

    Args:
        date: the date of the folder in Mmmm_d_yyyy format
        wind: "low", "med", or "high"
        suffix: the XXXX digits in IMG_XXXX.MOV
    """
    filename = _select_filename(date, wind, suffix, DATA_DIR)
    plume = PLUME(str(filename))
    source = tuple(int(coord) for coord in click_coordinates(plume.video_capture))
    with open(str(filename) + "_ctr.pkl", "wb") as ctr_file:
        pickle.dump(source, ctr_file)



if __name__ == "__main__":
    import sys
    create_metadata(*sys.argv[1:])

# %%
