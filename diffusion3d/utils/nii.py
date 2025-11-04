import traceback
import warnings
from typing import Literal

import numpy as np
from nibabel import Nifti1Image  # type: ignore

# R: Right, L: Left; S: Superio (up), I: Inverior (down); A: Anterior (front), P: Posterior (back)
Directions = Literal["R", "L", "S", "I", "A", "P"]
Ax_Codes = tuple[Directions, Directions, Directions]
LABEL_MAX = 256
Zooms = tuple[float, float, float]

Centroid_Dict = dict[int, tuple[float, float, float]]
Coordinate = tuple[float, float, float]
POI_Dict = dict[int, dict[int, Coordinate]]

Rotation = np.ndarray
Label_Map = dict[int | str, int | str] | dict[str, str] | dict[int, int]

_formatwarning = warnings.formatwarning


def formatwarning_tb(*args, **kwargs):
    s = "####################################\n"
    s += _formatwarning(*args, **kwargs)
    tb = traceback.format_stack()[:-3]
    s += "".join(tb[:-1])
    s += "####################################\n"
    return s


warnings.formatwarning = formatwarning_tb


from TPTBox.core.nii_wrapper import *  # noqa: E402
