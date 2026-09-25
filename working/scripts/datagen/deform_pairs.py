from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parent))
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
import numpy as np

from dataset_generation import deformed_np

from numpy.typing import NDArray
from BIDS import (
    BIDS_Global_info,
    Subject_Container,
    BIDS_FILE,
    to_nii,
    load_centroids,
    calc_centroids_labeled_buffered,
    Location,
)
from BIDS.registration.ridged_points import reg_segmentation
from joblib import Parallel, delayed

import pandas as pd
from functools import partial


from BIDS.snapshot2D import Snapshot_Frame, create_snapshot
from BIDS import NII

data_set_name = "spinegan_T2w_all_reg_iso"
for number in range(20):
    for i in Path(f"/media/data/robert/datasets/{data_set_name}/registration").iterdir():
        for session in i.glob("*"):
            if len(list(session.iterdir())) != 0:
                paths = {
                    "ct": next(session.glob("*_ct.nii.gz")),
                    "seg": next(session.glob("*_seg-vert_msk.nii.gz")),
                    "seg2": next(session.glob("*_e*_msk.nii.gz")),
                    "dixon_f": next(session.glob("*_dixon.nii.gz")),
                }
                niis = {k: NII.load(v, seg="msk" in str(v)) for k, v in paths.items()}
                arrs = {k: v.get_array() for k, v in niis.items()}
                lvl = {k: 0 if v.seg else 3 for k, v in niis.items()}

                out = deformed_np(arrs, lvl, crackup=0.6)
                for k, v in out.items():
                    path = paths[k]
                    out_name = str(path).replace("/registration/", f"/registration_deformed/{number}/")
                    Path(out_name).parent.mkdir(exist_ok=True, parents=True)
                    niis[k].set_array(v).save(out_name)
