from pathlib import Path
import pickle
from BIDS import BIDS_Global_info, BIDS_FILE, NII, to_nii
from BIDS.snapshot2D import Snapshot_Frame, create_snapshot
from joblib import Parallel, delayed
from dataclasses import dataclass

dataset_path = "/DATA/NAS/datasets_processed/SHIP"
import pandas as pd
import random
from BIDS.logger.log_file import No_Logger, Log_Type
from nipy.algorithms.registration.affine import Affine
import numpy as np
from BIDS.registration import registrate_nipy, apply_registration_nipy

from tqdm import tqdm

logger = No_Logger()
random.seed(42)


@dataclass
class DF2(pd.Series):
    Path: str
    T1w: str
    T2w: str
    fat: str
    water: str
    opp: str
    inp: str


def test(root, subpath):
    root = row.Path
    f = Path(root, subpath)
    if not f.exists():
        f = str(f).replace("rawdata", "derivatives")
    if not Path(f).exists():
        f = str(f).replace("derivatives", "rawdata")
    if not Path(f).exists():
        f = str(f).replace("rawdata", "resampled")
    if not Path(f).exists():
        f = str(f).replace("rawdata", "translated/rawdata")
    if not Path(f).exists():
        f = str(f).replace("rawdata", "translated/derivatives")
    return to_nii(f)


tasks = []
tasks_stitched = []
incomplete = ["10106491", "10002140", "10002011", "10104319", "10000952", "10000688"]
not_same_shape = []
unmatched_vibe = ["10105510", "10105798"]
not_bad_reg = [
    "10001369",
    "10001555",
    "10002686",
    "10004016",
    "10004294",
    "10104690",
    "10104883",
    "10105082",
    "10105268",
    "10105697",
    "10106541",
    "10106747",
    "10106947",
    "10107376",
    "10107478",
    "10107517",
    "10107698",
    "10108332",
]


brocken = []

below_16 = 0
below_16_after = 0
crop_l = 0
crop_r = 0
count = 0

for name, row in pd.read_excel(Path(dataset_path, "code/train_paired.xlsx")).iterrows():  # type: ignore
    row: DF2
    path = row.Path
    cmp: list[NII] = []
    try:
        sub = BIDS_FILE(Path(path, row.T1w), path).get("sub")
        if sub in not_bad_reg:
            continue
        nii_a = test(path, row.T1w)
        sl = nii_a.compute_crop_slice()[-1]

        count += 1
        if sl.start != 0:
            crop_l += 1
        if sl.stop != nii_a.shape[-1]:
            crop_r += 1
        if sl.stop - sl.start < 16:
            below_16_after += 1
        if nii_a.shape[-1] < 16:
            below_16 += 1

        print(
            f"{crop_l:4} {crop_r:4} | {below_16_after:4} {below_16:4} | {count:4} - {crop_l/count:.2f} {crop_r/count:.2f} | {below_16_after/count:.2f} {below_16/count:.2f} | {sl}, {nii_a.shape[-1]:3}",
            end="\r",
        )

    except Exception as e:
        print(e)
