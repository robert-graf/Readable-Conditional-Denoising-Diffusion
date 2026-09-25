from pathlib import Path
import pickle
from BIDS import BIDS_Global_info, BIDS_FILE, NII, to_nii
from BIDS.snapshot2D import Snapshot_Frame, create_snapshot
from joblib import Parallel, delayed

dataset_path = "/DATA/NAS/datasets_processed/SHIP"
# dataset_path = "/DATA/NAS/ongoing_projects/robert/SHIP"
dataset = BIDS_Global_info([dataset_path], parents=["rawdata"])
import pandas as pd
import random
from BIDS.logger.log_file import Reflection_Logger, Log_Type
from nipy.algorithms.registration.affine import Affine
import numpy as np
from BIDS.registration import registrate_nipy, apply_registration_nipy

logger = Reflection_Logger()
random.seed(42)
# same space
# Phase Path CT     T2w       T2w
# train path target condition condition ...
d = {"Phase": [], "Path": [], "T1w": [], "T2w": [], "T2star": []}
d2 = {"Phase": [], "Path": [], "T1w": [], "T2w": [], "fat": [], "water": [], "opp": [], "inp": [], "org": []}
from BIDS.stitching import stitching

buffer_path = Path(dataset_path, "code/vibe2T2w_registration")
if Path(buffer_path).exists():
    reg_buffer = pickle.load(open(buffer_path, "rb"))
else:
    reg_buffer = {}


def equal(a: BIDS_FILE, b: BIDS_FILE):
    a = a.open_nii()
    b = b.open_nii()
    if a.shape != b.shape:
        return False
    if a.orientation != b.orientation:
        return False

    return True


df2 = pd.read_excel(Path(dataset_path, "code/train_paired.xlsx"))
# df2 = pd.DataFrame(data=d2)

from dataclasses import dataclass


@dataclass
class DF2(pd.Series):
    Path: str
    T1w: str
    T2w: str
    fat: str
    water: str
    opp: str
    inp: str
    org: str


def test(root, subpath):
    root = row.Path
    f = Path(root, subpath)
    if not f.exists():
        f = str(f).replace("resampled", "rawdata")
    if not Path(f).exists():
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


out: list[pd.Series] = []
out_q: list[float] = []
print("QA")
from tqdm import tqdm

remove = ["10000315", "10000446", "10000682", ("10000833", "0-81")]
for name, row in df2.iterrows():  # tqdm(df2.iterrows(), total=len(df2)):  # type: ignore
    row: DF2
    path = row.Path
    bf = BIDS_FILE(row.T1w, path)
    sub = bf.get("sub")
    if sub in remove:
        print(row)
        if sub in reg_buffer:
            del reg_buffer[sub]
        continue
    cmp: list[NII] = []
    try:
        # test(path, row.T1w)
        # test(path, row.T2w)
        # test(path, row.fat)
        # test(path, row.water)
        # test(path, row.opp)
        # test(path, row.inp)
        out.append(row)
    except Exception as e:
        pass
        # print(e)
# pickle.dump(reg_buffer, open(buffer_path, "wb"))
# df2 = pd.DataFrame(out)
# df2.to_excel(Path(dataset_path, "code/train_paired.xlsx"))
