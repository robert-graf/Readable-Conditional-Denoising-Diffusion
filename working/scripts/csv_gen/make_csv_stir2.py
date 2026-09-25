from pathlib import Path
from BIDS import BIDS_Global_info, BIDS_FILE, NII, BIDS_Family
from joblib import Parallel, delayed
import numpy as np

dataset_path_in = ["/DATA/NAS/datasets_processed/spine/dataset-DIXCTWS/", "/DATA/NAS/datasets_processed/spine/dataset-wsgan/"]

dataset_path = "/DATA/NAS/datasets_processed/combinde/dataset-stir/"


import pandas as pd
import random
from BIDS.logger.log_file import Reflection_Logger, Log_Type

logger = Reflection_Logger()
random.seed(42)

df = pd.read_excel(Path(dataset_path, "train_paired_t1_t2.xlsx"))

import shutil

t1w_c = 0
w_c = 0
f_c = 0

for i, row in df.iterrows():
    path: str = row.Path
    t2w_path = Path(path, row.T2w)
    t2w_nii = NII.load(t2w_path, False, 0)
    t1w_path = Path(path, row.T1w)
    t1w_nii = NII.load(t1w_path, False, 0)
    w_path = Path(path, row.water)
    w_nii = NII.load(w_path, False, 0)
    try:
        f_path = Path(path, row.fat)
        f_nii = NII.load(f_path, False, 0)
    except:
        f_path = None
        f_nii = None
    if t2w_nii.shape != t1w_nii.shape:
        t1w_c += 1
        t1w_nii.resample_from_to_(t2w_nii)
        t1w_nii.save(t1w_path)
    if t2w_nii.shape != w_nii.shape:
        w_c += 1
        w_nii.resample_from_to_(t2w_nii)
        w_nii.save(w_path)
    if f_nii is not None and t2w_nii.shape != f_nii.shape:
        f_c += 1
        f_nii.resample_from_to_(t2w_nii)
        f_nii.save(f_path)
print(t1w_c, f_c, w_c)
exit()
