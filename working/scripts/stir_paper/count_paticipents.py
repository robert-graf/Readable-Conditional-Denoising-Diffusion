from pathlib import Path
from BIDS.snapshot2D import create_snapshot, Snapshot_Frame
import pandas as pd
import random
import secrets
from BIDS import BIDS_FILE, BIDS_Global_info
import numpy as np
from tqdm import tqdm

ds_path = "/DATA/NAS/datasets_processed/SHIP/code/train_paired.xlsx"
df = pd.read_excel(ds_path)
stats = {"M": [], "F": [], "O": []}
for name, row in tqdm(df.iterrows(), total=len(df)):
    root = row["Path"].replace("resampled", "rawdata")
    if not str(root).startswith("/"):
        root = Path(ds_path).parent / root

    path = Path(root, row["T2w"].replace(".nii.gz", ".json").replace("_res-T2w-1-vibe_desc-n4stitched", ""))
    if "res-0" in str(path):
        continue
    if "res-1" in str(path):
        continue
    if "DIXCTWS" in str(path):
        continue

    json = BIDS_FILE(path, root).open_json()
    age = int(json["PatientAge"].replace("Y", ""))
    gender = json["PatientSex"]
    stats[gender].append(age)

su = 0
for k, v in stats.items():
    print(k, len(v))
    su += len(v)
    v = np.array(v)
    print(np.mean(v), np.std(v))
print("total", su)

stats = {"M": [], "F": [], "O": []}
ds_path = "/DATA/NAS/datasets_processed/MRI_spine/dataset-wsgan"
ds_path = "/DATA/NAS/datasets_processed/MRI_spine/dataset-wsgan"
bgi = BIDS_Global_info([ds_path], ["rawdata"])
for name, subj in bgi.enumerate_subjects():
    q = subj.new_query()
    q.flatten()
    q.filter("part", "inphase", required=False)
    q.unflatten()
    q.filter_format("T1w")
    q.filter_format("T2w")

    l = list(q.loop_dict())
    json = l[0]["T2w"][0].open_json()
    age = int(json["PatientAge"].replace("Y", ""))
    gender = json["PatientSex"]
    stats[gender].append(age)


su = 0
for k, v in stats.items():
    l = len(v)
    if l == 0:
        continue
    print(k, l)
    su += l
    v = np.array(v)
    print(np.mean(v), np.std(v))
print("total", su)
exit()
