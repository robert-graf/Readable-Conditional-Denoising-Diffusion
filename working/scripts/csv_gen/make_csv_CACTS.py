from pathlib import Path
from BIDS import BIDS_Global_info, BIDS_FILE
from joblib import Parallel, delayed

dataset_path = "/DATA/NAS/datasets_processed/CT_fullbody/dataset-CACTS"
import pandas as pd
import random
from BIDS.logger.log_file import Reflection_Logger, Log_Type

logger = Reflection_Logger()
random.seed(42)
# same space
# Phase Path CT     T2w       T2w
# train path target condition condition ...
d = {"Phase": [], "Path": [], "CT": [], "msk1": [], "msk2": []}


def rest_path(x: Path | None):
    if x is None:
        return None
    return str(x).replace(dataset_path + "/", "")


for dataset in Path(dataset_path, "generated_segmentations").iterdir():
    i = 0
    for p1 in dataset.iterdir():
        for p in p1.glob("*/*combined.nii.gz"):
            mask1 = p
            mask2 = Path(p.parent, p.name.replace("combined.nii", "part_259.nii"))
            ct = Path(str(p).replace("/generated_segmentations/", "/ct_ready_1mm/").replace("_combined.nii", "_0000.nii"))
            ct = Path(ct.parent.parent, ct.name)
            if ct.exists() and mask1.exists() and mask2.exists():
                split = random.random()
                if split < 0.8:
                    phase = "train"
                elif split < 0.85:
                    phase = "val"
                else:
                    phase = "test"
                i += 1
                d["Phase"].append(phase)
                d["Path"].append(dataset_path)
                d["CT"].append(rest_path(ct))
                d["msk1"].append(rest_path(mask1))
                d["msk2"].append(rest_path(mask2))
                print(ct.name, end="\r")
            else:
                print(ct.exists(), mask1.exists(), mask2.exists())
                print(ct)
                # /DATA/NAS/datasets_processed/spine/dataset-CACTS/ct_ready_1mm/0015_lndb/images/LNDb-0238/LNDb-0238_0000.nii.gz
                # /DATA/NAS/datasets_processed/spine/dataset-CACTS/ct_ready_1mm/0015_lndb/images/LNDb-0238_0000.nii.gz

    print(dataset.name, i)

    ################################################################################################################################
df = pd.DataFrame(data=d)
df.to_excel(Path(dataset_path, "training.xlsx"))
