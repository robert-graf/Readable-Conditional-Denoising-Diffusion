from dataclasses import dataclass
from pathlib import Path
from BIDS.bids_files import BIDS_Global_info, Searchquery, Subject_Container
from BIDS.nii_utils import (
    to_nii,
    nii2arr,
    nii2arr,
)

# sudo chmod -R 777 *

import nibabel as nib
import numpy as np

# fmt: off
v_idx2name = {
     1: "C1",     2: "C2",     3: "C3",     4: "C4",     5: "C5",     6: "C6",     7: "C7", 
     8: "T1",     9: "T2",    10: "T3",    11: "T4",    12: "T5",    13: "T6",    14: "T7",    15: "T8",    16: "T9",    17: "T10",   18: "T11",   19: "T12",
    20: "L1",    21: "L2",    22: "L3",    23: "L4",    24: "L5",    25: "L6",    
    26: "Sacrum",27: "Cocc",  28: "T13",
}  
# fmt: on
v_name2idx = {v: k for k, v in v_idx2name.items()}


@dataclass
class datacontainer:
    subject_name = "mask_case213"
    sequ = "1"
    shift = 0
    mapping = {9: 0}
    # mapping = {"L1": "L2", "L2": "L3", "L3": "L4"}
    # mapping = {"T12": "T11", "T11": "T10", "T10": "T9"}
    # mapping = {"L5": 0, "T11": "T10"}
    # mapping = {"L5": "L6"}


# 0015 CUT away top
# [*] Calc centroids  fxclass0072
# [*] Calc centroids  fxclass0044
# [*] Calc centroids  fxclass0002
# [*] Calc centroids  fxclass0011
# [*] Calc centroids  fxclass0122


######
# 0044 #Shape miss-match(512, 1039, 103)(512, 1039, 61)
# 0122 #zero-size array to reduction operation minimum which has no identity
# 0011 # Broken file
# 0027 # Broken file

######################################
def change(subj_name, subject: Subject_Container | Path, data: datacontainer):
    if isinstance(subject, Subject_Container):
        query1: Searchquery = subject.new_query()
        # It must exist a dixon and a msk
        query1.filter("sequ", data.sequ)
        query1.flatten()
        query1.filter("format", "msk")
        query1.filter("seg", "vert", required=True)

        assert len(query1.candidates) == 1
        assert isinstance(query1.candidates, list), "query not flattened"
        bids_file = query1.candidates[0]
        file = bids_file.file["nii.gz"]
    else:
        bids_file = subject
        file = subject
    nii = to_nii(bids_file)
    arr = nii2arr(nii)
    print(subj_name)
    print(data.sequ)
    arr[arr == -1024] = 0
    print([v_idx2name[i] for i in np.unique(arr)[1:]])
    print(np.unique(arr))

    arr += 1000
    for old, neu in data.mapping.items():
        if isinstance(old, str):
            old = v_name2idx[old]
        if isinstance(neu, str):
            neu = v_name2idx[neu]
        arr[old + 1000 == arr] = neu
    arr[arr >= 1000] -= 1000
    arr[arr != 0] += data.shift

    print(np.unique(arr))
    print([v_idx2name[i] for i in np.unique(arr)[1:]])
    print("Store in", file)
    a = input("do you want to execute this (yes/no)\n")

    if "y" in a:
        print("Store in", file)
        nib.save(nib.Nifti1Image(arr, nii.affine, nii.header), file)
    else:
        print("no")


if __name__ == "__main__":
    ####################

    info = datacontainer()
    ##################
    a = ""
    # ridged_point_registration()
    from pathlib import Path

    # src = "/media/data/robert/datasets/fx_T1w/"
    # global_info = BIDS_Global_info(
    #    [src],
    #    [
    #        "test_nii",
    #        "rawdata",
    #        "rawdat___",
    #        "derivatives",
    #        "derivatives_old",
    #    ],  # "sourcedata"
    #    additional_key=["sequ", "seg", "e", "ovl"],
    #    verbose=False,
    # )
    # for subj_name, subject in global_info.enumerate_subjects():
    #    if info.subject_name == subj_name:
    #        change(subj_name, subject, info)
    src = "/media/data/robert/datasets/MRSpineSeg_Challenge/train_ours/Mask/"
    for subject in Path(src).iterdir():
        if info.subject_name in subject.stem:
            change(subject.stem, subject, info)
