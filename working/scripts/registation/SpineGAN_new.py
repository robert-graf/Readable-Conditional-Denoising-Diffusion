import numpy as np
from pathlib import Path


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

# Save registered file
def generate_file_path_reg(out, file: BIDS_FILE, other_file_id, folder, keys=["sub", "ses", "sequ", "reg", "acq"]):
    info = file.info.copy()
    info["reg"] = other_file_id
    file.info.clear()

    def pop(key):
        if key in info:
            file.info[key] = info.pop(key)

    for k in keys:
        pop(k)
    for k, v in info.items():
        file.info[k] = v
    if file.get("e") is not None and file.has_json():

        tl = file.open_json()["ImageType"]
        if "IP" in tl:
            info["acq"] = "real"
        elif "W" in tl:
            info["acq"] = "water"
        elif "F" in tl:
            info["acq"] = "fat"
        del info["e"]

    out_file: Path = file.get_changed_path(
        dataset_path=f"/media/data/robert/datasets/{out}/",
        file_type="nii.gz",
        parent="registration",
        path="{sub}/" + folder,
        info=info,
        from_info=True,
    )
    return out_file


def register(n_jobs=10):
    global_info = BIDS_Global_info(
        ["/media/data/robert/datasets/spine_transition_new/"],
        ["sourcedata", "rawdata", "rawdata_ct", "rawdata_dixon", "derivatives", "translated/derivatives"],
        additional_key=["sequ", "seg", "ovl", "e"],
    )

    Parallel(n_jobs=n_jobs)(
        delayed(__register)(subj_name, subject) for subj_name, subject in global_info.enumerate_subjects()
    )


def __register(subj_name, subject: Subject_Container):
    # subject: Subject_Container
    query = subject.new_query()
    # It must exist a dixon and a msk
    query.filter("format", "dixon")
    # A nii.gz must exist
    query.filter("Filetype", "nii.gz")
    query.filter("format", "msk")
    # print(query)
    # query.filter("label", "procspin")
    d = None
    for bids_file_d in query.loop_dict():
        # Do something with the files.
        # print(bids_file)
        dixon = bids_file_d["dixon"]

        query = subject.new_query(flatten=False)
        # A nii.gz must exist
        query.filter("Filetype", "nii.gz")
        # It must be a ct
        query.filter("format", "ct")
        query.filter("seg", "vert")
        for bids_file in query.loop_dict():
            assert len(bids_file["ct"]) == 1, bids_file["ct"]
            assert len(bids_file["msk_vert"]) == 1, bids_file["msk_vert"]
            # assert len(bids_file["msk"]) == 1, bids_file
            ct = bids_file["ct"][0]
            vert_ct = bids_file["msk_vert"][0]
            subreg_ct = bids_file["msk_subreg"][0]
            vert_di = bids_file_d["msk_vert"][0]
            subreg_di = bids_file_d["msk_subreg"][0]
            ctd_di = calc_centroids_labeled_buffered(
                msk_reference=vert_di,
                subreg_reference=subreg_di,
                subreg_id=[Location.Vertebra_Corpus.value, Location.Spinosus_Process.value],
                override=True,
            )
            ctd_ct = calc_centroids_labeled_buffered(
                msk_reference=vert_ct,
                subreg_reference=subreg_ct,
                subreg_id=[Location.Vertebra_Corpus.value, Location.Spinosus_Process.value],
                override=True,
            )
            assert ctd_di.shape != None
            reg_segmentation.ridged_registration_by_dict_from_ctd(
                [*dixon, vert_di],
                [ct, vert_ct],
                ctd_di,
                ctd_ct,
                generate_file_names=partial(generate_file_path_reg, "spine_transition_new"),
                axcodes_to=("R", "I", "P"),
                voxel_spacing=(-1, 1, 1),
                snap_shot_folder="/media/data/robert/datasets/spine_transition_new/",
            )


if __name__ == "__main__":
    register(n_jobs=16)
    # main()
    pass
