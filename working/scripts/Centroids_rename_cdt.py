from dataclasses import dataclass
from pathlib import Path
from BIDS.bids_files import BIDS_Global_info, Searchquery, Subject_Container
from BIDS import to_nii, v_idx2name, v_name2idx, Centroids

# sudo chmod -R 777 *

import nibabel as nib
import numpy as np


@dataclass
class data_container:
    subject_name = "spinegan0010"
    sequ = "301"
    shift = 0  # Late!
    mapping = {"C7": 0, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16}
    introduce: int | str | None = "C3"
    purge: int | str | None = None
    del_arr = True


def change(subj_name, subject: Subject_Container, data: data_container):
    query1: Searchquery = subject.new_query()
    # It must exist a dixon and a msk
    query1.filter("sequ", data.sequ)
    query1.filter("format", "msk")
    query1.filter("seg", "vert", required=True)
    l = list(query1.loop_dict())
    if len(l) == 0:
        return
    print(l)
    assert len(l) == 1
    fam = l[0]
    assert len(fam["ctd_subreg"]) == 1
    assert len(fam["msk_subreg"]) == 1
    assert len(fam["msk_vert"]) == 1
    jsn = fam["ctd_subreg"][0].file["json"]
    cdt = Centroids.load(jsn)
    cdt_out = cdt.copy().map_labels_(data.mapping)

    if data.introduce is not None:
        introduce = v_name2idx[data.introduce] if data.introduce in v_name2idx else int(data.introduce)
        if introduce in cdt_out:
            o = {}
            for k, v in cdt_out.items():
                if introduce <= k:
                    o[k + 1] = v
                else:
                    o[k] = v
            cdt_out.centroids = o
        a = cdt_out[introduce - 1]
        b = cdt_out[introduce + 1]
        cdt_out[introduce] = tuple((a + b) / 2 for a, b in zip(a, b))
        cdt_out.sorting_list = list(range(255))
        cdt_out.sort()
    if data.purge is not None:
        print(data.purge)
        purge = v_name2idx[data.purge] if data.purge in v_name2idx else int(data.purge)
        if purge in cdt_out:
            o = {}
            for k, v in cdt_out.items():
                if purge < k:
                    o[k - 1] = v
                else:
                    o[k] = v
            cdt_out.centroids = o
            # print(o)
        cdt_out.sorting_list = list(range(255))
        cdt_out.sort()
        cdt_out.sorting_list = None
    if data.shift != 0:
        o = {}
        for k, v in cdt_out.items():
            if k + data.shift > 0:
                o[k + data.shift] = v
        cdt_out.centroids = o

    print("in:", [v_idx2name[c] for c in cdt.keys()])
    print("out:", [v_idx2name[c] for c in cdt_out.keys()])
    print("Store in", jsn)
    print(cdt_out)
    a = input("do you want to execute this (yes/no)\n")
    if "y" in a:
        print("Store in", jsn)
        cdt_out.save(jsn)
        if data.del_arr:
            fam["msk_subreg"][0].file["nii.gz"].unlink()
            fam["msk_vert"][0].file["nii.gz"].unlink()
    else:
        print("no")


if __name__ == "__main__":
    ####################

    info = data_container()
    ##################
    a = ""
    # ridged_point_registration()
    from pathlib import Path

    # src = "/media/data/robert/datasets/spinegan_T2w/translation/result_exp_syndiff_t2w/"
    # global_info = BIDS_Global_info(
    #    [src],
    #    [
    #        "rawdata",
    #        "derivatives",
    #    ],  # "sourcedata"
    #    additional_key=["sequ", "seg", "e", "ovl"],
    #    verbose=False,
    # )
    # for subj_name, subject in global_info.enumerate_subjects():
    #    if info.subject_name in subj_name:
    #        change(subj_name, subject, info)

    src = "/media/data/robert/datasets/spinegan_T2w/translation/result_exp_syndiff_t2w/"
    global_info = BIDS_Global_info(
        [src],
        [
            "rawdata",
            "derivatives",
        ],  # "sourcedata"
        additional_key=["sequ", "seg", "e", "ovl"],
        verbose=False,
    )
    from BIDS import calc_centroids

    for subj_name, subject in global_info.enumerate_subjects():
        q = subject.new_query()
        q.filter("format", "ct")
        for fam in q.loop_dict():
            ct = fam["ct"][0]
            # print(ct)
            f = f"/media/data/robert/datasets/spinegan_T2w/sourcedata/rawdata_dixon/{subj_name}/sub-{subj_name}_ses-{ct.info['ses']}_sequ-{ct.info['sequ']}_e-{ct.info['e']}_msk.nii.gz"
            f = Path(f)
            out = f"/media/data/robert/datasets/spinegan_T2w/translation/result_exp_syndiff_t2w/derivatives/sub-{subj_name}_ses-{ct.info['ses']}_sequ-{ct.info['sequ']}_e-{ct.info['e']}_seg-subreg_ctd.json"
            
            if f.exists() and not Path(out).exists():
                calc_centroids(f).save(out)
            