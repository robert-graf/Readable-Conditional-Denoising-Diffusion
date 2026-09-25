from pathlib import Path
from BIDS import BIDS_Global_info, BIDS_FILE, calc_centroids, Location, Centroids
from BIDS.snapshot2D import create_snapshot, Snapshot_Frame
from BIDS.registration.ridged_intensity.register import register_native_res
from joblib import Parallel, delayed

dataset_path = "/media/data/NAKO/NAKO/MRT/"
dataset = BIDS_Global_info(
    [dataset_path], parents=["rawdata/101", "derivatives_rostock-generated"]
)
import pandas as pd
import random
from BIDS.logger.log_file import Reflection_Logger, Log_Type
from enum import Enum, auto

logger = Reflection_Logger()
random.seed(42)


def resample(t2w: BIDS_FILE, t2w_stitched: BIDS_FILE, seg: BIDS_FILE):
    path = t2w.get_changed_path(
        "json",
        parent="derivatives_translated",
        format="ctd",
        info={"seg": "subreg", "desc": "rostock-generated"},
    )
    path2 = t2w.get_changed_path(
        "json",
        parent="derivatives_translated",
        format="ctd",
        info={"seg": "subreg", "desc": "rostock-generated-no-reg"},
    )
    if path.exists() and path2.exists():
        return
    if path2.exists():
        ctd2 = Centroids.load(path2)
    else:
        _, ctd2 = __resample(path2, t2w, t2w_stitched, seg, reg=Registration_Type.NO)
        if path.exists():
            ctd = Centroids.load(path)
            logger.print(path, len(ctd2), len(ctd), type=Log_Type.STRANGE)
            if len(ctd2) > len(ctd) + 1:
                pass
                path.unlink(missing_ok=True)
            else:
                return

    seg_nii, ctd = __resample(path, t2w, t2w_stitched, seg)

    if len(ctd2) > len(ctd) + 1:
        logger.print("Failed to reg with cc try MI", type=Log_Type.STRANGE)
        logger.flush()
        seg_nii, ctd = __resample(
            path, t2w, t2w_stitched, seg, reg=Registration_Type.MI
        )
        if len(ctd2) > len(ctd) + 1:
            logger.print(path, len(ctd2), len(ctd), type=Log_Type.FAIL)
            path.unlink(missing_ok=True)
            return

    create_snapshot(
        Path(
            "/media/data/NAKO/NAKO/MRT/snapshot_rostock",
            str(path.name).replace("json", "png"),
        ),
        frames=[
            Snapshot_Frame(t2w, seg_nii, ctd, coronal=True),
            Snapshot_Frame(t2w, None, ctd2, coronal=True),
        ],
    )


class Registration_Type(Enum):
    NO = auto()
    CC = auto()
    MI = auto()


def __resample(
    path,
    t2w: BIDS_FILE,
    t2w_stitched: BIDS_FILE,
    seg: BIDS_FILE,
    reg=Registration_Type.CC,
):
    t2w_nii = t2w.open_nii()
    seg_nii = seg.open_nii_reorient(t2w_nii.orientation)
    # print(seg_nii.unique(), seg_nii.shape)
    if Registration_Type.CC.name == reg.name:
        if t2w.info["chunk"] != "HWS":
            st = t2w_stitched.open_nii_reorient(t2w_nii.orientation)

            _, _, _, a = register_native_res(
                st, t2w_nii, other_moving=[seg_nii], similarity=reg.name.lower()  # type: ignore
            )
            seg_nii = a[0]
    # print(seg_nii.unique(), seg_nii.shape)
    seg_nii.resample_from_to_(t2w_nii)
    seg_nii.remove_labels(60, 100, inplace=True)
    try:
        # remove last
        if t2w.info["chunk"] != "LWS":
            seg_nii.remove_labels(seg_nii.unique()[-1], inplace=True)
        # remove first
        if t2w.info["chunk"] != "HWS":
            seg_nii.remove_labels(seg_nii.unique()[0], inplace=True)
    except IndexError:
        return seg_nii, Centroids(seg_nii.orientation)
    ctd = calc_centroids(msk=seg_nii, location=Location.Vertebra_Corpus)
    ctd.remove_centroid(60)
    ctd.remove_centroid(Location.Vertebra_Disc)
    logger.print("keys", ctd.keys())
    try:
        ctd.save(path, verbose=True)
    except Exception:
        pass
    return seg_nii, ctd


tasks = []
for name, subj in dataset.enumerate_subjects():
    q = subj.new_query(flatten=True)
    q.filter_non_existence("chunk")
    q.filter_non_existence("rec")
    q.filter_format(filter_fun=lambda x: x not in ["pd", "T1dixon"])
    q.unflatten()
    seg = None
    t2w_stitched = None
    for fam in q.loop_dict(key_addendum=["desc"]):
        if "T2w" in fam:
            t2w_stitched = fam["T2w"][0]
        if "msk_seg-vert_desc-rostock-generated" in fam:
            seg = fam["msk_seg-vert_desc-rostock-generated"][0]
    if seg is None or t2w_stitched is None:
        continue
    q = subj.new_query(flatten=True)
    q.filter_format("T2w")
    q.filter("chunk", lambda x: True)
    for t2w in q.loop_list():
        try:
            resample(t2w, t2w_stitched, seg)
        except Exception:
            logger.print_error()
