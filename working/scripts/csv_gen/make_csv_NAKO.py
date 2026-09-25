import math
from pathlib import Path
from BIDS import BIDS_Global_info, BIDS_FILE, BIDS_Family, NII
from joblib import Parallel, delayed

dataset_path = "/DATA/NAS/datasets_processed/NAKO/MRT/"
dataset = BIDS_Global_info([dataset_path], parents=["rawdata"])

import pandas as pd
import random
from BIDS.logger.log_file import Reflection_Logger, Log_Type

logger = Reflection_Logger()
random.seed(42)
from BIDS.registration import register_native_res
import numpy as np
from skimage.metrics import structural_similarity as ssim


def rest_path(x: BIDS_FILE | Path | None):
    if x is None:
        return None
    if isinstance(x, BIDS_FILE):
        x = x.file["nii.gz"]
    par = x.parent.name
    return par + "/" + x.name


def check_quality(outs: list[Path]):
    try:
        nii = NII.load(outs[0], False)
        arr = nii.get_array()

        arr = arr / float(arr.max())
        m = 1
        ssim_min = []
        redo = False
        for path2 in outs[1:]:
            nii2 = NII.load(path2, False)
            arr2 = nii2.get_array()
            m = min(m, arr2.max())
            arr2 = arr2 / float(arr2.max())
            try:
                ssim_v = ssim(arr, arr2, data_range=1)
            except Exception:
                ssim_v = -1
                redo = True
            if ssim_v <= 0.1:
                redo = True
            ssim_min.append(f"{ssim_v:0.2f}")
            if math.isnan(ssim_v):
                redo = True
            print(f"{ssim_v:0.2f} ", end=" ")
        print()
        # if m == 0:
        #    exit("m == 0")
        return str(ssim_min), redo
    except Exception:
        logger.print_error()
        return "unk", True


def resample_t2w_t1dixon(t1d: BIDS_Family, t2w: BIDS_Family, split, skip):  #
    t2 = t2w["T2w"][0]
    inp = t1d["t1dixon_rec-in"][0]
    fat = t1d["t1dixon_rec-fat"][0]
    water = t1d["t1dixon_rec-water"][0]
    opp = t1d["t1dixon_rec-opp"][0]
    t2_nii = t2.open_nii()
    inp_nii: NII = inp.open_nii()
    if t2_nii.is_intersecting_vertical(inp_nii, 40):
        region = t2.info["chunk"] if "chunk" in t2.info else "stitched"
        info = {"res": f"T2w-{region}-T1dixon-{inp.info['chunk']}"}
        b_file = t2, inp, fat, water, opp

        outs = [i.get_changed_path(parent="resampled", info=info) for i in b_file]

        if all([o.exists() for o in outs]):
            print("skip exists", info["res"])  # , [o.exists() for o in outs], outs)
            q, redo = check_quality(outs)
            if not redo:
                return (*outs, split, q, redo)
        if "stitched" == region or skip < 0.9:
            # print("skip", skip, region)
            return None
        inp_nii, _, T, _ = register_native_res(inp_nii, t2_nii)
        inp_nii.set_dtype_(np.uint16)
        t2_nii.set_dtype_(np.uint16)

        crop = t2_nii.reorient_(("P", "S", "R")).compute_crop_slice()
        crop = inp_nii.resample_from_to_(t2_nii).compute_crop_slice(other_crop=crop)
        try:
            t2_nii.apply_crop_slice_(crop).save(outs[0])
            inp_nii.apply_crop_slice_(crop).save(outs[1])
            t2_nii.save(outs[0])
            inp_nii.save(outs[1])
        except IndexError:
            return None

        for x, out in zip([fat, water, opp], outs[2:]):
            x = x.open_nii().resample_from_to_(t2_nii, verbose=False)
            x.save(out)
        return *outs, split, *check_quality(outs)
    return None


tasks = []
for name, subj in dataset.enumerate_subjects():
    q = subj.new_query()
    q.flatten()
    q.filter_format(lambda x: x != "mevibe")
    q.unflatten()

    split = random.random()
    if split < 0.8:
        phase = "train"
    elif split < 0.85:
        phase = "val"
    else:
        phase = "test"

    skip = random.random()
    t1dixon = []
    t2 = []
    for fam in q.loop_dict(key_addendum=["rec"]):
        if "t1dixon_rec-in" in fam:
            t1dixon.append(fam)
        if "T2w" in fam:
            t2.append(fam)
    for a in t1dixon:
        for b in t2:
            # resample_t2w_t1dixon(a, b)
            # if skip > 0.9:
            tasks.append(delayed(resample_t2w_t1dixon)(a, b, phase, skip))
out: list[None | tuple[Path, Path, Path, Path, Path, str, float, bool]] | None = Parallel(n_jobs=8, backend="threading")(t for t in tasks)
# print(out)
assert out is not None
d = {"Phase": [], "Path": [], "T2w": [], "fat": [], "water": [], "opp": [], "in": [], "ssim": []}
for o in out:
    if o is None:
        continue
    t2, inp, fat, water, opp, split, ssim_v, skip = o
    if skip:
        continue
    d["Path"].append(t2.parent.parent)
    d["Phase"].append(split)
    d["T2w"].append(rest_path(t2))
    d["fat"].append(rest_path(fat))
    d["water"].append(rest_path(water))
    d["opp"].append(rest_path(opp))
    d["in"].append(rest_path(inp))
    d["ssim"].append(ssim_v)
df2 = pd.DataFrame(data=d)
print(df2.head())
df2.to_excel(Path(dataset_path, "train_paired_resampled_t2w_t1dixon.xlsx"))
## delayed(sub_log_call_func)(
# Parallel(n_jobs=32, backend="threading")(t for t in tasks)
