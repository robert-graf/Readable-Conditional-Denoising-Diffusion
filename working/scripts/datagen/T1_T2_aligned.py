from BIDS.registration.ridged_intensity.register import crop_shared_, register_native_res, registrate_nipy
from BIDS import NII, BIDS_Global_info, BIDS_FILE, to_nii
from pathlib import Path
import numpy as np
import secrets
from joblib import Parallel, delayed

out_path = Path("/media/data/robert/datasets/nako_upscale_t1_t2/")


def intersect_z(a: NII, b: NII, min_size=40):
    x1 = a.affine.dot([0, 0, 0, 1])[2]
    x2 = a.affine.dot(a.shape + (1,))[2]
    y1 = b.affine.dot([0, 0, 0, 1])[2]
    y2 = b.affine.dot(b.shape + (1,))[2]
    if x1 + min_size <= y1 and x2 >= y1 - min_size:
        return True
    if x1 + min_size <= y2 and x2 >= y2 - min_size:
        return True
    return False


def slice_and_save(a: NII, b: NII, water: NII, fat: NII, phase: str, idx):

    outfile = out_path / phase
    outfile.mkdir(exist_ok=True)
    t1 = a.get_array()
    t2 = b.get_array()
    w = water.get_array()
    f = fat.get_array()
    for i in range(a.shape[-1]):
        st = f"{idx}_{i}_{secrets.token_urlsafe(3)}.npz"
        np.savez_compressed(outfile / st, t1=t1[..., i], t2=t2[..., i], water=w[..., i], fat=f[..., i])


def main(n_jobs=10):
    # Find all nako files:

    bids_global = BIDS_Global_info(["/media/data/new_NAKO/NAKO/MRT/"], parents=["rawdata"])
    # bids_global = BIDS_Global_info(["/media/data/new_NAKO/NAKO/MRT/"], parents=["rawdata"])
    counter_to_many = 0
    subj_list: list[tuple[str, dict[str, BIDS_FILE], dict[str, BIDS_FILE]]] = []
    for subj, sub_container in bids_global.enumerate_subjects():

        # print(subj)
        query = sub_container.new_query(flatten=True)
        query.filter("format", lambda x: x in ["T2w", "t1dixon"])
        query_t2 = query.copy()
        query_t2.filter("chunk", lambda x: x in ["LWS", "HWS", "BWS"], required=True)
        query_t1 = query.copy()
        query_t1.filter("rec", "in", required=True)
        # T1
        region_t1: dict[str, BIDS_FILE] = {}
        for bf in query_t1.loop_list():
            c = bf.get("chunk")
            assert c not in region_t1 and c is not None
            region_t1[c] = bf

        # T2
        if len(list(query_t2.loop_list())) != 3:
            counter_to_many += 1
            continue
        region_t2: dict[str, BIDS_FILE] = {}
        for key in query_t2.loop_list():
            c = key.get("chunk")
            assert c is not None
            region_t2[c] = key[0]
        assert len(region_t2.keys()) == 3
        subj_list.append((subj, region_t1, region_t2))
    print(counter_to_many)
    print(len(subj_list))
    max_img = 1000
    if len(subj_list) > max_img:
        import random

        random.seed(1337)
        subj_list = random.sample(subj_list, max_img)

    if n_jobs > 1:
        print("[*] Running {} parallel jobs. Note that stdout will not be sequential".format(n_jobs))
    # generate(4, out)
    Parallel(n_jobs=n_jobs)(
        delayed(__help)(idx, sub, region_t1, region_t2, max_img)
        for idx, (sub, region_t1, region_t2) in enumerate(subj_list)
    )


def __help(idx, sub, region_t1: dict[str, BIDS_FILE], region_t2: dict[str, BIDS_FILE], max_img):
    phase = "train"
    if idx >= max_img * 0.6:
        phase = "val"
    if idx >= max_img * 0.8:
        phase = "test"
    for key, val in region_t1.items():
        moving = NII.load_bids(val)
        water = val.get_changed_path("nii.gz", info={"rec": "water"}, parent="rawdata", make_parent=False)
        fat = val.get_changed_path("nii.gz", info={"rec": "fat"}, parent="rawdata", make_parent=False)
        for s in ["HWS", "BWS", "LWS"]:
            fixed = NII.load_bids(region_t2[s])
            if intersect_z(moving, fixed):
                # start = moving.affine.dot([0, 0, 0, 1])[2]

                # Registration
                try:
                    m_out, f_out, transformation, other_moving = register_native_res(
                        moving, fixed, other_moving=[NII.load(water, False, 0), NII.load(fat, False, 0)]
                    )
                    # goal = m_out.affine.dot([0, 0, 0, 1])[2]
                    length = np.linalg.norm(transformation.translation)
                    if length >= 3.5:
                        continue
                    # Crop raster
                    crop = crop_shared_(m_out, f_out)
                    [i.apply_crop_slice_(crop) for i in other_moving]
                    print(f_out.shape)
                except ValueError as ex:
                    continue
                assert f_out.orientation == ("L", "P", "S")

                # skip images with small vertical intersections
                if f_out.shape[2] <= 30:
                    continue
                # print(m_out.shape, f_out.shape, f_out.orientation)
                slice_and_save(
                    m_out, f_out, other_moving[0], other_moving[1], phase, str(idx) + "_" + s + f"_{length:.3f}"
                )


def copy_bad_quality():
    # Find all nako files:
    bids_global = BIDS_Global_info(["/media/data/new_NAKO/NAKO/MRT/"], parents=["rawdata"])
    for subj, sub_container in bids_global.enumerate_subjects():

        # print(subj)
        query = sub_container.new_query(flatten=True)
        query.filter("format", lambda x: x in ["T2w", "t1dixon"])
        query_t2 = query.copy()
        query_t2.filter("chunk", lambda x: x in ["LWS", "HWS", "BWS"], required=True)
        # T2
        if len(list(query_t2.loop_list())) <= 3:
            continue

        for f in query_t2.loop_list():
            out = f.get_changed_path("nii.gz", parent="rawdata_low_quality_ds")
            # print(out)
            f.open_nii().save(out, make_parents=True)


if __name__ == "__main__":
    main(n_jobs=16)
    # copy_bad_quality()
