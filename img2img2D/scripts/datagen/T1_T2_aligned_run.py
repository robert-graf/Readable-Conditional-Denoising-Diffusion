from math import ceil, floor
from BIDS.registration.ridged_intensity.register import crop_shared_, register_native_res, registrate_nipy
from BIDS import NII, BIDS_Global_info, BIDS_FILE, to_nii
from pathlib import Path
import numpy as np
import secrets
from joblib import Parallel, delayed
from T1_T2_aligned import intersect_z
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from diffusion import Diffusion
import torch
from torch import Tensor

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

out_path = Path("/media/data/robert/datasets/nako_upscale_t1_t2/")


def main(n_jobs=10, max_files=3, water=False):
    # Find all nako files:
    bids_global = BIDS_Global_info(["/media/data/new_NAKO/NAKO/MRT/"], parents=["rawdata/100"])
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
    if max_files is not None:
        subj_list = subj_list[:max_files]
    # generate(4, out)
    Parallel(n_jobs=n_jobs)(
        delayed(__help)(idx, sub, region_t1, region_t2, max_img, water)
        for idx, (sub, region_t1, region_t2) in enumerate(subj_list)
    )


def __help(idx, sub, region_t1: dict[str, BIDS_FILE], region_t2: dict[str, BIDS_FILE], max_img, water_b: bool):
    for key, val_t1 in region_t1.items():
        moving = NII.load_bids(val_t1)
        water = val_t1.get_changed_path("nii.gz", info={"rec": "water"}, parent="rawdata", make_parent=False)
        fat = val_t1.get_changed_path("nii.gz", info={"rec": "fat"}, parent="rawdata", make_parent=False)
        for s in ["HWS", "BWS", "LWS"]:
            fixed = NII.load_bids(region_t2[s])

            if intersect_z(moving, fixed, min_size=0):
                # start = moving.affine.dot([0, 0, 0, 1])[2]
                out_path = val_t1.get_changed_path(
                    "nii.gz",
                    parent="rawdata_super_resolution",
                    info={"sequ": s, "rec": "in" if not water_b else "water"},
                    make_parent=False,
                )
                if out_path.exists():
                    continue
                # Registration
                try:
                    fixed.reorient_(("L", "P", "S"))
                    moving2 = moving.rescale_and_reorient(("L", "P", "S"), (-1, -1, fixed.zoom[-1]))
                    f_out = fixed.copy()
                    # print(fixed.zoom, moving2.zoom, moving.zoom)
                    # exit()
                    f_out.resample_from_to_(moving2)
                    m_out, transformation, other_moving = registrate_nipy(
                        moving2, f_out, other_moving=[NII.load(water, False, 0), NII.load(fat, False, 0)]
                    )
                    # TODO do it twice for iterative optimization
                    # goal = m_out.affine.dot([0, 0, 0, 1])[2]
                    length = np.linalg.norm(transformation.translation)
                    if length >= 10:
                        print("[!] skip, bad reg", length)
                        continue
                    # Crop raster
                    crop = crop_shared_(m_out, f_out)
                    [i.apply_crop_slice_(crop) for i in other_moving]
                    print(f_out.shape)
                except ValueError as ex:
                    continue
                assert f_out.orientation == ("L", "P", "S")

                # skip images with small vertical intersections
                # if f_out.shape[2] <= 30:
                #    continue
                # print(m_out.shape, f_out.shape, f_out.orientation)
                if water:
                    m_out = other_moving[0]
                upscale(m_out, f_out, out_path)


@torch.no_grad()
def upscale(a: NII, b: NII, out_path):
    global model
    condition = Tensor(a.get_array().astype(np.float32)).unsqueeze_(1)
    w, h = condition.shape[-2], condition.shape[-1]
    hp = max((model.opt.size_w - w) / 2, 0)
    vp = max((model.opt.size - h) / 2, 0)
    padding = [int(floor(vp)), int(floor(hp)), int(ceil(vp)), int(ceil(hp))]

    condition = TF.pad(condition, padding)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(condition, output_size=(model.opt.size_w, model.opt.size))
    condition = TF.crop(condition, i, j, h, w)

    condition /= float(condition.max()) / 2
    condition -= 1

    out = model.forward_ddim(a.shape[0], list(range(0, 1000, 250)), eta=0, x_conditional=condition, w=1)
    print(out[0].shape)
    a.set_array(out[0][:, 0].numpy())
    a.save(out_path, make_parents=True)


if __name__ == "__main__":
    water = True
    if water:
        p = Path(
            "/media/data/robert/code/cyclegan/logs_diffusion/nako_upscale_t1_t2_water/version_1/checkpoints/"
        ).glob("*.ckpt")
    else:
        p = Path("/media/data/robert/code/cyclegan/logs_diffusion/nako_upscale_t1_t2/version_1/checkpoints/").glob(
            "*.ckpt"
        )
    model = Diffusion.load_from_checkpoint(str(next(p)))
    main(n_jobs=1, water=water)
    # copy_bad_quality()
