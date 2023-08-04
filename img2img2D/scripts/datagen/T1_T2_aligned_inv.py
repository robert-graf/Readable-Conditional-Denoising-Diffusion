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


def main(max_files=None):
    # Find all nako files:
    bids_global = BIDS_Global_info(["/media/data/new_NAKO/NAKO/MRT/"], parents=["rawdata"])
    counter_to_many = 0
    subj_list: list[tuple[str, dict[str, BIDS_FILE]]] = []
    for subj, sub_container in bids_global.enumerate_subjects():
        # print(subj)
        query = sub_container.new_query(flatten=True)
        query.filter("format", lambda x: x in ["T2w", "t1dixon"])
        query_t2 = query.copy()
        query_t2.filter("chunk", lambda x: x in ["LWS", "HWS", "BWS"], required=True)
        region_t2: dict[str, BIDS_FILE] = {}
        for key in query_t2.loop_list():
            c = key.get("chunk")
            assert c is not None
            region_t2[c] = key[0]
        assert len(region_t2.keys()) == 3
        subj_list.append((subj, region_t2))
    max_img = 1000
    if len(subj_list) > max_img:
        import random

        random.seed(1337)
        subj_list = random.sample(subj_list, max_img)
    if max_files is not None:
        subj_list = subj_list[:max_files]

    for subj, region_t2 in subj_list:

        for s in ["HWS", "BWS", "LWS"]:
            file = region_t2[s]
            fixed = NII.load_bids(file)
            fixed.reorient_(("L", "P", "S"))
            z = (
                fixed.zoom[0] * 1.5,
                fixed.zoom[1] * 1.5,
                fixed.zoom[2],
            )
            fixed_r = fixed.rescale(z)

            transfrom(fixed_r)


@torch.no_grad()
def transfrom(a: NII):
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
    p = Path("/media/data/robert/code/cyclegan/logs_diffusion/nako_upscale_t1_t2_water/version_1/checkpoints/").glob(
        "*.ckpt"
    )
    model_water = Diffusion.load_from_checkpoint(str(next(p)))
    p = Path("/media/data/robert/code/cyclegan/logs_diffusion/nako_upscale_t1_t2/version_1/checkpoints/").glob("*.ckpt")
    model = Diffusion.load_from_checkpoint(str(next(p)))
    main()
    # copy_bad_quality()
