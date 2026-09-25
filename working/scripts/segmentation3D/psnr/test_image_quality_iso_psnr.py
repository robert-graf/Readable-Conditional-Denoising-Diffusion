from __future__ import annotations
import json

import sys
from pathlib import Path
from typing import Callable

file = Path(__file__).resolve()
sys.path.append(str(file.parents[3]))
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
from scripts.segmentation3D.run3D import revert_iso_to_original


import pickle

import numpy as np
import torch

###### Model
from scipy import ndimage
from torch import Tensor


from scripts import reload_any

# CUDA_VISIBLE_DEVICES=1
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_cut --new --factor 1 -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_cut_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_pix2pix --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_pix2pix_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_pcut_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --syndiff -en exp_syndiff --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --ddim --eta 0 -t 20 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --ddim --eta 1 -t 20 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --ddim --eta 0 -t 50 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --ddim --eta 1 -t 50 -en paper_T1_diffusion --new -bs 24 --factor 10

# python3 scripts/statistics/test_image_quality.py --ddpm               -en paper_T1_diffusion --new -bs 24 --factor 10

# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_cut --new --factor 1 -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_cut_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_pix2pix --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_pix2pix_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_pcut_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --syndiff -en exp_syndiff_t2w --new -bs 4 --factor 10

# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 0 -t 20 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 1 -t 20 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 0 -t 50 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 1 -t 50 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddpm               -en paper_T2_diffusion --new -bs 24 --factor 10


def test_image_quality(opt: reload_any.Reload_Any_Option):
    print(opt)
    exp_name = opt.exp_name
    version: str = opt.version
    ddim = opt.ddim
    cut = opt.cut
    guidance_w = opt.guidance_w
    test = opt.test
    root = opt.root
    do_reload = False  # not opt.new

    _, checkpoint = reload_any.get_model(opt, no_reload=True)

    pickle_file = Path(checkpoint).parent.parent
    pickle_file = Path(pickle_file, f"quality/qa_{opt.get_result_folder_name()}.pkl")

    #### Dataset
    import random
    from piq import DISTS, vif_p

    ####### Loop
    from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    metric: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
        "L1": lambda x, y: torch.mean(torch.abs(x - y)),
        "MSE": torch.nn.functional.mse_loss,
        "PSNR": peak_signal_noise_ratio,
        # "SSIM": structural_similarity_index_measure,  # type: ignore
        # https://piq.readthedocs.io/en/latest/overview.html
        # "VIFp": vif_p,
        # "DISTS": DISTS(),
    }
    batchable = ["SSIM", "VIFp", "DISTS"]
    # batchable = []
    metric_list: dict[str, list[float]] = {}
    if do_reload:
        if pickle_file.exists():
            with open(pickle_file, "rb") as f:
                metric_list = pickle.load(f)

    pickle_file.parent.mkdir(exist_ok=True)
    shortest_key_count = 100000000
    for x in metric:
        if x not in metric_list:
            metric_list[x] = []
            shortest_key_count = 0
        else:
            if shortest_key_count >= len(metric_list[x]):
                shortest_key_count = len(metric_list[x])

    bs = opt.bs
    image_list = []

    root = "/media/data/robert/datasets/spinegan_T2w/test_nii"
    files_gt = sorted(list(Path(root).glob("*/*/*_ct.nii.gz")))
    out = "/media/data/robert/test/T2w_2"
    files_seg = sorted(list(Path(root).glob("*/*/*seg.nii.gz")))
    add = ""
    if not "CUT" in opt.exp_name:
        add = f"*{opt.timesteps}_w"
    files_pred = sorted(list(Path(out).glob(f"*{opt.exp_name}{add}*/translated/*/*/*/*_ct.nii.gz")))
    print(files_gt[0], len(files_gt))
    print(files_pred[0], len(files_pred))
    print(files_seg[0], len(files_seg))
    from BIDS import NII

    for j, (file_pred, file_gt, files_seg) in enumerate(zip(files_pred, files_gt, files_seg), 1):
        seg_nii = NII.load(files_seg, True).reorient_(("R", "I", "P"))
        seg_nii.dilate_msk(mm=2, inplace=True, verbose=False)

        target_nii = NII.load(file_gt, False)
        json_file = str(file_pred).replace("nii.gz", "json")
        pred_nii = NII.load(file_pred, False)

        target_nii = target_nii.reorient_(("R", "I", "P"))
        out_json = {}
        if Path(json_file).exists():
            with open(json_file) as f:
                out_json = json.load(f)

        pad = out_json["padding"]

        out = pred_nii.get_array()
        out = out[pad[4] : pad[5], pad[2] : pad[3], pad[0] : pad[1]]
        pred_nii = pred_nii.set_array(out)

        pred_nii = pred_nii.reorient_(("R", "I", "P"))
        pred_nii.pad_to(target_nii.shape, inplace=True)
        target = torch.from_numpy(target_nii.get_array()).unsqueeze_(0)
        pred = torch.from_numpy(pred_nii.get_array()).unsqueeze_(0)
        target = np.clip((target / 2000) + 0.5, 0, 2)
        pred = np.clip((pred / 2000) + 0.5, 0, 2)

        seg = torch.from_numpy(seg_nii.get_array().astype(float)).unsqueeze_(0)

        roll_ct = -1

        arr = pred
        arr[:, roll_ct:] = arr[:, :-roll_ct].clone()
        pred = arr

        target_s = target * seg
        pred_s = pred * seg
        for key, metric_fun in metric.items():
            if key in batchable:
                # m = metric_fun(target_s, out_s)
                # metric_list[key].append(m.detach().cpu().item())
                for i in range(pred_s.shape[0]):
                    m = metric_fun(pred_s[i].unsqueeze(1), target_s[i].unsqueeze(1))
                    metric_list[key].append(m.detach().cpu().item())
            else:
                for i in range(pred_s.shape[0]):
                    m = metric_fun(pred_s[i], target_s[i])
                    metric_list[key].append(m.detach().cpu().item())
            if j % 100 == 99:
                with open(pickle_file, "wb") as filehandler:
                    pickle.dump(metric_list, filehandler)
            if test and j == 100:
                break
        print(f"\rCompute metrics on id {j}/{len(files_pred)}    \t\t{np.array(metric_list['PSNR']).mean()}", end="\r")
        # metric_list["PSNR"] = []

    # with open(pickle_file, "wb") as filehandler:
    #    pickle.dump(metric_list, filehandler)
    print(metric_list)
    for key, value_list in metric_list.items():
        l = np.array(value_list)
        print(f"{key}\t {l.mean():.4f} ± {l.var():.4f}                \t\t{l.shape}")
    print("\n")
    for i, (key, value_list) in enumerate(metric_list.items()):
        l = np.array(value_list)
        if i == 0 or i == 1:
            print(f"{l.mean():.4f}", end="\t")
        else:
            print(f"{l.mean():.3f}", end="\t")
    print("\n")


if __name__ == "__main__":
    # exp_names = ["diffusion3D_T2_CT","CUT_diffusion3D_T2_CT" "dif_spinegan3D_T2_CT", "dif_spinegan3D_T2_CT_4_steps", "dif_spinegan3D_T2_CT_4_steps_noise"]
    # log_dir_name = ["logs_diffusion", "logs_diffusion3D", "logs_diffusion3D", "logs_diffusion3D", "logs_diffusion3D"]

    # exp_names = ["CUT_diffusion3D_T2_CT"]
    # exp_names = ["diffusion3D_T2_CT"]
    exp_names = ["CUT_diffusion3D_T2_CT", "diffusion3D_T2_CT", "dif_spinegan3D_T2_CT_img"]
    log_dir_name = ["logs_diffusion3D", "logs_diffusion", "logs_diffusion3D"]

    parser = reload_any.get_option_reload()
    opt = reload_any.get_option(parser)
    opt.timesteps = 25
    opt.eta = 1
    opt.guidance_w = 0
    opt.ddim = True
    for name, dir in zip(exp_names, log_dir_name):
        opt.exp_name = name
        opt.log_dir_name = dir
        if "CUT" in name:
            opt.cut = True
            opt.ddim = False
            opt.ddpm = False
        test_image_quality(opt)
