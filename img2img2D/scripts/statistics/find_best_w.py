import sys
from pathlib import Path
import traceback

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
import json
import math
import torch
from diffusion import Diffusion

import numpy as np
import math
from dataloader.dataloader_nii_seg import nii_Dataset, nii_Datasets
import nibabel as nib


ws = [0, 0.1, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 1000]
fs = [
    lambda w, _: w,
    lambda w, t: w * t / 1000,
    lambda w, t: w * math.log2(t + 1),
    lambda w, t: w * t * t / 1000,
    lambda w, t: w * math.sqrt(t),
    lambda w, t: w * 1 if t > 500 else 0,
    lambda w, t: w * 1 if t > 200 else 0,
    lambda w, t: w * 1 if t > 50 else 0,
    lambda w, t: w * t / 1000 * math.sin(t / 10),
]
fs_s = [
    "w",
    "w*t-div-1000",
    "w*log2(t)",
    "w*t*t-div-1000",
    "w*sqrt(t)",
    "greater 500",
    "greater 200",
    "greater 50",
    "w*t-div-1000*sin(t-div-10)",
]

etas = [0, 0.5, 1, 2]
ts = [10, 20, 50, 100]


out_dict = {}
for w in ws:
    out_dict[str(w)] = {}
    for f_s in fs_s:
        out_dict[str(w)][str(f_s)] = {}
        for eta in etas:
            out_dict[str(w)][str(f_s)][str(eta)] = {}
            if eta == 2:
                out_dict[str(w)][str(f_s)][str(eta)]["1000"] = None
            else:
                for t in ts:
                    out_dict[str(w)][str(f_s)][str(eta)][str(t)] = None
        if w == 0:
            break

print(len(ws) * len(fs) * len(etas))

#### Load network ####
name = "wopathfx_256"
version: str = "4"

from loader.arguments import get_latest_Checkpoint

checkpoint = get_latest_Checkpoint(name, log_dir_name="logs_diffusion", best=False, version="*")
assert checkpoint is not None
model = Diffusion.load_from_checkpoint(checkpoint, strict=False)
model.cuda()
#### Load dataset ####
dataset = nii_Datasets(
    root="../../datasets/MRSpineSeg_Challenge/train/MR/",
)

from reload_any import dice, run_docker

#### the training function ###


def run_dl(w, w_map, eta, ts):
    with torch.no_grad():
        for i, nii_ds in enumerate(dataset):  # type: ignore
            nii_ds: nii_Dataset
            a = max(0, (nii_ds._nii.shape[-1] - 256) // 2)
            crop = (0, a, 256, 256)

            arr = nii_ds.get_copy(crop=crop)  # [from_idx:to_idx]
            arr /= arr.max()
            x_conditional = (arr * 2 - 1).unsqueeze(1).cuda()

            if eta == 2:
                out = model.forward(
                    x_conditional.shape[0],
                    x_conditional=x_conditional,
                    guidance_w=w,
                    guidance_mapping=w_map,
                )
            else:
                out = model.forward_ddim(
                    x_conditional.shape[0],
                    range(0, 1000, ts),
                    x_conditional=x_conditional,
                    w=w,
                    guidance_mapping=w_map,
                    eta=eta,
                )[0]
            assert not isinstance(out, tuple)
            out_save = nii_ds.insert_sub_corp(out.squeeze(1) * 2000 - 1000, offset=crop[:2], fill_value=-1000).numpy()

            ### Save CT image
            nii_ds.save(
                out_save,
                f'/media/data/robert/datasets/MRSpineSeg_Challenge/result/rawdata/same_{Path(nii_ds.file).stem.replace(".nii","")}_ct',
                revert_to_size=True,
            )  #
            # nii_ds.save(
            #    out_save,
            #    f'/media/data/robert/datasets/MRSpineSeg_Challenge/result/rawdata/1x1_{Path(nii_ds.file).stem.replace(".nii","")}_ct',
            #    revert_to_size=False,
            # )
            return nii_ds


def run_eval(nii_ds, w, w_map, eta, t):
    out_file = f'/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/same_{Path(nii_ds.file).stem.replace(".nii","")}_seg-subreg_msk.nii.gz'
    out_file_b = f'/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/same_{Path(nii_ds.file).stem.replace(".nii","")}_seg-vert_msk.nii.gz'
    # out_file2 = f'/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/1x1_{Path(nii_ds.file).stem.replace(".nii","")}_seg-vert_msk.nii.gz'
    nii1: nib.Nifti1Image = nib.load(out_file)
    nib.save(
        nii1,
        f"/media/data/robert/datasets/MRSpineSeg_Challenge/result/seg_test/eta-{eta:.1f}_t-{t:04d}_w-{w}_wmap-{w_map}_subreg.nii.gz",
    )
    nii1: nib.Nifti1Image = nib.load(out_file_b)
    nib.save(
        nii1,
        f"/media/data/robert/datasets/MRSpineSeg_Challenge/result/seg_test/eta-{eta:.1f}_t-{t:04d}_w-{w}_wmap-{w_map}_vert.nii.gz",
    )
    im1: np.ndarray = nii1.get_fdata()  # type: ignore
    im1[im1 != 0] = 1

    nii: nib.Nifti1Image = nib.load(nii_ds.file_seg)
    im2: np.ndarray = nii.get_fdata()  # type: ignore
    im2[im2 >= 9] = 0  # Bandscheiben und edge vertebra
    im2[im2 == 1] = 0  # Sacrum
    im2[im2 != 0] = 1
    # Run Evaluation
    out_dict[str(w)][str(w_map)][str(eta)][str(t)] = dice(im1, im2)
    print("[*] Dice score: ", out_dict[str(w)][str(w_map)][str(eta)][str(t)])
    try:
        Path(
            f'/media/data/robert/datasets/MRSpineSeg_Challenge/result/rawdata/same_{Path(nii_ds.file).stem.replace(".nii","")}_ct.nii.gz'
        ).unlink()
    except Exception:
        pass
    try:
        Path(
            f'/media/data/robert/datasets/MRSpineSeg_Challenge/result/rawdata/1x1_{Path(nii_ds.file).stem.replace(".nii","")}_ct.nii.gz',
        ).unlink()

    except Exception:
        pass
    try:
        Path(out_file_b).unlink()
    except Exception:
        pass
    with open("/media/data/robert/datasets/MRSpineSeg_Challenge/result/seg_test.json", "w") as f:
        json.dump(out_dict, f, indent=5)


if Path("/media/data/robert/datasets/MRSpineSeg_Challenge/result/seg_test.json").exists():
    with open("/media/data/robert/datasets/MRSpineSeg_Challenge/result/seg_test.json", "r") as f:
        out_dict = json.load(f)
else:
    print("[*] no json found, starting from scratch")


def main_loop():

    for w in ws:
        for f_s, w_map in zip(fs_s, fs):
            for eta in etas:
                for t in ts:
                    if eta == 2:
                        t = 1000
                    if out_dict[str(w)][str(f_s)][str(eta)][str(t)] != None:
                        print(f"[*] SKIP w-{w} eta-{eta:.1f} t-{t:04d} wmap-{f_s}")
                        continue
                    try:
                        print(f"[*] Run DL-Prediction w-{w} eta-{eta:.1f} t-{t:04d} wmap-{f_s}")
                        # break
                        ds = run_dl(w, w_map, eta, t)

                        print("[*] Run Docker")
                        run_docker()

                        print("[*] Run Eval")
                        run_eval(ds, w, f_s, eta, t)
                    except Exception as e:
                        print(traceback.format_exc())
                        out_dict[str(w)][str(f_s)][str(eta)][str(t)] = 0

                    if eta == 2:
                        break
            if w == 0:
                break


main_loop()
