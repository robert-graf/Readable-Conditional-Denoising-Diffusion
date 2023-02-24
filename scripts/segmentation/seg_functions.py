from __future__ import annotations
from math import ceil
from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))

import torch
from dataloader.dataloader_nii_seg import nii_Dataset, nii_Datasets
import reload_any
from reload_any import Reload_Any_Option as ROpt, TranslationType
import nibabel as nib
import numpy as np


####################################### Translation MRI to CT #####################################################


def run_model(
    opt: ROpt, dataset, no_translation=False, trans_type: TranslationType = TranslationType.V_STITCHING
) -> tuple[list[Path], list[Path], Path]:
    model, checkpoint = reload_any.get_model(opt)

    rawdata_folder = opt.get_folder_rawdata()
    rawdata_folder.mkdir(exist_ok=True, parents=True)

    ####### Loop
    file_ct_names = []
    file_mr_names = []
    length = len(dataset)
    with torch.no_grad():
        for i, nii_ds in enumerate(dataset):  # type: ignore
            # assert all([nii_ds.nii.affine[0, 0] != 1, nii_ds.nii.affine[1, 1] != 1, nii_ds.nii.affine[2, 2] != 1])
            print(f"[*]{i+1:4}/{length:4} - {opt.exp_name}\t                                               ", end="\r")
            out_save = None
            # Remember Paths
            f = rawdata_folder / (Path(nii_ds.file).stem.replace(".nii", "").rsplit("_", maxsplit=1)[0] + "_ct.nii.gz")
            file_ct_names.append(f)
            if opt.keep_resampled:
                nii_ds.load()
                mr_file = Path(f.parent, Path(nii_ds.file).name)
                print(mr_file)
                nib.save(nii_ds.nii, mr_file)
                file_mr_names.append(mr_file)

            else:
                file_mr_names.append(Path(nii_ds.file))
            if no_translation:
                continue
            if not opt.override and f.exists():
                print("[*] Skip! File already exist:", f)
                continue
            nii_ds: nii_Dataset
            nii_ds.load()
            out_save = translate(nii_ds, model, opt, trans_type)
            if out_save is None:
                continue
            assert out_save.min() >= -1000, f"[{out_save.min()} - {out_save.max()}] out >= -1000"
            assert out_save.max() <= 1000, f"[{out_save.min()} - {out_save.max()}] out <= 1000"
            ### Save CT image

            nii_ds.save(out_save, str(f), revert_to_size=not opt.keep_resampled, min_value=-1000)  #
            del out_save

            if opt.test:
                break
    return file_ct_names, file_mr_names, Path(rawdata_folder)


def translate(nii_ds: nii_Dataset, model: reload_any.Models, opt: ROpt, trans_type: TranslationType):
    if trans_type == TranslationType.V_STITCHING:
        return translate_v_stitching(nii_ds, model, opt)
    elif trans_type == TranslationType.TOP:
        return translate_top(nii_ds, model, opt)

    raise NotImplementedError(f"NotImplementedError in translate- {trans_type}")


@torch.no_grad()
def translate_v_stitching(nii_ds: nii_Dataset, model: reload_any.Models, opt: ROpt) -> None | torch.Tensor:
    out_save = None
    shape = nii_ds.nii.shape  # type: ignore
    size = model.opt.size
    steps = shape[-2] // size + 1
    step_size = shape[-2] // steps
    prev_bottom = 0

    for j in range(steps):

        a = max(0, (shape[-1] - size) // 2)
        b = max(0, min(step_size * j, shape[-2] - size))
        crop = (b, a, size, size)

        arr = nii_ds.get_copy(crop=crop)
        # print("\n", crop, step_size)
        # Normalize
        arr /= arr.max()
        arr[arr <= 0] = 0
        x_conditional = (arr * 2 - 1).unsqueeze(1).cuda()
        out = reload_any.get_image(x_conditional, model, opt)
        assert isinstance(out, torch.Tensor)
        assert out.min() >= 0, f"[{out.min()} - {out.max()}] out >= 0"
        assert out.max() <= 1, f"[{out.min()} - {out.max()}] out <= 0"
        out_padded = nii_ds.insert_sub_corp(out.squeeze(1) * 2000, offset=crop[:2], fill_value=0).numpy()
        if out_save is None:
            out_save = out_padded
        else:
            curr_up = b
            center = ceil((curr_up + prev_bottom) / 2)
            # if prev_bottom != 0:
            #    # print(curr_up, center, prev_bottom)
            #    out_save[:, center:prev_bottom] = 0
            #    out_padded[:, curr_up:center] = 0
            out_save[(out_save != 0) & (out_padded != 0)] *= 0.5
            out_padded[(out_save != 0) & (out_padded != 0)] *= 0.5
            out_save = out_save + out_padded
            # print(f"[{out_save.min()} - {out_save.max()}]", out_save.shape)
            assert out_save.min() >= 0, f"[{out_save.min()} - {out_save.max()}] out_save >= 0"
            assert out_save.max() <= 2000, f"[{out_save.min()} - {out_save.max()}] out_save <= 0"

        prev_bottom = b + size

    # [0,2000]
    if out_save is not None:
        out_save -= 1000
    # [-1000,1000]
    return out_save


@torch.no_grad()
def translate_top(nii_ds: nii_Dataset, model: reload_any.Models, opt: ROpt) -> torch.Tensor:
    a = max(0, (nii_ds.nii.shape[-1] - 256) // 2)
    crop = (0, a, 256, 256)

    arr = nii_ds.get_copy(crop=crop)  # [from_idx:to_idx]
    arr /= arr.max()
    arr[arr <= 0] = 0
    x_conditional = (arr * 2 - 1).unsqueeze(1).cuda()
    # print("in", x_conditional.shape)
    out = reload_any.get_image(x_conditional, model, opt)

    out_save = nii_ds.insert_sub_corp(out.squeeze(1) * 2000 - 1000, offset=crop[:2], fill_value=-1000).numpy()
    # [-1000,1000]
    return out_save


####################################################### DICE ###################################################################################


def compute_dice_on_nii(file_seg_fake, file_seg_org, rawdata_folder, n_jobs=1, mapping={}, mapping_pred={}):
    import pandas as pd
    from reload_any import compute_dice
    from utils.nii_utils import v_idx2name

    #### Compute Things
    # if n_jobs > 1:
    #    print("[*] Running {} parallel jobs. Note that stdout will not be sequential".format(n_jobs))
    result: list[list[str | int]] = [
        ["" if i != 26 else str(j.name).split("_", maxsplit=1)[0] for j in file_seg_fake] for i in range(27)
    ]

    for id, (file_ct, file_seg_gt) in enumerate(zip(file_seg_fake, file_seg_org)):
        print("[*] dice id =", id, end="\r")
        compute_dice(file_ct, file_seg_gt, id, result, mapping=mapping, mapping_pred=mapping_pred)

    df = pd.DataFrame(result).T
    df.to_excel(
        excel_writer=f"{str(rawdata_folder.parent)}/dice_scores.xlsx",
        header=["global"] + [v_idx2name[i] for i in range(1, 26)] + ["name"],
    )

    all = []
    for i, l in enumerate(result):
        l = [i for i in l if not isinstance(i, str)]

        result[i] = l  # type: ignore
        if len(l) <= 1:
            continue
        if i != 0:
            all += l
        print(i, np.mean(l), np.std(l))

    print("per vertebra", np.mean(all), np.std(all))
