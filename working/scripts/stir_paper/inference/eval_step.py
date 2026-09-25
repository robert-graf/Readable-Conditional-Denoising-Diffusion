import json
import sys
from datetime import date
from pathlib import Path

from CUT import CUT
from cycleGAN import CycleGAN

root = str(Path().parent.parent)
sys.path.append(root)
# water = "/DATA/NAS/datasets_processed/spine/dataset-DIXCTWS/resampled/sub-DIXCTWS148/sub-DIXCTWS148_sequ-3_rec-water_chunk-HWS_res-0-5x0-5-mm_T2w.nii.gz"
t1 = "/DATA/NAS/datasets_processed/spine/dataset-DIXCTWS/resampled/sub-DIXCTWS148/sub-DIXCTWS148_sequ-3_chunk-HWS_res-0-5x0-5-mm_T1w.nii.gz"
t2 = "/DATA/NAS/datasets_processed/spine/dataset-DIXCTWS/resampled/sub-DIXCTWS148/sub-DIXCTWS148_sequ-3_rec-in_chunk-HWS_res-0-5x0-5-mm_T2w.nii.gz"


import random
from math import ceil, floor
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nio
import nibabel.processing as nip
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from TPTBox import BIDS_FILE, NII
from TPTBox.logger.log_file import Log_Type, Logger_Interface

random.seed(1223)
mri_names = ["MRI", "T1", "t1", "T2", "t2", "T1GD", "FLAIR", "water", "fat", "T1w", "T2w"]


def _load(f, key="MRT"):
    if isinstance(f, NII):
        nii = f.nii
    else:
        nii: nib.Nifti1Image = nib.load(str(f))  # type: ignore
    aff = nii.affine
    ornt_fr = nio.io_orientation(aff)
    try:
        arr: np.ndarray = nii.get_fdata()
    except EOFError as e:
        print("EOF-ERROR", f)
        Path(f).unlink()
        raise e
    ornt_to = nio.axcodes2ornt(("R", "I", "P"))
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)

    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    new_aff = np.matmul(aff, aff_trans)
    if str(key).upper() == "CT":
        arr /= 1000
        arr = arr.clip(-1, 1)
        max_v = 1000
    else:
        max_v = float(np.max(arr))
        arr = arr / max(max_v, 0.000001)
        arr = arr.clip(0, 1e6)
        # arr *= 2
        # arr -= 1
    return arr, new_aff, max_v


def padded_shape(shape):
    shape = list(shape)
    for i, j in enumerate(shape):
        # if j % 8 != 0:
        shape[i] = j + 8 - (j % 8)
    return shape


def random_crop(target_shape: tuple[int, int, int], *arrs: Tensor):
    sli = [slice(None), slice(None), slice(None)]
    for i in range(3):
        z = max(0, arrs[0].shape[-i] - target_shape[-i])
        if z != 0:
            r = random.randint(0, z)
            r2 = r + target_shape[-i]
            sli[-i] = slice(r, r2 if r2 != arrs[0].shape[-i] else None)

    return tuple(a[..., sli[0], sli[1], sli[2]] for a in arrs)


def pad_size(x: Tensor, target_shape, mode="constant"):
    padding = []
    for in_size, out_size in zip(reversed(x.shape[-3:]), reversed(target_shape), strict=False):  # type: ignore
        to_pad_size = max(0, out_size - in_size) / 2.0
        padding.extend([ceil(to_pad_size), floor(to_pad_size)])
    x_ = (
        F.pad(x.unsqueeze(0).unsqueeze(0), padding, mode=mode).squeeze(0).squeeze(0)
    )  # mode – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
    return x_


def anti_pad(arr: np.ndarray, og_shape: tuple[int, ...]):
    x = -(og_shape[-3] - arr.shape[-3]) / 2
    y = -(og_shape[-2] - arr.shape[-2]) / 2
    z = -(og_shape[-1] - arr.shape[-1]) / 2
    m = lambda x: slice(ceil(x), -floor(x) if floor(x) != 0 else None)
    # print(x, y, z)
    # print(m(x), m(y), m(z))
    return arr[..., m(x), m(y), m(z)]


def transform(
    items_in,
    target_shape,
    condition_types=None,
    padding="reflect",
    flip=False,
):
    # Transform to tensor
    if condition_types is None:
        condition_types = ["T1w", "T2w"]
    items = map(Tensor, items_in)

    ## Padding
    items = list(map(lambda x: pad_size(x, target_shape, padding), items))
    # Coordinate-encoding
    shape = items[0].shape
    l1 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
    l2 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
    l3 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
    l1 = Tensor(l1).permute(2, 0, 1)
    l2 = Tensor(l2).permute(0, 2, 1)
    l3 = Tensor(l3)
    assert l1.shape == l2.shape, (l1.shape, l2.shape)
    assert l3.shape == l2.shape, (l3.shape, l2.shape)
    assert shape == l2.shape, (shape, l2.shape)
    items.append(l1)
    items.append(l2)
    items.append(l3)
    ## Random crop
    items = list(random_crop(target_shape, *items))

    for i, (x, y) in enumerate(zip(items, condition_types, strict=False)):
        if y in mri_names:
            items[i] = x  # mri_transform(x)
        if y.upper() != "CT":
            items[i] = items[i] * 2 - 1
    # Random flipping
    if flip and random.random() > 0.5:
        assert False
    out = tuple(a.to(torch.float32).unsqueeze_(0) for a in items)
    try:
        return torch.cat(out, 0)
    except:
        print([o.shape for o in out])
        exit()


@torch.no_grad()
def eval_t1_t2_pix2pix(t1: Path | str | None, t2: Path | str, model: CUT | CycleGAN, gpu, two_d=False):
    t2_arr, new_aff, max_v = _load(t2)
    if t1 is None:
        list_of_items = [t2_arr]
    else:
        list_of_items = [_load(t1)[0], t2_arr]
    og_shape = list_of_items[0].shape
    target_shape = padded_shape(og_shape)
    condition2 = transform(list_of_items, target_shape=target_shape)
    with torch.no_grad():
        if gpu:
            condition2 = condition2.cuda()
            model.cuda()
        # print(two_d, condition2.shape, len(list_of_items), t1)
        if two_d:
            condition2 = condition2[:-3].swapaxes_(0, 1)
            out = (model.forward(condition2).detach().cpu().numpy()[:, 0] + 1) / 2
        else:
            out = (model.forward(condition2.unsqueeze(0)).detach().cpu().numpy()[0, 0] + 1) / 2
    out = anti_pad(out * max_v, og_shape)
    nii = nib.nifti1.Nifti1Image(out, new_aff)
    return nii, target_shape


# @torch.no_grad()
# def eval_t1_t2_diffusion(t1: Path | str, t2: Path | str, model: Diffusion, gpu, steps=500, eta=0.0, w=0):
#    t2_arr, new_aff, max_v = _load(t2)#

#    list_of_items = [_load(t1)[0], t2_arr]
#    og_shape = list_of_items[0].shape
#    target_shape = padded_shape(og_shape)
#    condition2 = transform(list_of_items, target_shape=target_shape)
#    with torch.no_grad():
#        if gpu:
#            condition2 = condition2.cuda()
#            model.cuda()
#        if len(condition2.shape) == 4:
#            condition2 = condition2.unsqueeze(0)
#        out = model.forward_ddim(1, range(0, 1000, 1000 // steps), eta=eta, x_conditional=condition2, w=w)[0]
#        out = out.squeeze().cpu().numpy()
#        torch.cuda.empty_cache()
#        # out = (model.forward(condition2.unsqueeze(0)).detach().cpu().numpy()[0, 0] + 1) / 2
#    out = anti_pad(out * max_v, og_shape)
#    nii = nib.nifti1.Nifti1Image(out, new_aff)
#    return nii, target_shape


@torch.no_grad()
def eval_pix2pix(files: list[Path | str] | list[NII], condition_types, model: CUT, gpu, reference=-1, two_d=False):
    _, new_aff, max_v = _load(files[reference])

    list_of_items = [_load(file)[0] for file in files]
    for i in list_of_items:
        if list_of_items[0].shape != i.shape:
            raise ValueError(f"{list_of_items[0].shape},{i.shape},\n\n{files}")

    og_shape = list_of_items[0].shape
    target_shape = padded_shape(og_shape)
    condition2 = transform(list_of_items, target_shape=target_shape, condition_types=condition_types)
    with torch.no_grad():
        if gpu:
            condition2 = condition2.cuda()
            model.cuda()
        if two_d:
            condition2 = condition2[:-3].swapaxes_(0, 1)
            out = model.forward(condition2).detach().cpu().numpy()
            out = [(out[:, i] + 1) / 2 for i in range(out.shape[1])]
        else:
            out = model.forward(condition2.unsqueeze(0)).detach().cpu().numpy()
            out = [(out[0, i] + 1) / 2 for i in range(out.shape[1])]
    out = [anti_pad(out * max_v, og_shape) for out in out]
    niis = [nib.nifti1.Nifti1Image(out, new_aff) for out in out]
    return niis, target_shape


def make_json_for_generated_image(
    algorithm_name,
    resolution: NII,
    json_file: BIDS_FILE,
    model_name: str,
    model_file_name: str,
    source_images: list[BIDS_FILE] | list[Path | str],
    out_pth: Path,
    logger: Logger_Interface,
    verbose=True,
):
    ref_json = json_file.open_json()
    todays_date = date.today()
    json_out = {
        "ImageType": ["DERIVED", "SECONDARY", "WATER", "STIR"],
        "ImageComments": f"Generated image by network named {model_file_name} ({model_name}); {resolution!s}",
        "GeneralReference": {
            "SourceImageSequence": [str(a.file["nii.gz"] if isinstance(a, BIDS_FILE) else str(a)) for a in source_images],
            "DerivationDescription": f"Generated image by a network named '{model_file_name}' ({model_name}) from SourceImageSequence",
            "DerivationCode": algorithm_name,
            "ReferencedImageSequence": model_file_name,
        },
        "AcquisitionDate": f"{todays_date.year:04}{todays_date.month:02}{todays_date.day:02}",
    }
    for key in [
        "StudyDate",
        "Patient",
        "Clinical Trial Subject",
        "General Study",
        "Patient Study",
        "Clinical Trial Study",
        "General Series",
        "Clinical Trial Series",
        "General Equipment",
        "SC Equipment",
        "General Image",
        "ImagePixel",
        "Device",
        "Specimen",
        "SC Image",
        "Overlay Plane",
        "Modality LUT",
        "VOI LUT",
        "ICC Profile",
        "SOP Common",
        "Common Instance Reference",
    ]:
        key = key.replace(" ", "")
        for key2 in ref_json.keys():
            key2 = str(key2)
            if key2.startswith(key):
                json_out[key2] = ref_json[key2]
    json_out["ProtocolName"] = str(ref_json["ProtocolName"]) + "_stir"
    out_pth.parent.mkdir(exist_ok=True)
    json_path = str(out_pth).replace(".nii.gz", ".json")
    with open(json_path, "w") as outfile:
        json.dump(json_out, outfile, indent=3)
        logger.print("Save", json_path, type=Log_Type.SAVE, verbose=verbose)


if __name__ == "__main__":
    pass
    # model = Pix2pix.load_from_checkpoint(
    #    "/DATA/robert/cyclegan/logs_diffusion3D/CUT_01_DIX_T1w_T2w_to_T2dixon/version_1/checkpoints/epoch=1021-step=858400_train_All=0_latest.ckpt"
    # )
    # model.eval()
    # nib.save(
    #    eval_t1_t2_pix2pix(t1, t2, model),
    #    "/DATA/NAS/ongoing_projects/robert/test2.nii.gz",
    # )  # type:ignore
