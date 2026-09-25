from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from pl_models.diffusion.diffusion import Diffusion
from train3D import Diffusion3D
from utils.nii import NII


def make_poss_embedding(shape):
    assert len(shape) == 3
    l1 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
    l2 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
    l3 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
    l1 = Tensor(l1).permute(2, 0, 1)
    l2 = Tensor(l2).permute(0, 2, 1)
    l3 = Tensor(l3)
    return l1, l2, l3


@torch.no_grad()
def run_model(model: Diffusion | Diffusion3D, conditional: Tensor, gpu, eta, w, steps, depth=0, max_shape=460 * 460 * 70):
    assert conditional.max() < 1.3
    assert len(conditional.shape) == 3
    l1, l2, l3 = make_poss_embedding(conditional.shape)
    conditional_ = torch.stack([conditional, l1, l2, l3]).unsqueeze(0)
    if depth != 4:
        padding = 48
        s = conditional.shape[-2]
        vol = conditional.shape[-1] * conditional.shape[-2] * conditional.shape[-3]
        if vol > max_shape:
            print(
                "split the MRI in two because it is bigger than",
                max_shape,
                "- Change max_shape to increase this if you think your GPU can handel more.",
                conditional.shape,
                "vol=",
                vol,
            )
            s1 = s // 2
            s1 = s1 + 8 - (s1 % 8)
            out1 = run_model(model, conditional[..., : s1 + padding, :], gpu, eta, w, steps, depth=depth + 1)
            out2 = run_model(model, conditional[..., s1 - padding :, :], gpu, eta, w, steps, depth=depth + 1)
            assert out1 is not None
            assert out2 is not None
            print(out1[..., :s1].shape, out2[..., padding:].shape)
            out = torch.cat([out1[..., :s1, :], out2[..., padding:, :]], -2)
            return out
    out = __run_model(model, conditional_, gpu, eta, w, steps, depth=0)
    return out


def __run_model(model: Diffusion | Diffusion3D, conditional: Tensor, gpu, eta, w, steps, depth=0):
    print(f"Run Diffusion; steps = {steps}, eta = {eta:.1f}, w = {w}")
    with torch.no_grad():
        if gpu:
            if isinstance(gpu, str):
                model.to(gpu)
                conditional = conditional.to(gpu)
            else:
                model.cuda()
                conditional = conditional.cuda()
        else:
            model.cpu()
            conditional = conditional.cpu()
        if isinstance(model, Diffusion3D):
            model = model.diffusion_net
        out = model.forward_ddim(1, range(0, 1000, 1000 // steps), eta=eta, x_conditional=conditional, w=w)[0]
        out *= 2
        out -= 1
        out = out.squeeze().cpu() * 1000
        torch.cuda.empty_cache()
        return out


def padded_shape(shape):
    shape = list(shape)
    for i, j in enumerate(shape):
        # if j % 8 != 0:
        shape[i] = j + 8 - (j % 8)
    return shape


def signed_pad(i, p):
    if p == 0:
        return 0
    return int(p) if i % 2 == 0 else -int(p)


def pad_size3D(x: Tensor, target_shape: list[int] | None = None):
    if target_shape is None:
        target_shape = padded_shape(x.shape)
    padding = []
    for in_size, out_size in zip(reversed(x.shape[-3:]), reversed(target_shape), strict=False):
        to_pad = max(0, out_size - in_size) / 2.0
        padding.extend([floor(to_pad), ceil(to_pad)])  # left, right order

    x_padded = F.pad(x, padding, mode="constant")
    signed = [signed_pad(i, p) for i, p in enumerate(padding)]
    return x_padded, signed


def revert_iso_to_original(nii_iso: NII, nii: NII | None, padding, out_file: Path | None = None):
    pad = [p if p is not None else 0 for p in padding]
    arr = nii_iso.get_array()
    D, H, W = arr.shape[-3:]

    arr = arr[
        pad[4] : D - pad[5],
        pad[2] : H - pad[3],
        pad[0] : W - pad[1],
    ]

    nii_iso = nii_iso.set_array(arr)

    if nii is not None:
        nii_iso.reorient_(nii.orientation).rescale_(nii.zoom)
        nii_iso.pad_to(nii.shape, inplace=True)

    if out_file is not None:
        nii_iso.set_dtype_("smallest_int").save(out_file, make_parents=True)

    return nii_iso
