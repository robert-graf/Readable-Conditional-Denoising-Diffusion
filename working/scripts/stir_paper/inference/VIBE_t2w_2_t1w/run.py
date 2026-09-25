import sys
from pathlib import Path

from scripts.stir_paper.inference.eval_step import eval_pix2pix

root = str(Path().parent.parent)
sys.path.append(root)

# from multiprocessing import Lock
from typing import Literal

import torch
import yaml
from TPTBox import BIDS_FILE, NII, to_nii
from TPTBox.logger.log_file import No_Logger

from CUT import CUT
from cycleGAN import CycleGAN
from factory import arguments

available_models = Literal["Pix2Pix", "Pix2Pix2D"]
loaded_models: dict[str, CUT | CycleGAN] = {}
derivative_name = "derivatives_translated"

logger = No_Logger()


def load_model(model_name: str, model_typ: available_models):
    if model_name in loaded_models:
        return loaded_models[model_name]
    pt = arguments.get_latest_Checkpoint(model_name, log_dir_name="logs_diffusion3D")

    if pt is None:
        exit("FileNotFound" + str(model_name))
    # dic = torch.load(pt, map_location="cpu")

    if model_name == "Diffusion":
        raise NotImplementedError()
        with open(str(pt).replace(".pt", ".yaml")) as file:
            opt = arguments.Diffusion_Option()
            opt.__dict__ = yaml.full_load(file)
            in_channel = opt.in_channel  # type: ignore
        diffusion = Diffusion(opt, in_channel=in_channel)
        diffusion.load_state_dict(dic, strict=True)
        loaded_models[model_name] = diffusion
    elif "Pix2Pix" in model_typ:
        p = CUT.load_from_checkpoint(pt)
        loaded_models[model_name] = p
    elif "cycleGAN" in model_typ:
        p = CycleGAN.load_from_checkpoint(pt)
        loaded_models[model_name] = p
    else:
        raise ValueError(model_name)
    return loaded_models[model_name]


def to_t1w_name(t2w_bf: BIDS_FILE, parent: str = "derivatives_stitched", out_dataset_path: Path | str | None = None):
    return t2w_bf.get_changed_path(
        "nii.gz",
        parent=parent,
        path=str(Path(t2w_bf.get_path_decomposed()[2]).parent),
        additional_folder="T1w",
        make_parent=False,
        info={"desc": "generated"},
        bids_format="T1w",
        dataset_path=str(out_dataset_path) if out_dataset_path is not None else None,
    )


def run_VIBE_t2w_2_t1w(
    t2w_nii: NII,
    vibe: dict[str, NII],
    out_pth: Path | None = None,
    condition_names=None,
    model_name: str = "Pix2Pix",
    model_type: available_models = "Pix2Pix",
    gpu=True,
    override: bool = False,
    ori=("P", "I", "R"),
    zoom=(-1, -1, -1),
    verbose=False,
):
    if condition_names is None:
        condition_names = ["in", "opp", "water", "fat", "T2w"]
    print(t2w_nii)
    if out_pth is not None and out_pth.exists() and not override:
        return to_nii(out_pth)
    model = load_model(model_name, model_type)
    model.eval()
    try:
        if "in_vibe" in model.opt.condition_types or "inp" in model.opt.condition_types:
            t2w_nii.reorient(ori).rescale_(zoom)
            input_ = [
                vibe["inphase"].resample_from_to_(t2w_nii),
                vibe["outphase"].resample_from_to_(t2w_nii),
                vibe["water"].resample_from_to_(t2w_nii),
                vibe["fat"].resample_from_to_(t2w_nii),
                t2w_nii,
            ]
        else:
            input_ = [t2w_nii.reorient(ori).rescale_(zoom)]
        out_nib, _ = eval_pix2pix(input_, condition_names, model, gpu, two_d=not model.opt.volumes)
        out_nib = out_nib[0]

    except ValueError:
        logger.print_error()
        return
    except Exception:
        logger.print_error()
        return
    nii = NII(out_nib)
    nii.set_dtype_("smallest_int")
    if out_pth is not None:
        out_pth.parent.mkdir(exist_ok=True, parents=True)
        nii.save(out_pth)

    return nii
