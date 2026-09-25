import sys
from pathlib import Path

root = str(Path().parent.parent.parent.parent)
sys.path.append(root)

# from multiprocessing import Lock
from typing import Literal

import torch
import yaml
from BIDS import BIDS_FILE, NII, to_nii
from BIDS.logger.log_file import Log_Type, No_Logger

import factory.arguments as arguments
from scripts.stir_paper.inference.eval_step import eval_t1_t2_pix2pix

from CUT import CUT
from cycleGAN import CycleGAN

available_models = Literal["Pix2Pix", "Pix2Pix2D"]
loaded_models: dict[str, CUT | CycleGAN] = {}
derivative_name = "derivatives_translated"

logger = No_Logger()


def load_model(model_name: str, model_typ: available_models):
    if model_name in loaded_models:
        return loaded_models[model_name]
    pt = arguments.get_latest_Checkpoint(model_name, log_dir_name="logs_diffusion3D")

    if pt is None:
        raise FileNotFoundError(str(model_name))
    dic = torch.load(pt, map_location="cpu")

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
        raise ValueError(model_name + "  " + model_typ)
    return loaded_models[model_name]


def to_stir_name(t2w_bf: BIDS_FILE, parent: str = "derivatives_stitched", out_dataset_path: Path | str | None = None):
    return t2w_bf.get_changed_path(
        "nii.gz",
        parent=parent,
        path=str(Path(t2w_bf.get_path_decomposed()[2]).parent),
        additional_folder="T2w",
        make_parent=False,
        info={"desc": "generated"},
        format="STIR",
        dataset_path=str(out_dataset_path) if out_dataset_path is not None else None,
    )


@torch.no_grad()
def run_t1w_t2w_2_stir(
    t1w_nii: NII | None,
    t2w_nii: NII,
    out_pth: Path | None = None,
    model_name: str = "Pix2Pix",
    model_type: available_models = "Pix2Pix",
    gpu=True,
    override: bool = False,
    ori=("P", "I", "R"),
    zoom=(-1, -1, -1),
):
    print(t2w_nii)
    if out_pth is not None and out_pth.exists() and not override:
        return to_nii(out_pth)
    model = load_model(model_name, model_type)
    model.eval()
    try:
        if "pix2pix" in model_type.lower() or "cyclegan" == model_type.lower():
            if t1w_nii is not None:
                t1w_nii = t1w_nii.reorient(ori).rescale_(zoom)
            out_arr, _ = eval_t1_t2_pix2pix(t1w_nii, t2w_nii.reorient(ori).rescale_(zoom), model, gpu, two_d=not model.opt.volumes)
        else:
            raise NotImplementedError(model_name)
            # logger.print(f"Run Diffusion; steps = {steps}, eta = {eta:.1f}, w = {w}")
            out_arr, _ = eval_t1_t2_diffusion(t1w_nii.reorient(ori).rescale_(zoom), t2w_nii.reorient(ori).rescale_(zoom), model, gpu)

    except ValueError:
        logger.print_error()
        return
    except Exception:
        logger.print_error()
        return
    nii = NII(out_arr)
    nii.set_dtype_("smallest_int")
    if out_pth is not None:
        out_pth.parent.mkdir(exist_ok=True, parents=True)
        nii.save(out_pth, verbose=True)
    return nii


# def run_t1w_t2w_2_w(
#    dataset_path="/DATA/NAS/datasets_processed/SHIP/",
#    t1w_names=["T1w"],
#    t2w_names=["T2w"],
#    model_name: available_models = "Pix2Pix",
#    gpu=True,
# ):
#    dataset = BIDS_Global_info([dataset_path], parents=["rawdata"])
#    format_names = t1w_names + t2w_names
#    for name, subj in dataset.enumerate_subjects():
#        logger.print(name)
#        pass
#        q = subj.new_query(flatten=True)
#        q.filter("sequ", lambda x: "Composed" not in str(x))
#        q.filter_format(lambda x: x in format_names)
#        q.filter_json(lambda x: "ImageType" in x and "W" not in x["ImageType"] and "F" not in x["ImageType"], required=False)
#        q.filter("rec", "in", required=False)
#        q_t1w = q.copy()
#        q_t1w.filter_format(lambda x: x in t1w_names)
#        q_t2w = q.copy()
#        q_t2w.filter_format(lambda x: x in t2w_names)#

#        for t2w in q_t2w.loop_list():
#            logger.print(t2w.file)
#            t2w_nii = t2w.open_nii()
#            max_v = 0
#            t1w_nii_max = None
#            for t1w in q_t1w.loop_list():
#                if "res" in t1w.info:
#                    if t1w.info["res"] != t2w.info.get("res"):
#                        continue
#                t1w_nii = t1w.open_nii()
#                v = t1w_nii.get_intersecting_volume(t2w_nii)#

#                if v > max_v:
#                    max_v = v
#                    t1w_nii_max = t1w
#            if t1w_nii_max is None:
#                logger.print(t2w, "found no intersecting t1w", type=Log_Type.NEUTRAL)
#                continue
#            load_model(model_name)
#            model = loaded_models[model_name]
#            model.eval()
#            assert isinstance(model, Pix2pix)#

#            out_arr, _ = eval_t1_t2_pix2pix(t1w_nii_max.file["nii.gz"], t2w.file["nii.gz"], model, gpu)
#            out_arr = NII(out_arr)
#            t2w_nii = t2w.open_nii()
#            orientation = t2w_nii.orientation
#            out_arr.reorient_(orientation, verbose=logger)
#            # translated/rawdata/#

#            model_file_name = paths[model_name]
#            source_images = [t1w_nii_max, t2w]
#            parent = t2w.get_parent()#

#            out_pth = Path(
#                str(t2w.get_changed_path(format="STIR", info={"desc": "generated"}, parent=parent, make_parent=False)).replace(
#                    "/T2w/", "/STIR/"
#                )
#            )
#            out_arr.set_dtype_("smallest_int")
#            out_arr.save(out_pth)
#            make_json_for_generated_image("run_t1w_t2w_2_w", out_arr, t2w, model_name, model_file_name, source_images, out_pth, logger)


if __name__ == "__main__":
    t1w = to_nii("/DATA/NAS/datasets_processed/SHIP/rawdata/10105/sub-10105021/T1w/sub-10105021_sequ-T1w-2_T1w.nii.gz", False)
    t2w = to_nii("/DATA/NAS/datasets_processed/SHIP/rawdata/10105/sub-10105021/T2w/sub-10105021_sequ-T2w-2_T2w.nii.gz", False)
    out = Path("/DATA/NAS/datasets_processed/SHIP/rawdata/10105/sub-10105021/STIR/test.nii.gz")
    run_t1w_t2w_2_stir(t1w, t2w, out, "Diffusion", override=True)
