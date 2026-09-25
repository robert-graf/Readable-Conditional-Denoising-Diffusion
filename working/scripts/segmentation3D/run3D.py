import json
import os
import sys
from pathlib import Path

root = str(Path().parent.parent)
sys.path.append(root)
# from multiprocessing import Lock
import traceback
from math import ceil, floor
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
import torch
import yaml
from BIDS import BIDS_FILE, NII, BIDS_Global_info, to_nii, Centroids
from BIDS.docker.docker import run_docker_on_sample_advanced
from BIDS.logger.log_file import Log_Type, Logger, Logger_Interface, No_Logger, String_Logger, sub_log_call_func
from joblib import Parallel, delayed
from torch import Tensor
from torch.nn import functional as F
from diffusion import Diffusion
from CUT import CUT as Pix2pix
from factory.arguments import get_latest_Checkpoint

derivative_name = "derivatives"
logger = No_Logger()
max_shape = 400

loaded_models = {}


def load_model(opt, no_reload=False, verbose=True):
    checkpoint = get_latest_Checkpoint(opt.exp_name, log_dir_name=opt.log_dir_name, best=False, version=opt.version, verbose=verbose)
    assert checkpoint is not None
    if checkpoint in loaded_models:
        return loaded_models[checkpoint]
    model = None
    if checkpoint is not None and no_reload:
        return None, checkpoint  # type: ignore
    if opt.cut:
        assert checkpoint is not None, f"{opt.exp_name} - version {opt.version}, does not exist"
        model = Pix2pix.load_from_checkpoint(checkpoint, strict=False)
    elif opt.adversarial:
        assert False
    elif opt.syndiff:
        assert False
    else:
        model = Diffusion.load_from_checkpoint(checkpoint, strict=False)
    if model is not None:
        model.cuda()
        model.eval()
    loaded_models[checkpoint] = model
    return model


def map(i, p):
    if p == 0:
        return None
    if i % 2 == 0:
        return int(p)
    return -int(p)


def padded_shape(shape):
    shape = list(shape)
    for i, j in enumerate(shape):
        # if j % 8 != 0:
        shape[i] = j + 8 - (j % 8)
    return shape


def pad_size3D(x: Tensor, target_shape: list[int]):
    padding = []
    for in_size, out_size in zip(reversed(x.shape[-3:]), reversed(target_shape)):
        to_pad_size = max(0, out_size - in_size) / 2.0
        padding.extend([ceil(to_pad_size), floor(to_pad_size)])
    x_ = F.pad(x, padding, mode="constant")
    padding = [map(i, p) for i, p in enumerate(padding)]
    return x_, padding


def make_poss_embedding(shape):
    assert len(shape) == 3
    l1 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
    l2 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
    l3 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
    l1 = Tensor(l1).permute(2, 0, 1)
    l2 = Tensor(l2).permute(0, 2, 1)
    l3 = Tensor(l3)
    return l1, l2, l3


def get_ctd(file: BIDS_FILE):
    in_ctd = file.get_changed_path(
        "json", "ctd", parent=derivative_name, info={"seg": "subreg", "desc": "rostock-generated"}, make_parent=False
    )
    if in_ctd.exists():
        return BIDS_FILE(in_ctd, file.dataset)
    in_ctd = file.get_changed_path("json", "msk", parent="derivatives", info={"seg": "subreg"})
    if in_ctd.exists():
        return BIDS_FILE(in_ctd, file.dataset)
    in_ctd = file.get_changed_path("json", "ctd", parent="derivatives", info={"seg": "subreg"})
    if in_ctd.exists():
        return BIDS_FILE(in_ctd, file.dataset)

    in_ctd = file.get_changed_path("json", "ctd", parent=derivative_name, info={"mod": file.format, "seg": "subreg"}, make_parent=False)
    if in_ctd.exists():
        return BIDS_FILE(in_ctd, file.dataset)
    in_ctd = file.get_changed_path("json", "ctd", parent="derivatives", info={"mod": "T2w", "seg": "subreg"})
    if in_ctd.exists():
        return BIDS_FILE(in_ctd, file.dataset)
    in_ctd = file.get_changed_path("json", "ctd", parent="derivatives", info={"acq": "iso", "mod": "T2w", "seg": "subreg"})
    if in_ctd.exists():
        return BIDS_FILE(in_ctd, file.dataset)
    a = file.get_sequence_files()
    if "ctd_label-full" in a:
        in_ctd = a["ctd_label-full"][0].file["json"]
    if in_ctd.exists():
        return BIDS_FILE(in_ctd, file.dataset)

    if "ctd_seg-subreg" in a:
        in_ctd = a["ctd_seg-subreg"][0].file["json"]
        if in_ctd.parent.name == "derivatives":
            in_ctd.rename(Path(in_ctd.parent, file.get_path_decomposed()[2], in_ctd.name))
            in_ctd = Path(in_ctd.parent, file.get_path_decomposed()[2], in_ctd.name)
    if in_ctd.exists():
        return BIDS_FILE(in_ctd, file.dataset)
    else:
        return None


def get_iso_files(file: BIDS_FILE, save_iso_mr: bool):
    out_vert = file.get_changed_path("nii.gz", "msk", parent=derivative_name, info={"mod": file.format, "seg": "vert"}, make_parent=True)
    out_subreg = file.get_changed_path("nii.gz", "msk", parent=derivative_name, info={"mod": file.format, "seg": "subreg"})
    par = "translated/rawdata"
    out_ct_iso = file.get_changed_path("nii.gz", "ct", parent=par, info={"acq": "iso", "mod": file.format}, make_parent=False)
    mr_out_path_iso = file.get_changed_path("nii.gz", parent=par, info={"acq": "iso"}, make_parent=False) if save_iso_mr else None
    par = "translated/derivatives"
    out_vert_iso = file.get_changed_path(
        "nii.gz", "msk", parent=par, info={"acq": "iso", "mod": file.format, "seg": "vert"}, make_parent=False
    )
    out_subreg_iso = file.get_changed_path(
        "nii.gz", "msk", parent=par, info={"acq": "iso", "mod": file.format, "seg": "subreg"}, make_parent=False
    )

    out_ct_ref = BIDS_FILE(out_ct_iso, Path(file.dataset, "translated"))
    json_file = str(out_ct_iso).replace(".nii.gz", ".json")
    return out_vert, out_subreg, out_ct_iso, mr_out_path_iso, out_vert_iso, out_subreg_iso, out_ct_ref, json_file


def convert(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


@torch.no_grad()
def run_model(model, opt, conditional: Tensor, out_json, gpu, eta, w, steps, logger: Logger_Interface, depth=0):
    if depth != 4:
        padding = 48
        s = conditional.shape[-2]
        if s > max_shape:
            logger.print(
                "split the MRI in two because it is bigger than",
                max_shape,
                "- Change max_shape to increase this if you think your GPU can handel more.",
                type=Log_Type.STRANGE,
            )

            s1 = s // 2
            s1 = s1 + 8 - (s1 % 8)
            out1 = run_model(model, opt, conditional[..., : s1 + padding, :], out_json, gpu, eta, w, steps, logger, depth=depth + 1)
            out2 = run_model(model, opt, conditional[..., s1 - padding :, :], out_json, gpu, eta, w, steps, logger, depth=depth + 1)
            assert out1 is not None
            assert out2 is not None
            print(out1[..., :s1].shape, out2[..., padding:].shape)
            out = torch.cat([out1[..., :s1, :], out2[..., padding:, :]], -2)
            out_json["split-lvl"] = depth
            return out

    # try:
    out = __run_model(model, opt, conditional, out_json, gpu, eta, w, steps, logger, depth=0)
    return out
    # except RuntimeError as e:
    #    if 'out of memory' in str(e):
    #        logger.print('FAIL: ran out of memory, Change max_shape and return this script. Failed shape',conditional.shape,max_shape,type=Log_Type.WARNING)#
    #        for p in model.parameters():
    #            if p.grad is not None:
    #                del p.grad  # free some memory
    #    else:
    #        raise e


def __run_model(model, opt, conditional: Tensor, out_json, gpu, eta, w, steps, logger: Logger_Interface, depth=0):
    if isinstance(model, Diffusion):
        out_json["eta"] = eta
        out_json["w"] = w
        out_json["steps"] = steps

        logger.print(f"Run Diffusion; steps = {steps}, eta = {eta:.1f}, w = {w}")

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
            out = model.forward_ddim(
                1, range(0, model.opt.timesteps, max(model.opt.timesteps // steps, 1)), eta=eta, x_conditional=conditional, w=w
            )[0]
            out *= 2
            out -= 1
            out = out.squeeze().cpu() * 1000
            torch.cuda.empty_cache()
            return out
    elif isinstance(model, Pix2pix):
        logger.print("Run Pix2Pix")
        with torch.no_grad():
            if gpu:
                if isinstance(gpu, str):
                    model.to(gpu)
                    conditional = conditional.to(gpu)
                else:
                    model.cuda()
                    conditional = conditional.cuda()
            # print(conditional.min(), conditional.max())
            out = model.forward(conditional)
            # print(out.min(), out.max())
            logger.print("Fin Pix2Pix")
            out = out.squeeze().cpu() * 1000
            torch.cuda.empty_cache()
            return out
    else:
        raise NotImplementedError(opt)


def phase1(
    name: str,
    logger: Logger_Interface,
    mr_path: Path | str | NII,
    out_ct_iso: Path | str,
    out_ct_ref: BIDS_FILE,
    out_vert_iso: Path,
    out_subreg_iso: Path,
    out_vert: Path,
    out_subreg: Path,
    json_file: str,
    opt,
    mr_out_path_iso: Path | None = None,
    ctd_subreg_ref: BIDS_FILE | None = None,
    roll_ct=2,
    crop_initial=False,
    override_gpu=False,
    use_existing_ctd=True,
    gpu=True,
    force_rerun_docker=False,
):
    try:
        out_ct_iso = Path(out_ct_iso).absolute()

        mr_nii = to_nii(mr_path, False).clamp_(min=0)
        # import nibabel as nib

        # mr_nii.nii = nib.Nifti1Image(mr_nii.nii.get_fdata(), mr_nii.nii.affine)
        # mr_nii.clamp_(min=0).n4_bias_field_correction_()

        crop = None
        if crop_initial:
            crop = mr_nii.compute_crop_slice(minimum=10)
            mr_nii.apply_crop_slice_(crop)

        out_json = {"zoom": mr_nii.zoom, "orientation": mr_nii.orientation}
        mr_nii.rescale_((1, 1, 1), verbose=logger).reorient_(("R", "I", "P"), verbose=logger)
        # if name == "spinegan0077":
        #    print(mr_nii, mr_nii.max())
        #    mr_nii = mr_nii.clamp(min=0, max=2000)

        ### padding ###
        old = list(mr_nii.shape)
        shape = padded_shape(old)

        mr_arr, pad = pad_size3D(Tensor(mr_nii.get_array().astype(float)), shape)
        mr_nii.set_array_(mr_arr.numpy())
        out_json["padding"] = pad
        out_json["crop"] = crop
        ### Centroids
        if not use_existing_ctd:
            ctd_subreg_ref = None
        elif ctd_subreg_ref is not None:
            # Take existing centroids
            sli = tuple([slice(-pad[4], pad[5]), slice(-pad[2], pad[3]), slice(-pad[0], pad[1])])

            ctd = ctd_subreg_ref.open_ctd(mr_path).rescale_((1, 1, 1), verbose=logger).reorient_(("R", "I", "P"), verbose=logger)
            ctd.crop_centroids_(sli)
            # sub-101000_acq-iso_chunk-BWS_sequ-5029_mod-T2w_seg-subreg_ctd.json
            ctd_out = Path(str(out_subreg_iso).replace("_msk.nii.gz", "_ctd.json"))
            logger.print("using centroids:", ctd_subreg_ref, type=Log_Type.UNDERLINE)
            ctd.save(ctd_out, verbose=logger, save_hint=0, make_parents=True)
            ctd_subreg_ref = BIDS_FILE(ctd_out, str(ctd_out).split("translated/rawdata")[0])

        assert old == list(mr_arr[pad[4] : pad[5], pad[2] : pad[3], pad[0] : pad[1]].shape), (
            shape,
            old,
            mr_arr[pad[4] : pad[5], pad[2] : pad[3], pad[0] : pad[1]].shape,
            pad,
        )
        ### Normalize ###
        arr_max = np.quantile(mr_arr, 0.95)
        mr_arr = mr_arr / arr_max
        mr_arr[mr_arr >= 1] = 1
        mr_arr *= 2
        mr_arr -= 1
        ### https://arxiv.org/pdf/2303.15288.pdf ###
        l1, l2, l3 = make_poss_embedding(mr_arr.shape)
        conditional = torch.stack([mr_arr, l1, l2, l3]).unsqueeze(0)
        return (
            "phase1",
            phase2,
            {
                "conditional": conditional,
                "logger": logger,
                "out_json": out_json,
                "mr_nii": mr_nii,
                "out_ct_iso": out_ct_iso,
                "mr_out_path_iso": mr_out_path_iso,
                "mr_path": mr_path,
                "out_ct_ref": out_ct_ref,
                "out_vert_iso": out_vert_iso,
                "out_subreg_iso": out_subreg_iso,
                "out_vert": out_vert,
                "out_subreg": out_subreg,
                "ctd_subreg_ref": ctd_subreg_ref,
                "roll_ct": roll_ct,
                "override_gpu": override_gpu,
                "json_file": json_file,
                "opt": opt,
                "gpu": gpu,
                "force_rerun_docker": force_rerun_docker,
            },
        )
    except Exception:
        logger.print_error()
        logger.close()
        return None


def phase2(
    mr_out_path_iso: Path,
    out_ct_iso: Path,
    steps: int,
    eta: float,
    w: float,
    conditional: Tensor,
    logger: Logger_Interface,
    out_json: dict,
    mr_nii: NII,
    mr_path: Path | str | NII,
    out_ct_ref: BIDS_FILE,
    out_vert_iso: Path,
    out_subreg_iso: Path,
    out_vert: Path,
    out_subreg: Path,
    ctd_subreg_ref: None | BIDS_FILE,
    roll_ct: int,
    override_gpu: bool,
    json_file: str,
    opt,
    gpu: bool | str,
    force_rerun_docker: bool,
):
    model = load_model(opt)
    out_json["model"] = opt.exp_name
    try:
        out = None
        if override_gpu or not out_ct_iso.exists():
            out = run_model(model, opt, conditional, out_json, gpu, eta, w, steps, logger)
            if out is None:
                return
            if mr_out_path_iso is not None:
                mr_nii.save(mr_out_path_iso, verbose=logger, make_parents=True)
            with open(json_file, "w") as outfile:
                logger.print(out_json)
                json.dump(out_json, outfile, default=convert, indent=1)
            out = mr_nii.set_array(out.numpy(), inplace=False)
            out.save(out_ct_iso, verbose=logger, make_parents=True)
        # fix an offset of 2 issue (roll_ct is 2)
        else:
            logger.print("skip ct generation with ct. It already exits;", out_ct_iso.name)
        if roll_ct != 0:
            if Path(json_file).exists():
                with open(json_file) as f:
                    out_json = json.load(f)

            if "roll_ct" not in out_json:
                out = NII.load(out_ct_iso, False) if out is None else out
                arr = out.get_array()  # .get_fdata()  # .get_array()
                arr[:, roll_ct:] = arr[:, :-roll_ct]
                out.set_array_(arr)
                if out.get_array().dtype == np.uint8:
                    exit()
                out.save(out_ct_iso, make_parents=True)

                out_json["roll_ct"] = roll_ct
                logger.print("roll ct by", roll_ct)
                with open(json_file, "w") as outfile:
                    logger.print(out_json)
                    json.dump(out_json, outfile, default=convert, indent=1)
            elif force_rerun_docker:
                pass
            else:
                if out_vert.exists() and out_vert_iso.exists() and (os.path.getmtime(json_file) < os.path.getmtime(out_vert)):
                    logger.print("Skip roll_ct and docker", type=Log_Type.UNDERLINE)
                    if isinstance(logger, String_Logger):
                        logger.close()
                    return None

        return (
            "phase2",
            phase3,
            {
                "logger": logger,
                "mr_path": mr_path,
                "out_ct_iso": out_ct_iso,
                "out_ct_ref": out_ct_ref,
                "out_vert_iso": out_vert_iso,
                "out_subreg_iso": out_subreg_iso,
                "out_vert": out_vert,
                "out_subreg": out_subreg,
                "mr_out_path_iso": mr_out_path_iso,
                "ctd_subreg_ref": ctd_subreg_ref,
            },
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            global max_shape
            max_shape = min(max(conditional.shape), max_shape)
            logger.print(
                "FAIL: ran out of memory, skipping\n", mr_out_path_iso, "\n", conditional.shape, max_shape, "\n", str(e), type=Log_Type.FAIL
            )
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            return None
        else:
            raise e


def phase3(
    mr_path: Path | str | NII,
    out_ct_ref: BIDS_FILE,
    out_vert_iso,
    out_subreg_iso,
    out_ct_iso: Path,
    out_vert,
    out_subreg,
    logger: Logger_Interface,
    mr_out_path_iso,
    ctd_subreg_ref: BIDS_FILE | None,
    snapshot=False,
):
    try:
        arr = NII.load_bids(out_ct_ref).get_array()
    except Exception:
        exit()
    run_docker_on_sample_advanced(
        out_ct_ref, ctd_subreg_ref=ctd_subreg_ref, save_as_sourcedocker=False, cut_sacrum_ctd=False, save_log=False, logger=logger
    )
    if not Path(out_vert_iso).exists():
        logger.print("FileNotFound: derivatives were not created", type=Log_Type.FAIL)
        logger.close()
        return
    try:
        revert_ctd_to_original(
            out_subreg_iso,
            str(out_ct_iso).replace("nii.gz", "json"),
            Path(str(out_subreg).replace("msk.nii.gz", "ctd.json")),
        )
        mr_nii = to_nii(mr_path)
        revert_iso_to_original(NII.load(out_vert_iso, True), str(out_ct_iso).replace("nii.gz", "json"), out_vert, mr_nii)
        revert_iso_to_original(NII.load(out_subreg_iso, True), str(out_ct_iso).replace("nii.gz", "json"), out_subreg, mr_nii)

        logger.print(out_ct_iso.name, "successfully translated, segmented and back sampled.", type=Log_Type.OK)

        if snapshot:
            from BIDS.snapshot2D import Snapshot_Frame, Visualization_Type, create_snapshot

            path1 = Path(out_ct_ref.dataset, "snapshots", out_ct_iso.name.replace(".nii.gz", ".jpg"))
            path1.parent.mkdir(exist_ok=True)
            path2 = out_ct_ref.get_changed_path("jpg", parent="derivatives", format="snp")
            nii_ct_iso = to_nii(out_ct_iso)
            nii_ct_iso.set_array_(nii_ct_iso.nii.get_fdata())
            create_snapshot(
                [path1, path2],
                [
                    Snapshot_Frame(nii_ct_iso, out_vert_iso, centroids=get_ctd(out_ct_ref), mode="CT", coronal=True, crop_msk=True),
                    Snapshot_Frame(mr_out_path_iso, out_vert_iso, mode="MRI", coronal=True, crop_msk=True),
                    Snapshot_Frame(
                        image=nii_ct_iso,
                        segmentation=out_vert_iso,
                        mode="MINMAX",
                        sagittal=True,
                        coronal=True,
                        hide_segmentation=True,
                        only_mask_area=True,
                        visualization_type=Visualization_Type.Maximum_Intensity,
                        crop_msk=True,
                    ),
                ],
            )
    except Exception as e:
        logger.print(traceback.format_exc(), type=Log_Type.FAIL)
    if isinstance(logger, String_Logger):
        logger.close()


def revert_iso_to_original(nii: NII, json_dict: Path | str | dict, out_file: Path | None = None, org_nii: NII = None):
    """Takes an padded ISO nii.gz and reverts it to the original zoom,orientation,padding with the produced json

    Args:
        nii (NII): _description_
        json_dict (_type_): _description_
        out_file (Path | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if isinstance(json_dict, str) or isinstance(json_dict, Path):
        with open(json_dict, "r") as f:
            dic = json.load(f)
    else:
        dic = json_dict
    pad = dic["padding"]

    out = nii.get_array()
    out = out[pad[4] : pad[5], pad[2] : pad[3], pad[0] : pad[1]]
    nii = nii.set_array(out)
    nii_iso = nii.copy()
    nii.reorient_(dic["orientation"]).rescale_(dic["zoom"])
    if org_nii is not None and org_nii.shape != nii.shape:
        nii = nii.pad_to(org_nii.shape, inplace=True)
    assert org_nii is None or org_nii.shape == nii.shape, (org_nii.shape, nii.shape)
    if out_file is not None:
        nii.save(out_file, make_parents=True)
    return nii, nii_iso


def revert_ctd_to_original(out_subreg_iso: str, json_dict: Path | str | dict, out_file: Path | None = None):
    """Takes an padded ISO nii.gz and reverts it to the original zoom,orientation,padding with the produced json

    Args:
        nii (NII): _description_
        json_dict (_type_): _description_
        out_file (Path | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if isinstance(json_dict, str) or isinstance(json_dict, Path):
        with open(json_dict, "r") as f:
            dic = json.load(f)
    else:
        dic = json_dict
    nii = NII.load(out_subreg_iso, True)
    ctd = Centroids().load(str(out_subreg_iso).replace("msk.nii.gz", "ctd.json"))
    ctd.zoom = nii.zoom
    ctd.shape = nii.shape
    pad = dic["padding"]

    sl = tuple([slice(pad[4], pad[5]), slice(pad[2], pad[3]), slice(pad[0], pad[1])])
    ctd = ctd.crop_centroids(sl).round(3)
    assert ctd.shape != None and ctd.shape[0] >= 0, ctd
    ctd.reorient_(dic["orientation"]).rescale_(dic["zoom"])
    if out_file is not None:
        ctd.save(out_file, make_parents=True)
    return ctd


def mri2ct_on_sample(
    mr_file: BIDS_FILE,
    opt="diffusion1000",
    save_iso_mr=True,
    steps=50,
    eta=1.0,
    w=2,
    roll_ct=2,
    override=False,
    use_existing_ctd=True,
    snapshot=True,
    name="single",
    logger: Logger_Interface = No_Logger(),
    gpu=True,
):
    load_model(opt)
    out_vert, out_subreg, out_ct_iso, mr_out_path_iso, out_vert_iso, out_subreg_iso, out_ct_ref, json_file = get_iso_files(
        mr_file, save_iso_mr
    )
    if out_vert.exists() and not override:
        logger.print(out_vert, "exist. Skip", type=Log_Type.OK)
        return

    out = phase1(
        name=name,
        logger=logger,
        mr_path=mr_file.open_nii(),
        out_ct_iso=out_ct_iso,
        mr_out_path_iso=mr_out_path_iso,
        out_ct_ref=out_ct_ref,
        out_vert_iso=out_vert_iso,
        out_subreg_iso=out_subreg_iso,
        out_vert=out_vert,
        out_subreg=out_subreg,
        opt=opt,
        roll_ct=roll_ct,
        json_file=json_file,
        ctd_subreg_ref=get_ctd(mr_file),
        use_existing_ctd=use_existing_ctd,
        gpu=gpu,
    )
    if out is None:
        return
    _, _, args = out
    out = phase2(steps=steps, eta=eta, w=w, override_gpu=override, **args)
    if out is None:
        return
    _, _, args = out
    phase3(snapshot=snapshot, **args)


def mri2ct_on_ds(
    ds: str | Path | list[str | Path] = "/media/data/robert/datasets/spinegan_T2w_all_reg_iso/sourcedata/",
    opt="diffusion1000",
    formats=["T2w", "dixon", "mr"],
    save_iso_mr=True,
    steps=50,
    eta=1.0,
    w=2,
    roll_ct=2,
    override_gpu=False,
    override=False,
    use_existing_ctd=True,
    snapshot=True,
    max_parallel=16,
    gpu=True,
    force_rerun_docker=False,
    only_existing=False,
):
    a = []
    load_model(opt)
    if not isinstance(ds, list):
        ds = [ds]
    bids_glob = BIDS_Global_info(ds, ["rawdata", "rawdata_dixon", "derivatives"], clear=True)

    logger = Logger(ds[0], "desc-mri2ct", default_verbose=True)

    phase2_link = None
    phase3_link = None
    for name, subject in bids_glob.enumerate_subjects(sort=False):
        # logger.print(name)
        query = subject.new_query(flatten=True)
        query.filter_format(lambda x: x in formats)
        query.filter_json(
            lambda x: "W" not in x["ImageType"]
            and "F" not in x["ImageType"]
            and "FAT" not in x["ImageType"]
            and "WATER" not in x["ImageType"],
            required=False,
        )
        query.filter("rec", lambda x: "W" != x.upper() and "F" != x.upper() and "FAT" != x.upper() and "WATER" != x.upper(), required=False)
        query.filter("acq", lambda x: "W" != x.upper() and "F" != x.upper() and "FAT" != x.upper() and "WATER" != x.upper(), required=False)
        for mr_file in query.loop_list():
            try:
                out_vert, out_subreg, out_ct_iso, mr_out_path_iso, out_vert_iso, out_subreg_iso, out_ct_ref, json_file = get_iso_files(
                    mr_file, save_iso_mr
                )
                skip_all = False
                cdt = get_ctd(mr_file)
                # print(cdt)
                gpu_looked = (override_gpu or not out_ct_iso.exists()) and gpu

                if (not out_vert.exists() and cdt is None) and only_existing:
                    logger.print(out_vert.name, " is skipped because we only run on 'only_existing'. Skip", type=Log_Type.STRANGE)
                    continue
                if out_vert.exists() and out_vert_iso.exists() and not override and (not use_existing_ctd or cdt is None):
                    logger.print(out_vert.name, "exist. Skip", type=Log_Type.OK)
                    continue
                elif use_existing_ctd and cdt is not None and out_subreg.exists() and not override:
                    if os.path.getmtime(out_subreg) < os.path.getmtime(cdt.file["json"]):
                        logger.print("The centroid file was updated. So we run", out_subreg, "again", type=Log_Type.STRANGE)
                        force_rerun_docker = True
                    else:
                        continue
                else:
                    if not out_vert_iso.exists():
                        pass

                a.append(
                    delayed(sub_log_call_func)(
                        name,
                        logger,
                        phase1,
                        mr_path=mr_file.open_nii(),
                        out_ct_iso=out_ct_iso,
                        mr_out_path_iso=mr_out_path_iso,
                        out_ct_ref=out_ct_ref,
                        out_vert_iso=out_vert_iso,
                        out_subreg_iso=out_subreg_iso,
                        out_vert=out_vert,
                        out_subreg=out_subreg,
                        opt=opt,
                        roll_ct=roll_ct,
                        override_gpu=override_gpu,
                        json_file=json_file,
                        ctd_subreg_ref=cdt,
                        use_existing_ctd=use_existing_ctd,
                        gpu=gpu,
                        force_rerun_docker=force_rerun_docker,
                    )
                ) if not skip_all else None
                a.append(phase2_link) if phase2_link is not None else None
                a.append(phase3_link) if phase3_link is not None else None
                phase2_link = None
                phase3_link = None
                if gpu_looked or max_parallel < len(a):
                    out = Parallel(n_jobs=len(a), backend="threading")(a for a in a)
                    a.clear()
                    assert out is not None
                    for i in out:
                        if i is None:
                            pass
                        elif i[0] == "phase1":
                            a.append(delayed(i[1])(steps=steps, eta=eta, w=w, **i[2]))
                        elif i[0] == "phase2":
                            a.append(delayed(i[1])(snapshot=snapshot, **i[2]))
            except Exception as ex:
                logger.print(traceback.format_exc(), type=Log_Type.FAIL)
                raise ex

    out = Parallel(n_jobs=max_parallel, backend="threading")(a)
    while True:
        a.clear()
        assert out is not None
        for i in out:
            if i is None:
                continue
            elif i[0] == "phase1":
                a.append(delayed(i[1])(steps=steps, eta=eta, w=w, **i[2]))
            elif i[0] == "phase2":
                a.append(delayed(i[1])(snapshot=snapshot, **i[2]))
        out = Parallel(n_jobs=max_parallel, backend="threading")(a)
        if out is None or len(out) == 0:
            break


if __name__ == "__main__":
    # mri2ct_on_ds(
    #    "/media/data/NAKO/NAKO/MRT/",
    #    w=0,
    #    steps=25,
    #    override_gpu=False,
    #    roll_ct=2,
    #    max_parallel=16,
    #    only_existing=True,
    #    use_existing_ctd=True,
    #    force_rerun_docker=True,
    #    gpu=False,
    # )

    # mri2ct_on_ds(
    #    "/media/data/robert/datasets/spinegan_T2w_all",
    #    w=0,
    #    steps=25,
    #    override_gpu=False,
    #    roll_ct=2,
    #    max_parallel=5,
    #    gpu=True,
    #    use_existing_ctd=True
    # )
    from dataclasses import dataclass

    # @dataclass()
    # class Option:
    #    exp_name: str = "diffusion3D_T2_CT"
    #    log_dir_name: str = "logs_diffusion"
    #    version: str = "*"
    #    cut: bool = False
    #    adversarial: bool = False
    #    syndiff = False
    #
    # mri2ct_on_ds(
    #    "/media/data/old_NAKO/MRT/test/paper_104852/result_diffuion_3D/",
    #    opt=Option(),
    #    formats=["T1w", "T2w", "dixon", "mr"],
    #    w=0,
    #    steps=100,
    #    roll_ct=2,
    #    max_parallel=5,
    #    gpu=True,
    #    use_existing_ctd=True,
    #    override_gpu=False,
    #    override=True,
    # )

    @dataclass()
    class Option:
        exp_name: str = "CUT_diffusion3D_T2_CT"
        log_dir_name: str = "logs_diffusion3D"
        version: str = "*"
        ddim: bool = False
        cut: bool = True
        adversarial: bool = False
        syndiff: bool = False
        ddim: bool = False

    # mri2ct_on_ds(
    #    "/media/data/old_NAKO/MRT/test/paper_104852/result_Pix2Pix_3D/",
    #    opt=Option(),
    #    formats=["T1w", "T2w", "dixon", "mr"],
    #    w=0,
    #    steps=100,
    #    roll_ct=2,
    #    max_parallel=5,
    #    gpu=False,
    #    use_existing_ctd=True,
    #    override_gpu=False,
    #    override=True,
    # )

    o = Option(
        exp_name="dif_spinegan3D_T2_CT_img",
        log_dir_name="logs_diffusion3D",
        version="*",
        cut=False,
        adversarial=False,
        syndiff=False,
        ddim=True,
    )
    mri2ct_on_ds(
        "/media/data/old_NAKO/MRT/test/paper_104852/result_diffuion_3D_img/",
        opt=o,
        formats=["T1w", "T2w", "dixon", "mr"],
        w=0,
        steps=25,
        roll_ct=2,
        max_parallel=5,
        gpu=True,
        use_existing_ctd=True,
        override_gpu=False,
        override=True,
    )
