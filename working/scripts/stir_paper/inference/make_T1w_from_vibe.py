import os
from pathlib import Path
import pickle
from typing import Literal
from BIDS import NII, BIDS_Global_info, BIDS_Family, BIDS_FILE, Subject_Container, to_nii, Image_Reference
from BIDS.stitching import stitching, GNC_stitch_T2w
from BIDS.registration import registrate_nipy, apply_registration_nipy
from nipy.algorithms.registration.affine import Affine
import sys


sys.path.append(str(Path(__name__).parent.parent.parent.parent))
from scripts.stir_paper.inference.VIBE_t2w_2_t1w.run import load_model as load_model2, to_t1w_name, run_VIBE_t2w_2_t1w
from scripts.stir_paper.inference.match_space import make_and_load_stitched_vibe, make_stitched_T2w, make_stitched_T1w
from scripts.stir_paper.inference.t1w_t2w_2_w.run import load_model as load_model1, to_stir_name, run_t1w_t2w_2_stir
from BIDS.snapshot2D.snapshot_modular import create_snapshot, Snapshot_Frame

import numpy as np
from BIDS import No_Logger, Log_Type

log = No_Logger()
log.override_prefix = "T"
available_models = Literal["Pix2Pix", "Pix2Pix2D"]


def crop_left_right(p: Image_Reference, th=0.75, min_val=40):
    p_nii = NII.load(p, True)
    log.print(p_nii)
    p_nii = p_nii.clamp_(min_val, min_val + 1) - min_val
    # p_nii.dilate_msk_(2).erode_msk_(5)
    assert p_nii.orientation == ("P", "I", "R")
    arr = p_nii.get_array()
    arr[...] = arr.max(axis=0, keepdims=True)
    p_nii.set_array_(arr)
    arr2 = arr.sum(axis=1).sum(axis=0)
    m = arr2.max()

    arr[:, :, (arr2 / m) <= th] = 0
    # p_nii.save(out)
    return p_nii.compute_crop_slice()


def transform_to_stir(
    subject: Subject_Container,
    model_name: str | None,
    model_type: available_models,
    buffer_path="/DATA/NAS/datasets_processed/NAKO/notes/vibe2T2w_registration.pkl",
    out_dataset_path: Path | None = None,
    save_vibe_reg=True,
    override_vibe_stitching=False,
    override_vibe_reg=True,
    override_t1w=True,
    override_stir=True,
    override_snap=True,
    native_mode=False,
    derivative="derivatives_stir",
    no_gpu=False,
    use_real_T1w: str | bool = True,
):
    try:
        if native_mode:
            derivative += "_native"
        log.print("Start", subject.name, type=Log_Type.Yellow)

        ### Load and stitch T2w ###
        t2w_path = make_stitched_T2w(subject, derivative)
        if out_dataset_path is None:
            out_dataset_path = t2w_path.dataset
        ### STIR PATH ###
        stir_path = to_stir_name(t2w_path, parent=derivative, out_dataset_path=out_dataset_path)
        if not override_t1w and not override_vibe_reg and not override_stir and stir_path.exists():
            log.print(stir_path, "exists. skip", type=Log_Type.OK)
            return
        ### Registrate T2w and VIBE ###
        if Path(buffer_path).exists():
            reg_buffer = pickle.load(open(buffer_path, "rb"))
        else:
            reg_buffer = {}
        ### Load and stitch VIBE ###
        stitched_vibe = make_and_load_stitched_vibe(subject)
        if stitched_vibe is None:
            return
        T = None
        key = str(stitched_vibe["outphase"].get("sub"))

        def compute_T(transform):  # type: ignore
            if transform is None:
                if key in reg_buffer:
                    transform: Affine = reg_buffer[key]
                    if np.abs(transform.translation).sum() >= 100:
                        transform = None  # type: ignore
                if transform is None:
                    out, transform, others = registrate_nipy(
                        to_nii(stitched_vibe["outphase"]), stitched_t2w
                    )  # other_moving=[stitched_vibe[i] for i in ["inphase", "water", "fat"]
                    if np.abs(transform.translation).sum() >= 100:
                        out, transform, others = registrate_nipy(to_nii(stitched_vibe["outphase"]), stitched_t2w, similarity="pmi")
                    if np.abs(transform.translation).sum() >= 100:
                        log.print("Cant register", stitched_vibe["outphase"], type=Log_Type.FAIL)
                        return transform
                    reg_buffer[key] = transform
                    pickle.dump(reg_buffer, open(buffer_path, "wb"))
                log.print(transform)

            return transform

        t2w_cropped_path = t2w_path.get_changed_path("nii.gz", parent=derivative, info={"res": "T1w", "chunk": None})
        if not t2w_cropped_path.exists() or override_vibe_reg:
            stitched_t2w = to_nii(t2w_path, True)
            stitched_t2w = stitched_t2w.reorient(("P", "I", "R"))
            if not native_mode:
                stitched_t2w.rescale_((1.1, 1.1, -1))
            crop = crop_left_right(stitched_t2w)
            stitched_t2w.apply_crop_slice_(crop)
            stitched_t2w = stitched_t2w.apply_crop_slice_((slice(None), slice(None), slice(2, -2)))

            log.print("CROP HEAD")
            ### CROP HEAD ###
            T = compute_T(T)
            out = apply_registration_nipy(to_nii(stitched_vibe["outphase"]), stitched_t2w, T)
            crop = out.compute_crop_slice(minimum=20)
            stitched_t2w.apply_crop_slice_(crop)
            stitched_t2w.save(t2w_cropped_path)
            stitched_t2w_cropped = stitched_t2w
            log.print("apply registration vibe")
        else:
            stitched_t2w_cropped = to_nii(t2w_cropped_path)
        #################
        stitched_vibe_reg: dict[str, NII] = {}
        for key, vibe in stitched_vibe.items():
            out_p = vibe.get_changed_path("nii.gz", parent=derivative, info={"res": "T1w", "chunk": None})
            if out_p.exists() and not override_vibe_reg:
                out = to_nii(out_p)
            else:
                T = compute_T(T)
                out = apply_registration_nipy(to_nii(vibe), stitched_t2w_cropped, T)
                out.save(out_p) if save_vibe_reg else None
            stitched_vibe_reg[key] = out
        log.print("Make T1w from T2w and VIBE", type=Log_Type.Yellow)
        ### Make T1w from T2w and VIBE ###
        if no_gpu:
            return

        if isinstance(use_real_T1w, str):
            t1w_path = to_t1w_name(t2w_path, parent=derivative, out_dataset_path=out_dataset_path)
            t1w_nii = run_VIBE_t2w_2_t1w(
                stitched_t2w_cropped,
                stitched_vibe_reg,
                t1w_path,
                model_name=use_real_T1w,
                model_type=model_type,
                override=override_t1w,
                verbose=True,
            )  # TODO
        elif use_real_T1w:
            t1w_path = make_stitched_T1w(subject, derivative)
            t1w_nii = to_nii(t1w_path)
        else:
            t1w_path = None
            t1w_nii = None
        if model_name is not None:
            log.print("Make STIR from t1w t2w", type=Log_Type.Yellow)
            run_t1w_t2w_2_stir(
                t1w_nii, stitched_t2w_cropped, stir_path, model_name=model_name, model_type=model_type, override=override_stir
            )  # TODO
        if native_mode:
            snap = Path(out_dataset_path, derivative, "snapshot_native")
        else:
            snap = Path(out_dataset_path, derivative, "snapshot")
        snap.mkdir(exist_ok=True)
        snap = Path(snap, stir_path.name.replace("nii.gz", "jpg"))
        if not snap.exists() or override_snap:
            import gc, torch

            gc.collect()
            torch.cuda.empty_cache()
            log.print("Make SNAP-SHOT", type=Log_Type.Yellow)
            C = stitched_t2w_cropped
            create_snapshot(
                snap,
                frames=[
                    Snapshot_Frame(stitched_t2w_cropped),
                    Snapshot_Frame(t1w_nii if t1w_nii is not None else C * 0),
                    Snapshot_Frame(stir_path if model_name is not None else C * 0),
                    Snapshot_Frame(stitched_vibe_reg["inphase"]),
                    Snapshot_Frame(stitched_vibe_reg["outphase"]),
                    Snapshot_Frame(stitched_vibe_reg["water"]),
                    Snapshot_Frame(stitched_vibe_reg["fat"]),
                ],
            )

    except EOFError:
        try:
            [i.file["nii.gz"].unlink() for i in stitched_vibe.values]  # type: ignore
        except:
            pass
        try:
            t2w_path.unlink()  # type: ignore
        except:
            pass
        try:
            t2w_cropped_path.unlink()  # type: ignore
        except:
            pass
        try:
            STIR_path.unlink()  # type: ignore
        except:
            pass
    except Exception:
        log.print_error()


import pandas as pd


def run_on_test_ship(
    model_name: str | None,
    use_real_T1w: bool | str,
    model_type: available_models,
    device="3",
    glob=None,
    ds="/DATA/NAS/ongoing_projects/robert/paper/paper_T1w/",
    xlsx="/DATA/NAS/datasets_processed/SHIP/code/train_paired.xlsx",
):
    # use_real_T1w == False means don't use T1w at all.
    # use_real_T1w is str means use this network, with the model_type model_type
    if model_name is not None:
        name = model_name
    else:
        name = ""
    if isinstance(use_real_T1w, str):
        name += f"____{use_real_T1w}"

    dataset = Path(ds, name)
    dataset.mkdir(exist_ok=True, parents=True)
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    from joblib import Parallel, delayed

    log.print("run_on_test_and_val", type=Log_Type.Yellow)
    df = pd.read_excel(xlsx)
    names = [str(p).rsplit("-", maxsplit=1)[-1] for p in list(df.Path)]
    phases = list(df.Phase)
    name_2_phase = {n: p for n, p in zip(names, phases)}
    log.print("parse SHIP", type=Log_Type.Yellow)
    if glob is None:
        glob = BIDS_Global_info(datasets=["/DATA/NAS/datasets_processed/SHIP/"], parents=["rawdata"])
    tasks = []
    load_model1(model_name, model_type) if model_name is not None else None
    load_model2(use_real_T1w, model_type) if isinstance(use_real_T1w, str) else None
    for name, sub in glob.enumerate_subjects(sort=True):
        if name_2_phase.get(name, "train") == "test":
            d = delayed(transform_to_stir)(
                sub,
                model_name,
                model_type,
                "/DATA/NAS/datasets_processed/NAKO/notes/vibe2T2w_registration.pkl",
                out_dataset_path=dataset,
                native_mode=True,
                override_vibe_reg=False,
                override_t1w=False,
                override_stir=False,
                use_real_T1w=use_real_T1w,
            )
            tasks.append(d)
            if len(tasks) == 64:
                Parallel(n_jobs=1, backend="threading")(a for a in tasks)
                tasks = []
    Parallel(n_jobs=1, backend="threading")(a for a in tasks)


def run_on_older_than_on_nako(min_age=60, datasets="/DATA/NAS/datasets_processed/NAKO/MRT"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    glob = BIDS_Global_info(datasets=[datasets], parents=["rawdata"])  # "derivatives_stir"
    from joblib import Parallel, delayed

    job_list = []
    for name, subject in glob.enumerate_subjects(sort=True):
        t2w_s_q = subject.new_query(flatten=True)
        t2w_s_q.filter("desc", lambda x: False, required=False)
        t2w_s_q.filter_format("T2w")
        t2w_s_q.filter("chunk", "BWS")
        try:
            p_age = list(t2w_s_q.loop_list())[0].open_json()["PatientAge"]
            p_age = int(p_age.replace("Y", ""))
            if p_age >= min_age:
                x = list(to_stir_name(list(t2w_s_q.loop_list())[0], "derivatives_stir").parent.glob("*_STIR.nii.gz"))
                if len(x) != 0:
                    log.print(name, "exists; age = ", p_age, type=Log_Type.OK)
                    continue
                log.print("run", name, "; age = ", p_age, type=Log_Type.Yellow)
                job = delayed(transform_to_stir)(
                    subject,
                    "/DATA/NAS/datasets_processed/NAKO/notes/vibe2T2w_registration.pkl",
                    native_mode=False,
                    override_vibe_reg=False,
                    override_t1w=False,
                    override_stir=False,
                    # no_gpu=True,
                )
                job_list.append(job)
            else:
                log.print(name, "to young; age = ", p_age, type=Log_Type.OK)
            if len(job_list) >= 32:
                Parallel(n_jobs=8)(job_list)
                job_list.clear()
                job_list = []
                import gc, torch

                gc.collect()
                torch.cuda.empty_cache()
        except IndexError:
            log.print(name, "IndexError", type=Log_Type.FAIL)
        except Exception:
            log.print_error()
    log.print("remaining jobs", len(job_list), type=Log_Type.STRANGE)
    Parallel(n_jobs=8)(job_list)
    job_list.clear()
    job_list = []


if __name__ == "__main__":
    ### TABLE SHIP T2w + [T1-Dixon] -> T1w ###
    model_names = [
        (None, "cGAN_02b_SHIP_vibe_T2w_to_T1w_2D", "cycleGAN"),
        (None, "CUT_02b_SHIP_vibe_T2w_to_T1w", "Pix2Pix"),  # T2w, T1dixon -> T1w
        (None, "CUT_02b_SHIP_vibe_T2w_to_T1w_2D", "Pix2Pix2D"),  # T2w, T1dixon -> T1w 2D
        (None, "CUT_02c_SHIP_T2w_to_T1w", "Pix2Pix"),  # T2w -> T1w
        (None, "CUT_02b_SHIP_T2w_to_T1w_2D", "Pix2Pix2D")  # T2w -> T1w
        # TODO add cycleGAN
    ]
    glob = BIDS_Global_info(datasets=["/DATA/NAS/datasets_processed/SHIP/"], parents=["rawdata"])

    for model_name, use_real_T1w, model_type in model_names:
        run_on_test_ship(model_name=model_name, use_real_T1w=use_real_T1w, model_type=model_type, glob=glob)
