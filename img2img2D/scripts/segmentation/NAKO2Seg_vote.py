from __future__ import annotations
from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))

import torch
from dataloader.dataloader_nii_seg import nii_Dataset, nii_Datasets
import reload_any
from reload_any import Reload_Any_Option as ROpt, ResamplingType

from scripts.segmentation.seg_functions import run_model
import nibabel as nib
import numpy as np
import shutil
from scipy.stats import mode
import threading
from typing import Callable, Literal, overload


def get_vert_path(ct_file):
    file_seg_ct = str(ct_file).replace("rawdata", "derivatives").rsplit("_", maxsplit=1)[0] + "_seg-vert_msk.nii.gz"
    return Path(file_seg_ct)


def make_snapshot_help(file_ct, file_mri, rawdata_folder):
    from BIDS.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot

    file_seg_ct = get_vert_path(file_ct)
    mod = str(file_mri).rsplit("_", maxsplit=1)[-1].split(".", maxsplit=1)[0]
    a = [
        Snapshot_Frame(image=file_ct, segmentation=file_seg_ct, centroids=None, coronal=True, axial=True, mode="CT"),
        Snapshot_Frame(image=file_mri, segmentation=file_seg_ct, centroids=None, coronal=True, axial=True, mode="MRI"),
    ]

    # id = Path(file_ct).stem.replace("_ct.nii", "")
    id = Path(file_ct).stem.replace("_ct.nii", f"_mod-{mod}_snapshot.jpg")
    # sub-fxclass0002_sequ-2_snapshot.png
    folder = Path(rawdata_folder.parent) / "snapshot" / id
    folder.parent.mkdir(exist_ok=True)
    try:
        create_snapshot(
            [folder, Path(file_ct).parent / id],
            a,
            crop=True,
        )
    except FileNotFoundError as e:
        print(str(e))
    except ValueError as e:
        print(
            id,
        )
        print([folder, Path(file_ct).parent / id])
        raise e


def make_snapshot(file_ct_names: list[Path], file_mr_names: list[Path], rawdata_folder, n_jobs=16):
    from joblib import Parallel, delayed

    #### Compute Things
    if n_jobs > 1:
        print("[*] Running {} parallel jobs. Note that stdout will not be sequential".format(n_jobs))
    Parallel(n_jobs=n_jobs)(
        delayed(make_snapshot_help)(file_ct, file_mri, rawdata_folder)
        for file_ct, file_mri in zip(file_ct_names, file_mr_names)
    )


def force_round(im_path: str | Path | nib.Nifti1Image):
    if isinstance(im_path, nib.Nifti1Image):
        nii = im_path
    else:
        nii: nib.Nifti1Image = nib.load(str(im_path))
    arr = np.round(np.asanyarray(nii.dataobj, dtype=np.float32)).astype(np.uint8)
    return nib.Nifti1Image(arr, nii.affine, nii.header)


@overload
def segment_with_network(
    opt: ROpt,
    threading_post: Literal[True],
    snap=False,
    match_string="*-1_t2.nii*",
    no_translation=False,
    # fix_rotation_nako=False,
    **args,
) -> threading.Thread:
    ...


@overload
def segment_with_network(
    opt: ROpt,
    threading_post: Literal[False] = False,
    snap=False,
    match_string="*-1_t2.nii*",
    no_translation=False,
    # fix_rotation_nako=False,
    **args,
) -> tuple[list[Path], list[Path], Path]:
    ...


def get_dataset(opt: ROpt, match_string) -> nii_Datasets:
    if opt.resample == ResamplingType.ISO:
        return nii_Datasets(root=opt.root, match_string=match_string, keep_scale=[], min_value=0)
    elif opt.resample == ResamplingType.SAGITTAL:
        return nii_Datasets(root=opt.root, match_string=match_string, keep_scale=["R"], min_value=0)
    elif opt.resample == ResamplingType.NATIVE:
        return nii_Datasets(root=opt.root, match_string=match_string, keep_scale=("R", "I", "P"), min_value=0)
    else:
        raise NotImplementedError(opt.resample.name)


def segment_with_network(
    opt: ROpt,
    threading_post=False,
    snap=False,
    match_string="*-1_t2.nii*",
    no_translation=False,
    # fix_rotation_nako=False,
    **args,
) -> tuple[list[Path], list[Path], Path] | threading.Thread:
    dataset = get_dataset(opt, match_string)
    # if fix_rotation_nako:
    #    for i, nii_ds in enumerate(dataset):  # type: ignore
    #        nii_ds: nii_Dataset
    #        if all([nii_ds.nii.affine[0, 0] != 1, nii_ds.nii.affine[1, 1] != 1, nii_ds.nii.affine[2, 2] != 1]):
    #            continue
    #        import warnings

    #        warnings.warn(
    #            f"[!]\n{nii_ds.file}\n[!]This T2 images was attempted to be fix. This may cause issues!", UserWarning
    #        )
    #        f_nii_ref = nii_ds.file.replace("run-1_t2.nii.gz", "chunk-HWS_run-29_t2.nii.gz")
    #        nii_ref: nib.Nifti1Image = nib.load(f_nii_ref)
    #        nii_ds.nii
    #        nib.save(nii_ds.nii, nii_ds.file.replace("_t2.nii.gz", "backup.nii.gz"))
    #        nib.save(nib.Nifti1Image(nii_ds.nii.get_fdata()[:, ::-1], nii_ref.affine, nii_ref.header), nii_ds.file)
    #        print("[!]rotated and second axis swapped")
    #        # assert nii_ds.nii.shape[0] <= 100
    #    dataset = nii_Datasets(root=opt.root, match_string=match_string)
    file_ct_translated, file_mr_names, rawdata_folder = run_model(opt, dataset, no_translation=no_translation, **args)
    if threading_post:
        if no_translation:
            t = threading.Thread(target=lambda: 1)
        else:
            print("[>] start docker in an other thread.")
            t = threading.Thread(
                target=segment_and_post_processing,
                args=(file_ct_translated, file_mr_names, rawdata_folder, snap, opt, True),
            )
        t.start()
        return t

    if no_translation:
        return file_ct_translated, file_mr_names, rawdata_folder
    else:
        return segment_and_post_processing(file_ct_translated, file_mr_names, rawdata_folder, snap, opt, True)


def segment_and_post_processing(
    file_ct_translated: list[Path],
    file_mr_names: list[Path],
    rawdata_folder: Path,
    snap: bool,
    opt: ROpt,
    verbose=True,
) -> tuple[list[Path], list[Path], Path]:

    for i, mr in zip(file_ct_translated, file_mr_names):

        vert_path = get_vert_path(i)
        if opt.override:
            vert_path.unlink(missing_ok=True)
            Path(str(vert_path).replace("_seg-vert_msk.nii.gz", "_seg-subreg_msk.nii.gz")).unlink(missing_ok=True)
            Path(str(vert_path).replace("_seg-vert_msk.nii.gz", "_snapshot.png")).unlink(missing_ok=True)
            if not opt.keep_cdt:
                Path(str(vert_path).replace("_seg-vert_msk.nii.gz", "_seg-subreg_ctd.json")).unlink(missing_ok=True)
        if opt.cdt_from is not None:
            dst_cdt = str(vert_path).replace("_seg-vert_msk.nii.gz", "_seg-subreg_ctd.json")
            src_cdt = dst_cdt.replace(opt.get_result_folder_name(), opt.get_cdt_from_name())

            if Path(src_cdt) == Path(dst_cdt):
                pass
            elif Path(src_cdt).exists():
                print("[*] Copy from existing cdt", src_cdt)
                Path(dst_cdt).parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(src_cdt, dst_cdt)
            else:
                print("[*] CDT Copy does not existing", src_cdt)

    ### RUN DOCKER ###
    code = reload_any.run_docker(rawdata_folder.parent, verbose=verbose)
    ### RUN DOCKER ###

    # FIX Rounding errors, may be fixed with future docker ##
    # for i in file_ct_translated:
    #    vert_path = get_vert_path(i)
    #    nib.save(force_round(vert_path), vert_path) if vert_path.exists() else None
    #    sub_path = str(vert_path).replace("_seg-vert_msk.nii.gz", "_seg-subreg_msk.nii.gz")
    #    nib.save(force_round(sub_path), sub_path) if Path(sub_path).exists() else None

    if snap and code == 0:
        print("[*] Make snapshots")
        make_snapshot(file_ct_translated, file_mr_names, rawdata_folder)
    return file_ct_translated, file_mr_names, rawdata_folder


def majority_vote(models_opt: list[ROpt], opt: ROpt, n_jobs=16):
    file_ct_translated = None
    for opt_tmp in models_opt:
        print("#############################################################################")
        print(opt_tmp.get_result_folder_name())
        file_ct_translated, _, _ = segment_with_network(opt_tmp, match_string="*-1_t2.nii*", fix_rotation_nako=True)
    assert file_ct_translated is not None
    print("#############################################################################")
    print("#############################################################################")
    print("Majority Vote - multithreading")
    from joblib import Parallel, delayed

    # Run Majority Vote
    Parallel(n_jobs=n_jobs)(delayed(majority_vote_help)(fake_ct, opt) for fake_ct in file_ct_translated)
    # generate subreg with docker
    code = reload_any.run_docker(opt.get_folder())
    # FIX Rounding errors, may be fixed with future docker ##
    for fake_ct in file_ct_translated:
        dst_ct = Path(opt.get_folder_rawdata(), fake_ct.name)
        vert_path = get_vert_path(dst_ct)
        nib.save(force_round(vert_path), vert_path) if vert_path.exists() else None
        sub_path = str(vert_path).replace("_seg-vert_msk.nii.gz", "_seg-subreg_msk.nii.gz")
        nib.save(force_round(sub_path), sub_path) if Path(sub_path).exists() else None


def majority_vote_help(fake_ct, opt: ROpt):

    curr_file = get_vert_path(fake_ct)
    # Out path
    out_file = opt.get_folder_derivatives() / curr_file.name
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    if not opt.override and out_file.exists():
        print("Skip! exists:", out_file, "                  ", end="\n")
        return
    # load
    segs = [f.get_folder_derivatives() / curr_file.name for f in models_opt]
    segs = [force_round(s).get_fdata() for s in segs if s.exists()]
    # vote
    print("Majority Vote - compute", "                                            ", end="\r")
    segs_stack = np.stack(segs, 0)
    segs_stack[segs_stack == 0] = np.nan
    zeros: np.ndarray = np.zeros_like(segs_stack[[0]])
    segs_stack = np.concatenate([segs_stack, zeros], 0)
    out = mode(segs_stack, axis=0, keepdims=False, nan_policy="omit")[0]
    print("Majority Vote", out_file, "                  ", end="\n")
    # Save
    ref_nib: nib.Nifti1Image = nib.load(fake_ct)
    nib.save(force_round(nib.Nifti1Image(out, ref_nib.affine, header=ref_nib.header)), out_file)  # type: ignore
    nib.save(force_round(nib.Nifti1Image(out, ref_nib.affine, header=ref_nib.header)), opt.get_folder_derivatives() / "backup.nii.gz")  # type: ignore
    # Copy subreg_ctd and on CT
    src_vert = get_vert_path(fake_ct)
    src_cdt = str(src_vert).replace("_seg-vert_msk.nii.gz", "_seg-subreg_ctd.json")
    dst_cdt = Path(opt.get_folder_derivatives(), Path(src_cdt).name)
    dst_ct = Path(opt.get_folder_rawdata(), fake_ct.name)
    opt.get_folder_rawdata().mkdir(exist_ok=True)
    if Path(src_cdt).exists():
        print("[*] Copy from existing cdt", src_cdt)
        Path(dst_cdt).parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(src_cdt, dst_cdt)
        shutil.copyfile(fake_ct, dst_ct)


# CUDA_VISIBLE_DEVICES=0 python3 scripts/segmentation/NAKO2Seg_vote.py --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/110/sub-11081+-30/
if __name__ == "__main__":
    parser = reload_any.get_option_reload()
    opt = reload_any.get_option(parser)
    opt.exp_name = "majority_vote"
    a = "paper_T2_pix2pix_sa-unet"
    # opt.exp_name
    models_opt = [
        ROpt(
            "paper_T2_pix2pix_sa-unet",
            opt.root,
            out=opt.out,
            bs=24,
            cut=True,
            cdt_from=a,
            override=opt.override,
        ),
        ROpt(
            "paper_T2_pix2pix",
            opt.root,
            out=opt.out,
            bs=24,
            cut=True,
            keep_cdt=True,
            override=opt.override,
        ),
        ROpt(
            "paper_T2_diffusion",
            opt.root,
            out=opt.out,
            bs=24,
            ddim=True,
            timesteps=20,
            eta=0,
            cdt_from=a,
            override=opt.override,
        ),
        ROpt(
            "paper_T2_diffusion",
            opt.root,
            out=opt.out,
            bs=24,
            ddim=True,
            timesteps=20,
            eta=1,
            cdt_from=a,
            override=opt.override,
        ),
        # ROpt("paper_T2_pcut_sa-unet", opt.root, out=opt.out, bs=24, cut=True),
        # ROpt("exp_syndiff", opt.root, out=opt.out, bs=24, syndiff=True),
    ]
    majority_vote(models_opt, opt)
