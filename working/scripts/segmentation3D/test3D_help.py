from enum import Enum, auto
import glob
from pathlib import Path
import sys

root = str(Path().parent.parent)
sys.path.append(root)
from typing import Callable
import factory.arguments as arguments
from scripts.segmentation.SegTestCases import mapping_sg_chl
from scripts.segmentation.MRISeg_helper import get_dataset, get_subreg_path, get_vert_path, segment_and_post_processing
from scripts.segmentation.seg_functions import compute_dice_on_nii
from reload_any import ResamplingType


class DatasetEnum(Enum):
    T1w = auto()
    T2w = auto()
    Chl = auto()
    Chl_our = auto()
    All = auto()


def run_dice_on_DS(opt: arguments.Train_Option):
    if not hasattr(opt, "ds"):
        opt.ds = "All"
    ds: DatasetEnum = DatasetEnum[opt.ds]  # type: ignore

    if ds == DatasetEnum.All:
        for a in DatasetEnum:
            if a == DatasetEnum.All:
                continue
            else:
                opt.ds = a.name  # type: ignore
                try:
                    run_dice_on_DS(opt)
                except Exception as e:
                    raise e
                    print(e)

    elif ds == DatasetEnum.T1w:
        opt.root = "/media/data/robert/datasets/fx_T1w/test_nii/"
        opt.out = "/media/data/robert/test/T1w"
        mr2seg: Callable[[Path], Path] = lambda file_mr: Path(file_mr.parent, "seg_jos_corr.nii.gz")
        gt2subreg: Callable[[Path], Path] = lambda gt_seg: next(gt_seg.parent.glob("*_seg-subreg_msk.nii.gz"))
        run_dice_on_files_3d(opt, mr2seg, "*/*/*T1c.nii*", target2subreg=get_subreg_path, gt2subreg=gt2subreg)
    elif ds == DatasetEnum.T2w:
        opt.root = "/media/data/robert/datasets/spinegan_T2w_all_reg/test/"
        opt.out = "/media/data/robert/test/T2w_2"
        mr2seg: Callable[[Path], Path] = lambda file_mr: next(file_mr.parent.glob("*_seg-vert_msk.nii.gz"))
        gt2subreg: Callable[[Path], Path] = lambda gt_seg: next(gt_seg.parent.glob("*_seg-subreg_msk.nii.gz"))
        # mr2seg: Callable[[Path], Path] = lambda file_mr: list(file_mr.parent.glob("*_seg-vert_msk.nii.gz"))[0]

        run_dice_on_files_3d(opt, mr2seg, "*/*/*_dixon.nii*", target2subreg=get_subreg_path, gt2subreg=gt2subreg)

        # opt.root = "/media/data/robert/datasets/spinegan_T2w/test_nii/"
        # opt.out = "/media/data/robert/test/T2w"
        # mr2seg: Callable[[Path], Path] = lambda file_mr: Path(file_mr.parent, "seg.nii.gz")
        # gt2subreg: Callable[[Path], Path] = lambda gt_seg: next(gt_seg.parent.glob("*_seg-subreg_msk.nii.gz"))
        ## mr2seg: Callable[[Path], Path] = lambda file_mr: list(file_mr.parent.glob("*_seg-vert_msk.nii.gz"))[0]
        # run_dice_on_files_3d(opt, mr2seg, "*/*/*acq-real_dixon.nii*", target2subreg=get_subreg_path, gt2subreg=gt2subreg)

    elif ds == DatasetEnum.Chl:
        root_gt = "/media/data/robert/datasets/MRSpineSeg_Challenge/train_split1/MR/"
        mr2seg: Callable[[Path], Path] = lambda file_mr: Path(file_mr.parent.parent, "Mask", f"mask_case{file_mr.name.replace('Case','')}")
        out = "/media/data/robert/test/MRSpineSeg_Challenge_split1"
        opt.out = out
        opt.root = root_gt
        folder = get_result_folder_name(opt)
        rawdata_folder = Path(root_gt, folder)
        rawdata_folder.mkdir(exist_ok=True)
        rawdata_folder = Path(out, folder, "rawdata/")

        file_mr = [Path(p) for p in list(sorted(glob.glob(str(root_gt) + "/" + "Case*.nii.gz")))]
        # file_seg_fake = [get_vert_path(next(i.glob(f'*{str(file_mr[0]).replace(".nii.gz","")}*'))) for i in file_mr]
        file_seg_gt = [mr2seg(i) for i in file_mr]
        file_seg_fake = []
        for f in file_mr:
            name = f.name.replace(".nii.gz", "")
            file_seg_fake.append(Path(out, folder, "derivatives", f"sub-{name}_mod-T2w_seg-vert_msk.nii.gz"))
        run_dice_on_files_3d_version2(opt, file_mr, file_seg_fake, file_seg_gt, "Case*.nii.gz", mapping=mapping_sg_chl)
    elif ds == DatasetEnum.Chl_our:
        root_gt = "/media/data/robert/datasets/MRSpineSeg_Challenge/train_ours/MR/"
        mr2seg: Callable[[Path], Path] = lambda file_mr: Path(file_mr.parent.parent, "Mask", f"mask_case{file_mr.name.replace('Case','')}")
        mr2sub: Callable[[Path], Path] = lambda file_mr: Path(
            file_mr.parent.parent, "derivatives", f"{file_mr.name.replace('.nii.gz','_seg-subreg_msk.nii.gz')}"
        )

        out = "/media/data/robert/test/MRSpineSeg_Challenge_ours"
        opt.out = out
        opt.root = root_gt
        folder = get_result_folder_name(opt)
        rawdata_folder = Path(root_gt, folder)
        rawdata_folder.mkdir(exist_ok=True)
        rawdata_folder = Path(out, folder, "rawdata/")

        file_mr = [Path(p) for p in list(sorted(glob.glob(str(root_gt) + "/" + "Case*.nii.gz")))]
        # file_seg_fake = [get_vert_path(next(i.glob(f'*{str(file_mr[0]).replace(".nii.gz","")}*'))) for i in file_mr]
        file_seg_gt = [mr2seg(i) for i in file_mr]
        file_sub_gt = [mr2sub(i) for i in file_mr]
        file_seg_fake = []
        for f in file_mr:
            name = f.name.replace(".nii.gz", "")
            file_seg_fake.append(Path(out, folder, "derivatives", f"sub-{name}_mod-T2w_seg-vert_msk.nii.gz"))
        run_dice_on_files_3d_version2(
            opt, file_mr, file_seg_fake, file_seg_gt, "Case*.nii.gz", mapping=mapping_sg_chl, file_sub_gt=file_sub_gt
        )

        # opt.root = "/media/data/robert/datasets/MRSpineSeg_Challenge/train_ours/MR/"
        # opt.out = "/media/data/robert/test/MRSpineSeg_Challenge_ours"
        ## opt.translationType = reload_any.TranslationType.TOP
        # mr2seg: Callable[[Path], Path] = lambda file_mr: Path(file_mr.parent.parent, "Mask", f"mask_case{file_mr.name.replace('Case','')}")
        #
        # gt2subreg: Callable[[Path], Path] = lambda gt_seg: Path(
        #    gt_seg.parent.parent,
        #    "derivatives",
        #    gt_seg.name.replace("mask_c", "C").replace(".nii.gz", "_seg-subreg_msk.nii.gz"),
        # )
        # run_dice_on_files_3d(opt, mr2seg, "*Case*.nii.gz", mapping=mapping_sg_chl, target2subreg=get_subreg_path, gt2subreg=gt2subreg)
    else:
        raise NotImplementedError()


# /media/data/old_NAKO/MRT/test/paper/


def get_vert_path(ct_file):
    file_seg_ct = str(ct_file).replace("rawdata", "derivatives").rsplit("_", maxsplit=1)[0] + "_seg-vert_msk.nii.gz"
    p = Path(file_seg_ct)
    # if p.exists():
    return p
    # return next(Path(p.parent).glob(ct_file.name.rsplit("_", maxsplit=1)[0] + "*_seg-vert_msk.nii.gz"))


def run_dice_on_files_3d_version2(
    opt: arguments.Train_Option, file_mr, file_seg_fake, file_seg_gt, snap=False, matching_string="", mapping={}, file_sub_gt=None
):
    file_ct_translated, rawdata_folder = segment_with_network(opt, file_mr, snap=snap, match_string=matching_string)
    print(file_mr[0])
    print(file_ct_translated[0])
    print(file_seg_gt[0])
    print(file_seg_fake[0])
    print("compute dice")
    compute_dice_on_nii(
        file_seg_fake,
        file_seg_gt,
        rawdata_folder,
        mapping=mapping,
        filename="dice_score3D",
    )
    if file_sub_gt is not None:
        sub_reg = [get_subreg_path(x) for x in file_seg_fake]
        assert sub_reg[0].exists(), sub_reg[0]

        assert file_sub_gt[0].exists(), file_sub_gt[0]
        compute_dice_on_nii(
            file_seg_fake,
            file_seg_gt,
            rawdata_folder,
            mapping=mapping,
            filename="dice_score3D_subreg",
            sub_reg_gt=file_sub_gt,
            sub_reg=sub_reg,
            remove_subreg_id=[50, 49],
        )


def run_dice_on_files_3d(
    opt: arguments.Train_Option,
    mr2seg_gt_file: Callable[[Path], Path],
    matching_string="*/*/*T1c.nii*",
    snap=False,
    mapping={},
    target2subreg: Callable[[Path], Path] | None = None,
    gt2subreg: Callable[[Path], Path] = lambda x: x,
):
    opt.resample = ResamplingType.ISO
    out = opt.out
    folder = get_result_folder_name(opt)
    rawdata_folder = Path(out, folder)
    rawdata_folder.mkdir(exist_ok=True)
    rawdata_folder = Path(out, folder, "rawdata/")

    file_mr = [Path(p) for p in list(sorted(glob.glob(str(rawdata_folder) + "/" + matching_string)))]
    if len(file_mr) == 0:
        file_mr = [Path(p) for p in list(sorted(glob.glob(str(opt.root) + "/" + matching_string)))]

    file_seg_fake = [get_vert_path(i) for i in file_mr]
    file_seg_gt = [mr2seg_gt_file(i) for i in file_mr]
    # T1w
    for fk, gt in zip(file_seg_fake, file_mr):
        out = str(fk).replace("_seg-vert_msk.nii.gz", "_seg-subreg_ctd.json")
        inp = str(gt).replace("_T1c", "_seg-subreg_ctd.json")
        if opt.ds == DatasetEnum.T1w and not Path(out).exists():
            print("test")
            from BIDS import calc_centroids, Centroids

            Centroids.load(inp).save(out, verbose=True)

    # T2w-2 transferer centroids
    for fk, gt in zip(file_seg_fake, file_mr):
        out = str(fk).replace("_seg-vert_msk.nii.gz", "_seg-subreg_ctd.json")
        if opt.ds == DatasetEnum.T2w and Path(str(gt).replace("dixon", "msk")).exists() and not Path(out).exists():
            print("test")
            from BIDS import calc_centroids

            calc_centroids(str(gt).replace("dixon", "msk")).save(out, verbose=True)

    file_ct_translated, rawdata_folder = segment_with_network(opt, file_mr, snap=snap, match_string=matching_string)
    print(file_mr[0])
    print(file_ct_translated[0])

    # file_seg_fake_subreg = [get_subreg_path(i) for i in file_ct_translated]
    # file_seg_fake_cdt = [str(get_vert_path(i)).replace("_seg-vert_msk.nii.gz", "_seg-subreg_ctd.json") for i in file_ct_translated]
    # from BIDS import calc_centroids_from_subreg_vert

    # [
    #    calc_centroids_from_subreg_vert(vert, subreg, decimals=2).save(cdt)
    #    for vert, subreg, cdt in zip(file_seg_fake, file_seg_fake_subreg, file_seg_fake_cdt)
    # ]

    print(file_seg_gt[0])
    print(file_seg_fake[0])

    # /media/data/robert/datasets/MRSpineSeg_Challenge/train_ours/Mask/mask_case106.nii.gz
    # /media/data/robert/test/MRSpineSeg_Challenge_ours/result_diffusion3D_T2_CT_w=0/derivatives/Case106_seg-vert_msk.nii.gz
    print("compute dice")
    # compute_dice_on_nii(file_seg_fake, file_seg_gt, rawdata_folder, n_jobs=1, mapping=mapping)
    for idx, f in enumerate(file_seg_fake):
        if not f.exists():
            file_seg_fake[idx] = next(Path(f.parent).glob(str(f.name).split("_seg-vert")[0] + "*_seg-vert_msk.nii.gz"))

    compute_dice_on_nii(
        file_seg_fake,
        file_seg_gt,
        rawdata_folder,
        mapping=mapping,
        filename="dice_score3D",
        # sub_reg_gt=sub_reg_gt,
        # sub_reg=sub_reg,
    )
    if target2subreg is not None:
        from BIDS.docker import run_docker_subreg

        sub_reg = [Path(x.parent, x.name.replace("seg-vert_msk.nii.gz", "seg-subreg_msk.nii.gz")) for x in file_seg_fake]
        sub_reg_gt = [Path(x.parent, x.name.replace("seg-vert_msk.nii.gz", "seg-subreg_msk.nii.gz")) for x in file_seg_gt]

        for f, gt in zip(sub_reg_gt, file_seg_gt):
            if not f.exists():
                run_docker_subreg(gt)
        # sub_reg_gt = [gt2subreg(x) for x in file_seg_gt]
        assert sub_reg_gt[0].exists(), sub_reg_gt[0]
        assert sub_reg[0].exists(), sub_reg[0]
        # print(sub_reg[0])
        # print(sub_reg_gt[0])

        compute_dice_on_nii(
            file_seg_fake,
            file_seg_gt,
            rawdata_folder,
            mapping=mapping,
            filename="dice_score3D_subreg",
            sub_reg_gt=sub_reg_gt,
            sub_reg=sub_reg,
            remove_subreg_id=[50, 49],
        )


def get_result_folder_name(opt) -> str:
    result_folder = "result_" + opt.exp_name
    if opt.ddim or opt.adversarial:
        result_folder += f"_ddim_eta-{opt.eta:.1f}_{opt.timesteps}"
    elif opt.ddpm:
        result_folder += "_ddpm"
    if opt.guidance_w != 1:
        result_folder += f"_w={int(opt.guidance_w)}"
    return result_folder


import shutil, errno


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        # if exc.errno in (errno.ENOTDIR, errno.EINVAL):
        #    shutil.copy(src, dst)
        # else:
        raise exc


from scripts.segmentation3D import run3D


def segment_with_network(
    opt,
    file_mr_names,
    snap=False,
    match_string="*-1_t2.nii*",
    no_translation=False,
    # fix_rotation_nako=False,
    **args,
) -> tuple[list[Path], list[Path], Path]:
    dataset = get_dataset(opt, match_string)
    out = opt.out
    folder = get_result_folder_name(opt)
    rawdata_folder = Path(out, folder)
    rawdata_folder.mkdir(exist_ok=True)
    rawdata_folder = Path(out, folder, "rawdata/")
    print("\nrawdata:", rawdata_folder)

    # copy MRIs
    if not rawdata_folder.exists():
        copyanything(opt.root, rawdata_folder)
    # make lists

    root = str(opt.root).replace("+", "*")
    for f in rawdata_folder.iterdir():
        if "Case" in f.name and "T2w" not in f.name:
            f.rename(Path(f.parent, "sub-" + f.name.split(".")[0] + "_T2w.nii.gz"))

    file_ct_translated = [
        Path(f).parent / (Path(f).stem.replace(".nii", "").rsplit("_", maxsplit=1)[0] + "_ct.nii.gz") for f in file_mr_names
    ]
    if len(file_ct_translated) != 0 and "sub-" not in file_ct_translated[0].name:
        file_ct_translated = [Path(f.parent, "sub-" + f.name.split("_ct.nii")[0] + "_mod-T2w_ct.nii.gz") for f in file_ct_translated]
    if len(file_ct_translated) != 0 and "e-" in file_ct_translated[0].name:
        file_ct_translated = [Path(str(f).replace("_e-1", "").replace("_e-2", "").replace("_e-3", "")) for f in file_ct_translated]

    # call run model
    run3D.mri2ct_on_ds(
        ds=rawdata_folder.parent,
        opt=opt,
        formats=["T2w", "dixon", "mr", "T1c"],
        save_iso_mr=True,
        steps=opt.timesteps,
        eta=opt.eta,
        w=opt.guidance_w,
        roll_ct=2,
        override_gpu=False,
        override=False,
        use_existing_ctd=True,
        snapshot=True,
        max_parallel=16,
        gpu=True,
        force_rerun_docker=False,
        only_existing=False,
    )
    if not rawdata_folder.exists():
        rawdata_folder.mkdir(exist_ok=True, parents=True)
        exit(rawdata_folder)
    # if no_translation:
    return file_ct_translated, rawdata_folder
    # else:
    #    return segment_and_post_processing(file_ct_translated, file_mr_names, rawdata_folder, snap, opt, True)
