from __future__ import annotations
from enum import Enum, auto
from pathlib import Path
import sys


file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))


import nibabel as nib
import numpy as np
import reload_any
import torch
from dataloader.dataloader_nii_seg import nii_Dataset, nii_Datasets
from scripts.segmentation.seg_functions import run_model, compute_dice_on_nii
from scripts.segmentation.NAKO2Seg_vote import segment_with_network, get_vert_path, make_snapshot
from typing import Callable
from scripts.segmentation.NAKO2Seg_run_all import get_all_models


class DatasetEnum(Enum):
    T1w = auto()
    T2w = auto()
    Chl = auto()
    Chl_our = auto()
    All = auto()


def majority_vote(file_seg_fake: list[Path], file_mr_names: list[Path], result_folder: Path):
    voter_list: list[str] = [
        "result_exp_syndiff",
        "result_paper_T1_diffusion_ddim_eta-0.0_20",
        "result_paper_T1_diffusion_ddim_eta-0.0_50",
        "result_paper_T1_diffusion_ddim_eta-1.0_20",
        "result_paper_T1_diffusion_ddim_eta-1.0_50",
        "result_paper_T1_pcut_sa-unet",
        "result_paper_T1_pix2pix",
        "result_paper_T1_pix2pix_sa-unet",
    ]
    for curr_file in file_seg_fake:
        print("Majority Vote", end="\r")
        # /media/data/robert/test/T1result_{}_majority_vote/derivatives/sub-fxclass0017_sequ-1_reg-2_seg-vert_msk.nii.gz
        # result_folder
        # torch.argmax()
        import nibabel as nib

        segs = [result_folder.parent / f / "derivatives" / curr_file.name for f in voter_list]
        segs = [nib.load(str(s)).get_fdata() for s in segs if s.exists()]
        from scipy.stats import mode

        out = mode(np.stack(segs, 0), axis=0, keepdims=False)[0]
        ref_nib: nib.Nifti1Image = nib.load(file_mr_names[0])

        # print(out.shape, ref_nib.shape)
        out_file = result_folder / "derivatives" / curr_file.name
        Path(out_file).parent.mkdir(exist_ok=True, parents=True)
        print("Majority Vote", out_file, "                  ", end="\r")
        nib.save(nib.Nifti1Image(out, ref_nib.affine, header=ref_nib.header), out_file)  # type: ignore


def run_dice(
    opt: reload_any.Reload_Any_Option,
    mr2seg_gt_file: Callable[[Path], Path],
    matching_string="*/*/*T1c.nii*",
    snap=False,
    mapping={},
):
    if not opt.majority_vote:
        file_ct_translated, file_mr, rawdata_folder = segment_with_network(opt, snap=snap, match_string=matching_string)
    else:
        dataset = nii_Datasets(root=opt.root, match_string=matching_string)
        file_ct_translated, file_mr, rawdata_folder = run_model(opt, dataset, no_translation=True)

    file_seg_fake = [get_vert_path(i) for i in file_ct_translated]
    file_seg_gt = [mr2seg_gt_file(i) for i in file_mr]

    print("compute dice")
    compute_dice_on_nii(file_seg_fake, file_seg_gt, rawdata_folder, n_jobs=1, mapping=mapping)


mapping_sg_chl = {
    1: 0,
    2: 24,
    3: 23,
    4: 22,
    5: 21,
    6: 20,
    7: 19,
    8: 18,
    9: 17,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
}
forbidden_keys = [
    "cut",
    "ddim",
    "timesteps",
    "eta",
    "root",
    "bs",
    "exp_name",
    "config",
    "syndiff",
    "ddpm",
    "T1",
    "ds",
    "adversarial",
    "guidance_w",
]


def run_dice_on_DS(opt: reload_any.Reload_Any_Option):
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
                    print(e)

    elif ds == DatasetEnum.T1w:
        opt.root = "/media/data/robert/datasets/fx_T1w/test_nii/"
        opt.out = "/media/data/robert/test/T1w"
        mr2seg: Callable[[Path], Path] = lambda file_mr: Path(file_mr.parent, "seg_jos_corr.nii.gz")

        run_dice(opt, mr2seg, "*/*/*T1c.nii*")
    elif ds == DatasetEnum.T2w:
        opt.root = "/media/data/robert/datasets/spinegan_T2w/test_nii/"
        opt.out = "/media/data/robert/test/T2w"
        mr2seg: Callable[[Path], Path] = lambda file_mr: Path(file_mr.parent, "seg.nii.gz")
        # mr2seg: Callable[[Path], Path] = lambda file_mr: list(file_mr.parent.glob("*_seg-vert_msk.nii.gz"))[0]
        run_dice(opt, mr2seg, "*/*/*acq-real_dixon.nii*")

    elif ds == DatasetEnum.Chl:
        opt.root = "/media/data/robert/datasets/MRSpineSeg_Challenge/train_split1/MR/"
        opt.out = "/media/data/robert/test/MRSpineSeg_Challenge_split1"
        mr2seg: Callable[[Path], Path] = lambda file_mr: Path(
            file_mr.parent.parent, "Mask", f"mask_case{file_mr.name.replace('Case','')}"
        )
        run_dice(opt, mr2seg, "Case*.nii.gz", mapping=mapping_sg_chl)
    elif ds == DatasetEnum.Chl_our:
        opt.root = "/media/data/robert/datasets/MRSpineSeg_Challenge/train_ours/MR/"
        opt.out = "/media/data/robert/test/MRSpineSeg_Challenge_ours"
        opt.translationType = reload_any.TranslationType.TOP
        mr2seg: Callable[[Path], Path] = lambda file_mr: Path(
            file_mr.parent.parent, "Mask", f"mask_case{file_mr.name.replace('Case','')}"
        )
        run_dice(opt, mr2seg, "Case*.nii.gz", mapping=mapping_sg_chl)
    else:
        raise NotImplementedError()


def run_all_dice_on_DS(opt: reload_any.Reload_Any_Option):
    ds: DatasetEnum = DatasetEnum[opt.ds]  # type: ignore
    if ds == DatasetEnum.All:
        for a in DatasetEnum:
            if a == DatasetEnum.All:
                continue
            else:
                opt.ds = a.name  # type: ignore
                run_all_dice_on_DS(opt)
    else:

        opts = {k: v for k, v in opt.__dict__.items() if k not in forbidden_keys}

        models = get_all_models(root="", T1=ds == DatasetEnum.T1w, opts=opts)

        for m in models:
            m.__dict__["ds"] = ds.name
            try:
                run_dice_on_DS(m)
            except Exception as e:
                print(e)


###### Model

#######################################
# python3 scripts/segmentation/SegTestCases.py  -en All -ds All --translationType TOP
# python3 scripts/segmentation/SegTestCases.py  -en All -ds T2w --translationType TOP
# python3 scripts/segmentation/SegTestCases.py  --syndiff -en exp_syndiff_t2w -ds All --translationType TOP
# python3 scripts/segmentation/SegTestCases.py  --syndiff -en exp_syndiff_t2w_1p -ds All --translationType TOP
#######################################
# CUDA_VISIBLE_DEVICES=1 python3 scripts/segmentation/NAKO2Seg.py --ddim --timesteps 20 --eta 0 -en bailiang_256 --out /media/data/NAKO/MRT/test/ --root /media/data/robert/datasets/fx_T1w/test_nii/

# python3 scripts/segmentation/SegTestCases.py --cut -en paper_T1_cut --All
# python3 scripts/segmentation/SegTestCases.py --cut -en paper_T1_cut_sa-unet      --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
# python3 scripts/segmentation/SegTestCases.py --cut -en paper_T1_pcut_sa-unet     --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
# python3 scripts/segmentation/SegTestCases.py --cut -en paper_T1_pix2pix_sa-unet  --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
# python3 scripts/segmentation/SegTestCases.py --cut -en paper_T1_pix2pix          -ds All
# python3 scripts/segmentation/SegTestCases.py --ddim --timesteps 20 --eta 0 -en paper_T1_diffusion        --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
# python3 scripts/segmentation/SegTestCases.py --ddim --timesteps 20 --eta 1 -en paper_T1_diffusion        --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
# python3 scripts/segmentation/SegTestCases.py --ddim --timesteps 50 --eta 0 -en paper_T1_diffusion        --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
# python3 scripts/segmentation/SegTestCases.py --ddim --timesteps 50 --eta 1 -en paper_T1_diffusion        --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
# python3 scripts/segmentation/SegTestCases.py --ddpm -en paper_T1_diffusion        --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
# python3 scripts/segmentation/SegTestCases.py --syndiff -en exp_syndiff --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
# python3 scripts/segmentation/SegTestCases.py --majority_vote -en paper_T1_pix2pix --out /media/data/robert/test/T1 --root /media/data/robert/datasets/fx_T1w/test_nii/
if __name__ == "__main__":
    parser = reload_any.get_option_reload()
    parser.add_argument("-ds", default="All", choices=reload_any.translation_type(enum=DatasetEnum))
    opt = reload_any.get_option(parser)
    if opt.exp_name == "All":
        run_all_dice_on_DS(opt)
    else:
        run_dice_on_DS(opt)
