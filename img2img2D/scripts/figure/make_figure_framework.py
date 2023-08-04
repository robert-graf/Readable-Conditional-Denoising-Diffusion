import sys
from pathlib import Path


file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

import nibabel as nib
import numpy as np

from matplotlib.colors import ListedColormap

#### Load network ####
name = "bailiang_256"
version: str = "*"


#### Load dataset ####
# checkpoint = get_latest_Checkpoint(name, log_dir_name="logs_diffusion", best=False, version=version)
# assert checkpoint is not None
# model = Diffusion.load_from_checkpoint(checkpoint, strict=False)
# model.cuda()
# nii_ds_mrt_reg = nii_Dataset("")
# out_save = run_dl(nii_ds_mrt_reg, model, w=2, eta=0, crop=(0, 0, -1, -1))
# nii_ds_mrt_reg.save(out_save, "", revert_to_size=True)
# run_docker("")

from BIDS.wrapper.snapshot_modular import (
    Snapshot_Frame,
    create_snapshot,
)

from BIDS.bids_files import BIDS_FILE, BIDS_Global_info, global_bids_list, sequence_splitting_keys
from BIDS import (
    calc_centroids_from_subreg_vert,
    NII,
    Centroids,
    calc_centroids_labeled_buffered,
    load_centroids,
    to_nii,
)

# fmt: off
colors_itk = (1 / 255) * np.array(
    [
        [255, 0, 0],
        [189, 143, 248],
        [95, 74, 171],
        [165, 114, 253],
        [78, 54, 158],
        [129, 56, 255],
        [56, 5, 149],  # c1-7
        [119, 194, 244],
        [67, 120, 185],
        [117, 176, 217],
        [69, 112, 158],
        [86, 172, 226],
        [48, 80, 140],  # t1-6
        [17, 150, 221],
        [14, 70, 181],
        [29, 123, 199],
        [11, 53, 144],
        [60, 125, 221],
        [16, 29, 126],  # t7-12
        [4, 159, 176],
        [106, 222, 235],
        [3, 126, 140],
        [10, 216, 239],
        [10, 75, 81],
        [108, 152, 158],  # L1-6
        [203, 160, 95],
        [149, 106, 59],
        [43, 95, 199],
        [57, 76, 76],
        [0, 128, 128],
        [188, 143, 143],
        [255, 105, 180],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 239, 213],  # 29-39 unused
        [0, 0, 205    ],
        [255,   0,   0],#<- 40
        [  0, 255,   0],
        [  0,   0, 255],
        [255, 255,   0],
        [  0, 255, 255],
        [255,   0, 255],
        [255, 239, 213],
        [  0,   0, 205],
        [255, 128, 0  ],
        [255, 200, 0  ],  # Label 40-50 (subregions)
        [255, 200, 0  ], 
        [240, 255, 240],
        [245, 222, 179],
        [184, 134, 11],
        [32, 178, 170],
        [255, 20, 147],
        [25, 25, 112],
        [112, 128, 144],
        [34, 139, 34],
        [248, 248, 255],
        [255, 235, 205],
        [255, 228, 196],
        [218, 165, 32],
        [0, 128, 128],  # rest unused
    ]
)
# fmt: on
cm_itk = ListedColormap(colors_itk)  # type: ignore
cm_itk.set_bad(color="w", alpha=0)  # set NaN to full opacity for overlay

if Path("/media/data/robert/datasets/spinegan_T2w/raw/").exists():
    dataset_path = "/media/data/robert/datasets/spinegan_T2w/raw/"
    sequence_splitting_keys.append("reg")
    global_info = BIDS_Global_info(
        [dataset_path],
        [
            "sourcedata",
            "rawdata",
            "rawdata_ct",
            "rawdata_dixon",
            "derivatives",
            "derivatives_msk",
            "derivatives_spinalcord",
            "registration",
        ],
        additional_key=["sequ", "seg", "ovl", "e", "reg"],
    )

    ct_file_org: BIDS_FILE = global_bids_list["sub-spinegan0038_ses-20211009_sequ-11_ct"]
    ct_file_org: BIDS_FILE = global_bids_list["sub-spinegan0038_ses-20211009_sequ-11_ct"]

    def key_transform(f: BIDS_FILE):
        if f.format == "ctd" and "nii.gz" not in f.file:
            return "others"
        else:
            return None

    ##CT
    ct_family = ct_file_org.get_sequence_files(key_transform)
    ct_nii = NII(ct_file_org.open_nii())
    ct = ct_nii.get_array()[:, 50:350, 190:280]
    print(ct_nii.orientation)
    subreg = NII(ct_family["subreg"][0].open_nii(), True).get_seg_array()[:, 50:350, 190:280]
    vert = NII(ct_family["vert"][0].open_nii(), True).get_seg_array()[:, 50:350, 190:280]
    ct = nib.Nifti1Image(ct, ct_nii.affine)
    subreg = NII(nib.Nifti1Image(subreg, ct_nii.affine), True)
    vert = NII(nib.Nifti1Image(vert, ct_nii.affine), True)
    cdt = None
    cdt = calc_centroids_from_subreg_vert(vert, subreg, subreg_id=[50, 42], fixed_offset=100)
    del cdt.centroids[4219]
    # ctd = [i for i in ctd if i[0] != 15 and i[0] != 115 and i[0] != 114]
    ct_frame = Snapshot_Frame(image=ct, segmentation=None, centroids=cdt, mode="CT", cmap=cm_itk)
    ##MRT
    mr_file_org: BIDS_FILE = global_bids_list["sub-spinegan0038_ses-20211009_sequ-803_e-3_dixon"]
    mr_family = mr_file_org.get_sequence_files(key_transform)
    # spine = mr_family["spinalcord"][0].open_nii()
    print(mr_family.keys())
    mr_file_org: BIDS_FILE = global_bids_list["sub-spinegan0038_ses-20211009_sequ-803_e-3_dixon"]
    cdt = calc_centroids_labeled_buffered(mr_family["msk"], subreg_id=50)
    cdt2 = calc_centroids_labeled_buffered(mr_family["procspin"], subreg_id=42)
    print(cdt2)
    for k, v in cdt.items():
        cdt[k] = (v[0], v[1], 7.0)  # + 5
    for k, v in cdt2.items():
        cdt[k + 4200] = (v[0], v[1], 7.0)  # + 5
    print(cdt)
    del cdt.centroids[26]
    mr_frame = Snapshot_Frame(image=mr_file_org, segmentation=None, centroids=cdt, cmap=cm_itk)

    ##CT REG
    ctr_file_org: BIDS_FILE = global_bids_list["sub-spinegan0038_ses-20211009_sequ-11_reg-803_ct"]
    mrr_file_org: BIDS_FILE = global_bids_list["sub-spinegan0038_ses-20211009_sequ-803_reg-11_acq-real_dixon"]

    ctr_family = ctr_file_org.get_sequence_files(key_transform)
    # mrr_family = mrr_file_org.get_sequence_files(key_transform)

    cdt = calc_centroids_from_subreg_vert(
        ctr_family["vert"][0], ctr_family["subreg"][0], subreg_id=[50, 42], fixed_offset=100
    )
    print(cdt)
    del cdt.centroids[4219]
    # ctd = [i for i in ctd if i[0] != 15 and i[0] != 115 and i[0] != 114]
    # ctr_family["subreg"][0]
    ct_reg_frame = Snapshot_Frame(image=ctr_file_org, segmentation=None, centroids=cdt, mode="CT", cmap=cm_itk)
    mr_reg_frame = Snapshot_Frame(image=mrr_file_org, segmentation=None, centroids=cdt, mode="MRI", cmap=cm_itk)
    ct_transl: BIDS_FILE = global_bids_list["sub-spinegan0038_ses-20211009_sequ-803_e-3_ct"]
    ct_transl_s: BIDS_FILE = global_bids_list["sub-spinegan0038_ses-20211009_sequ-803_e-3_seg-subreg_msk"]
    ct_transl_v: BIDS_FILE = global_bids_list["sub-spinegan0038_ses-20211009_sequ-803_e-3_seg-vert_msk"]
    cdt = calc_centroids_from_subreg_vert(ct_transl_v, ct_transl_s, subreg_id=[50, 42], fixed_offset=100)
    ct_tr_frame = Snapshot_Frame(image=ct_transl, segmentation=None, centroids=cdt, mode="CT", cmap=cm_itk)
    x = to_nii(ct_transl)
    ct_tr_frame2 = Snapshot_Frame(image=ct_transl, segmentation=ct_transl_s, centroids=None, mode="MRI", cmap=cm_itk)
    create_snapshot(
        snp_path=Path("snapshot_test.jpg"),
        frames=[ct_frame, mr_frame, ct_reg_frame, mr_reg_frame, ct_tr_frame, ct_tr_frame2],
    )
else:
    input_frame = Snapshot_Frame(
        image="/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/Case10.nii.gz",
        segmentation="/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/same_Case10_seg-subreg_msk.nii.gz",
        centroids=None,
        mode="MRI",
        alpha=0.0,
        crop_img=True,
    )
    ct_frame = Snapshot_Frame(
        image="/media/data/robert/datasets/MRSpineSeg_Challenge/result/rawdata/same_Case10_ct.nii.gz",
        segmentation="/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/same_Case10_seg-subreg_msk.nii.gz",
        centroids=None,
        mode="CT",
        alpha=0.0,
        crop_img=True,
        gauss_filter=True,
    )
    output_frame = Snapshot_Frame(
        image="/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/Case10.nii.gz",
        segmentation="/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/same_Case10_seg-subreg_msk.nii.gz",
        centroids=(
            "/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/same_Case10_seg-vert_msk.nii.gz",
            "/media/data/robert/datasets/MRSpineSeg_Challenge/result/derivatives/same_Case10_seg-subreg_msk.nii.gz",
            [50, 42],
        ),
        mode="MRI",
        cmap=cm_itk,
        ignore_cdt_for_centering=True,
        crop_img=True,
    )
    create_snapshot(
        snp_path=Path("/media/data/robert/datasets/MRSpineSeg_Challenge/result/snapshot_test2.jpg"),
        frames=[input_frame, ct_frame, output_frame],
    )
