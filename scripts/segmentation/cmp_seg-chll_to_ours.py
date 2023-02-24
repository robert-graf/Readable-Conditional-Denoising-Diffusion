from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from scripts.segmentation.seg_functions import compute_dice_on_nii

org_path = Path("/media/data/robert/datasets/MRSpineSeg_Challenge/train_ours/Mask_org/")
our_path = Path("/media/data/robert/datasets/MRSpineSeg_Challenge/train_ours/Mask/")
file_seg_fake = list(org_path.iterdir())
file_seg_gt = list(our_path.iterdir())

print("compute dice")
compute_dice_on_nii(
    file_seg_fake,
    file_seg_gt,
    Path("/media/data/robert/datasets/MRSpineSeg_Challenge/train"),
    n_jobs=1,
    mapping={11: 0, 12: 0, 13: 0, 1: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0},
    mapping_pred={11: 0, 12: 0, 13: 0, 1: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0},
)
