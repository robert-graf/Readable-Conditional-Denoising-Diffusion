from math import ceil
from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))

import torch
from dataloader.dataloader_nii_seg import nii_Dataset, nii_Datasets
import reload_any
from reload_any import Reload_Any_Option as ROpt, TranslationType, ResamplingType
import nibabel as nib
import numpy as np

from BIDS import BIDS_FILE, NII
from scripts.segmentation.seg_functions import translate


def get_nii(nii: BIDS_FILE, resample_type: ResamplingType = ResamplingType.NATIVE) -> NII:
    if resample_type == ResamplingType.ISO:
        return nii.open_nii_reorient(("R", "I", "P")).rescale_((1, 1, 1))
    elif resample_type == ResamplingType.SAGITTAL:
        return nii.open_nii_reorient(("R", "I", "P")).rescale_((1, -1, -1))
    elif resample_type == ResamplingType.NATIVE:
        return nii.open_nii_reorient(("R", "I", "P"))
    else:
        raise NotImplementedError(resample_type)


def get_dataset(nii: BIDS_FILE, resample_type: ResamplingType = ResamplingType.NATIVE) -> nii_Dataset:
    if resample_type == ResamplingType.ISO:
        return nii_Dataset(nii.file["nii.gz"], orientation=("R", "I", "P"), keep_scale=(), min_value=0)
    elif resample_type == ResamplingType.SAGITTAL:
        return nii_Dataset(nii.file["nii.gz"], orientation=("R", "I", "P"), keep_scale=("R"), min_value=0)
    elif resample_type == ResamplingType.NATIVE:
        return nii_Dataset(nii.file["nii.gz"], orientation=("R", "I", "P"), keep_scale=("R", "I", "P"), min_value=0)
    else:
        raise NotImplementedError(f"NotImplementedError - {resample_type}")


buffer = {}


def run_model_one_file(
    opt: ROpt,
    in_file: BIDS_FILE,
    path: Path,
    trans_type: TranslationType = TranslationType.V_STITCHING,
    resample_type: ResamplingType = ResamplingType.SAGITTAL,
):
    if opt.exp_name in buffer:
        model = buffer[opt.exp_name]
    else:
        model, checkpoint = reload_any.get_model(opt)
        buffer.clear()
        buffer[opt.exp_name] = model
    ####### Loop
    with torch.no_grad():
        nii_ds = get_dataset(in_file, resample_type)
        out_save = translate(nii_ds, model, opt, trans_type)
        if out_save is None:
            return
        assert out_save.min() >= -1000, f"[{out_save.min()} - {out_save.max()}] out >= -1000"
        assert out_save.max() <= 1000, f"[{out_save.min()} - {out_save.max()}] out <= 1000"
        nii_ds.save(out_save, path, revert_to_size=not opt.keep_resampled, min_value=-1000)  #
