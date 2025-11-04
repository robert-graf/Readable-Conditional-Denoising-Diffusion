from pathlib import Path

import torch
import yaml
from torch import Tensor
from TPTBox import NII, Print_Logger, to_nii
from TPTBox.core.bids_files import BIDS_FILE, Buffered_BIDS_Global_info

from pl_models.diffusion.diffusion import Diffusion
from utils import arguments
from utils.preprocessing import padded_shape, run_model

root = Path(__file__).parent
log = Print_Logger()
log.on_log("Load Model")
model_path = root / "logs_diffusion3D/Diffusion_3D_T2w_CT_spine_sag/version_0/T2_to_CT_iso_diffusion_img.pt"

dataset = "/DATA/NAS/datasets_processed/NAKO/dataset-nako"
with open(str(model_path).replace(".pt", ".yaml")) as file:
    opt = arguments.Diffusion_Option()
    opt.__dict__ = yaml.full_load(file)
    opt.conditional_label_size = 0
    opt.conditional_dimensions = 4
    in_channel = opt.in_channel  # type: ignore
diffusion = Diffusion(opt, channel=in_channel)
dic = torch.load(model_path)
diffusion.load_state_dict(dic, strict=True)


def make_crop_divisible_by_8(crop, shape):
    """
    Adjusts crop slices so that the resulting crop size is divisible by 8.
    Expands the crop (never shrinks) and clips to the image shape.

    Parameters
    ----------
    crop : list[slice]
        Original crop [z_slice, y_slice, x_slice]
    shape : tuple[int]
        Shape of the original image (z, y, x)
    """
    new_crop = []
    for sl, dim in zip(crop, shape, strict=False):
        start, stop = sl.start, sl.stop
        size = stop - start
        remainder = size % 16
        if remainder != 0:
            expand = 16 - remainder
            stop = min(stop + expand, dim)  # increase end but stay within bounds
        new_crop.append(slice(start, stop))
    return new_crop


def prepare_nii(mri_path, ref_nii: NII):
    nii = NII.load(mri_path, False, 0)

    nii_iso = nii.rescale((1, 1, 1), verbose=True).reorient(("R", "I", "P"))
    ref_nii = ref_nii.resample_from_to(nii_iso)
    crop = ref_nii.compute_crop(0, 40)
    crop = make_crop_divisible_by_8(crop, nii_iso.shape)
    nii_iso = nii_iso.apply_crop_(crop)
    nii_iso /= nii_iso.max()
    nii_iso = nii_iso.clamp(min=0)
    nii_iso = nii_iso * 2 - 1
    target_shape = padded_shape(nii_iso.shape)
    nii_iso = nii_iso.pad_to(target_shape, mode="reflect")

    arr = Tensor(nii_iso.get_array().astype(float))
    # arr, padding = pad_size3D(arr)
    return arr, nii, nii_iso


def translate(mri_path: BIDS_FILE, use_cpu=False):
    log.on_log(f"translate {mri_path}")
    ref = mri_path.get_changed_path("nii.gz", "msk", "derivatives_spine_inference_162_sacrumfix", info={"seg": "spine", "chunk": None, "mod": "T2w", "sequ": "stitched"})
    ref_nii = to_nii(ref, True)
    arr, nii, nii_iso = prepare_nii(mri_path, ref_nii)
    if torch.cuda.is_available() and not use_cpu:
        # If you run out of memory use max_shape= (arr.shape[-1]*arr.shape[-2]*arr.shape[-3])//2
        ct_arr = run_model(diffusion, conditional=arr, gpu=True, eta=1, w=0, steps=25, depth=0)
    else:
        print("!!! Fall back to less steps on CPU instead of 25 GPU. !!!")
        ct_arr = run_model(diffusion, conditional=arr, gpu=False, eta=1, w=0, steps=5, depth=0)
    ct_nii_iso = nii_iso.set_array(ct_arr.numpy())
    ct_nii_iso[nii_iso <= -0.99] = -1024
    print("nii", nii)
    ct_nii: NII = ct_nii_iso.set_dtype("smallest_int").resample_from_to(nii, mode="constant")
    ct_nii = ct_nii.apply_crop(ct_nii.compute_crop())
    ct_path = mri_path.get_changed_path("nii.gz", "ct", "rawdata_synthetic", info={"desc": "generated", "acq": "sag"})
    ct_path_iso = mri_path.get_changed_path("nii.gz", "ct", "rawdata_synthetic", info={"desc": "generated", "acq": "iso"})
    print("ct", ct_nii)
    print("ct_nii", ct_nii_iso)

    ct_nii.set_dtype_("smallest_int").save(ct_path)
    ct_nii_iso.set_dtype_("smallest_int").save(ct_path_iso)
    return ct_nii, ct_nii_iso


if __name__ == "__main__":

    def filter_file(file: Path):
        return True
        return "118688_sequ-" in file.name

    bgi = Buffered_BIDS_Global_info(dataset, parents=["rawdata_stitched"], filter_file=filter_file)
    for name, subj in bgi.enumerate_subjects(shuffle=True):
        q = subj.new_query(flatten=True)
        q.filter_format("T2w")
        q.filter_filetype("nii.gz")
        for f in q.loop_list():
            translate(f)
        # exit()
