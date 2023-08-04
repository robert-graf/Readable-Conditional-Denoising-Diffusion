from __future__ import annotations
import glob
from math import ceil, floor
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
from utils.nii_utils import resample_nib, reorient_to
import numbers

import torchvision
from torch import Tensor
from typing import Literal, Tuple
from torchvision.transforms.functional import crop as f_crop, center_crop


def pad_to(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    padding = []
    for dim, trg in zip((x.shape), (shape)):
        a = (trg - dim) / 2
        padding.extend([(floor(a), ceil(a))])
    x = np.pad(x, padding, mode="symmetric")
    return x


def crop_to(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    p = []
    for dim, trg in zip((x.shape), (shape)):
        a = -(trg - dim) / 2
        p.append((floor(a), ceil(a)))
    x_new = x[
        p[0][0] : -p[0][1],
        p[1][0] : -p[1][1],
        p[2][0] : -p[2][1],
    ]
    return x_new


class nii_Datasets(Dataset):
    def __init__(
        self,
        root: str | Path = "/media/data/robert/test/2022_06_21_T1_CT_wopathfx/dataset/traindata/train/**/",
        _transform=None,
        orientation=("R", "I", "P"),  # Right
        keep_scale=["R"],
        match_string="*.nii*",
        min_value=-1024,
    ) -> None:
        root = str(root).replace("+", "*")
        self.files = sorted(glob.glob(root + match_string))
        assert len(self.files) > 0, f"No file found in {root+match_string},{len(self.files)}"
        self._transform = _transform
        self.orientation = orientation
        self.keep_scale = keep_scale
        self.min_value = min_value

    def __getitem__(self, index) -> nii_Dataset:
        return nii_Dataset(
            self.files[index],
            _transform=self._transform,
            orientation=self.orientation,
            keep_scale=self.keep_scale,
            min_value=self.min_value,
        )

    def __len__(self) -> int:
        return len(self.files)


class nii_Dataset(Dataset):
    def __init__(
        self,
        file: str | Path,
        _transform=None,
        orientation: Tuple[str, str, str] = ("R", "I", "P"),
        keep_scale=["R"],
        min_value=0,
    ) -> None:
        #'S': 'ax', 'I': 'ax', 'L': 'left/sag', 'R': 'right/sag', 'A': 'cor', 'P': 'cor'
        self.file = str(file)
        self._nii: nib.Nifti1Image = nib.load(str(file))
        f = Path(file)
        # TODO a more general way to compute the seg_path
        self.file_seg = str(Path(f.parent.parent, "Mask", "mask_" + f.name.lower()))
        try:
            self.nii_seg: nib.Nifti1Image | None = nib.load(self.file_seg)
        except:
            self.nii_seg = None
        self.arr_data: np.ndarray = None  # type: ignore
        self._transform = _transform
        self.orientation = orientation
        self.keep_scale = keep_scale
        self.min_value = min_value

    @property
    def nii(self) -> nib.Nifti1Image:
        self.load()
        return self._nii

    def load(self) -> None:
        if self.arr_data is None:
            self.affine_old = self._nii.affine
            self.header_old = self._nii.header
            # rotate
            self._nii = reorient_to(self._nii, axcodes_to=self.orientation)
            if self.nii_seg is not None:
                self.nii_seg = reorient_to(self.nii_seg, axcodes_to=self.orientation)
            # respace to (1,1,1) or keep_scale does not change
            new_spacing = [1, 1, 1]
            zoom = self._nii.header.get_zooms()  # type: ignore
            for i, key in enumerate(self.orientation):
                if key in self.keep_scale:
                    new_spacing[i] = zoom[i]
            self._nii = resample_nib(self._nii, voxel_spacing=new_spacing, c_val=self.min_value)
            if self.nii_seg is not None:
                self.nii_seg = resample_nib(self.nii_seg, voxel_spacing=new_spacing, c_val=0)

            self.arr_data = self._nii.get_fdata()  # type: ignore
            if self.nii_seg is not None:
                self.arr_seg = self.nii_seg.get_fdata()
            self.affine = self._nii.affine
            self.header: nib.nifti1.Nifti1Header = self._nii.header  # type: ignore

    def save(self, arr: np.ndarray | Tensor, file: str | Path, revert_to_size=True, min_value=None) -> None:
        """save the array to a file and may revert sampling changes

        Args:
            arr (np.ndarray | Tensor):
                This tensor will be stored in a .nii.gz
            file (str):
                file path with or without the .nii.gz ending
            revert_to_size (bool, optional):
                The load method may resample and rotate to the same space. When True, this process is reverted. Defaults to True.
        """
        if min_value == None:
            min_value = self.min_value
        if isinstance(arr, Tensor):
            img: np.ndarray = arr.clone().detach().cpu().numpy()
        else:
            img: np.ndarray = arr.copy()
        img = pad_to(img, (img.shape[0] + 2, img.shape[1] + 2, img.shape[2] + 2))
        img_nii = nib.Nifti1Image(img, self.affine, self.header)  # self.header
        if revert_to_size:
            import nibabel.orientations as nio

            axc = nio.io_orientation(self.affine_old)
            axc = nio.ornt2axcodes(axc)
            img_nii = reorient_to(img_nii, axcodes_to=axc)
            img_nii = resample_nib(img_nii, voxel_spacing=self.header_old.get_zooms(), c_val=min_value)  # type: ignore
            org = nib.load(self.file)
            img_nii = nib.Nifti1Image(crop_to(img_nii.get_fdata().round(5), org.shape), img_nii.affine, self.header)

        if not str(file).endswith(".nii.gz"):
            file = str(file) + ".nii.gz"
        # save
        nib.save(img_nii, file)

    def get_copy(
        self, crop: int | Tuple[int, int] | Tuple[int, int, int, int] | None = None, clear: bool = False, seg=False
    ) -> Tensor:
        """get a transformed and optionally cropped 3D Tensor

        Args:
            crop (int | Tuple[int, int] | Tuple[int, int, int, int] | None, optional):
                Crop dimension;
                None is no crop
                1D or 2D is a Center_crop
                4D is a Crop with offset in the first two dimensions.
                zero padding if necessary.
                Defaults to None.
            clear (bool, optional): Clear the values to 0. Defaults to False.
            seg (bool, optional): Load the segmentation instead of the image. Defaults to False.

        Returns:
            Tensor:
        """
        self.load()
        # print("crop", crop)

        if seg:
            data = Tensor(self.arr_seg.copy())
        else:
            assert self.arr_data is not None
            data = Tensor(self.arr_data.copy())
        if clear:
            data *= 0
        if crop is None:
            return data
        if isinstance(crop, int):
            crop = (int(crop), int(crop))
        if len(crop) == 1:
            crop = (crop[0], crop[0])
        if len(crop) == 2:
            return center_crop(data, list(crop))
        if len(crop) == 4:
            return f_crop(data, crop[0], crop[1], crop[2], crop[3])
        assert False, f"{data.shape}|{crop} not supported"

    def insert_sub_corp(self, in_data: Tensor, offset: Tuple[int, int] | None = None, fill_value=0) -> Tensor:
        if isinstance(in_data, Tensor):
            in_data = in_data.detach().cpu()
        else:
            in_data = Tensor(in_data)
        data = Tensor(self.arr_data.copy())
        data *= 0
        data += fill_value
        if offset is None:
            return center_crop(in_data, list(data.shape[-2:]))
        if len(offset) == 2:

            if in_data.shape[-2] > data.shape[-2]:
                in_data = in_data[..., : data.shape[-2], :]

            if in_data.shape[-1] > data.shape[-1]:
                in_data = in_data[..., : data.shape[-1]]

            h_max = offset[0] + in_data.shape[-2]
            w_max = offset[1] + in_data.shape[-1]
            if h_max > data.shape[-2]:
                data = data[..., : data.shape[-2] - offset[0], :]
                h_max = data.shape[-2]
            if w_max > data.shape[-1]:
                data = data[..., : data.shape[-1] - offset[1]]
                w_max = data.shape[-1]
            data[..., offset[0] : h_max, offset[1] : w_max] = in_data
            return data
        assert False, f"{data.shape}|{offset} not supported"

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        self.load()
        if self._transform is not None:
            return self._transform(self.arr_data[index])
        return self.arr_data[index], self.arr_seg[index]

    def __len__(self) -> int:
        return self.arr_data.shape[0]

    def __str__(self) -> str:
        self.load()
        return f"Nii_dataset: \n file = {self.file}\n axis = {self.orientation}\n shape = {self._nii.shape}"


if __name__ == "__main__":
    in_files = "C:/Users/rober/Desktop/"

    out_file = "C:/Users/rober/Desktop/9x9_vergleich_test.nii.gz"
    # axis is the axis you receive for looping over the data set
    for i, nii_ds in enumerate(nii_Datasets(root=in_files)):  # type: ignore
        print(nii_ds)
        arr = nii_ds.get_copy() * 0
        for j, layer in enumerate(nii_ds):
            arr[j] = layer
            if j % 20 == 0:
                arr[j] = 100
        # Do a smarter way on productions out_file
        nii_ds.save(arr, out_file)
