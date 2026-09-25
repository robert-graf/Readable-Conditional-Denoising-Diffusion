import glob
import numpy as np
from torch.utils.data import Dataset
from math import floor, ceil
import os
import nibabel as nib


class nii_Datasets(Dataset):
    def __init__(
        self,
        root="/media/data/robert/test/2022_06_21_T1_CT_wopathfx/dataset/traindata/train/**/",
        _transform=None,
        axis=-1,
        flip_last=False,
    ):
        self.files = sorted(glob.glob(root + "*.nii*"))
        assert len(self.files) > 0, f"No file found in {root},{len(self.files)}"
        self._transform = _transform
        self.axis = axis
        self.flip_last = flip_last

    def __getitem__(self, index):
        return nii_Dataset(self.files[index], axis=self.axis, _transform=self._transform, flip_last=self.flip_last)

    def __len__(self):
        return len(self.files)


class nii_Dataset(Dataset):
    def __init__(self, file, _transform=None, axis=-1, flip_last=False):
        self.file = file
        self.nii = nib.load(file)
        self.arr_data = None
        self._transform = _transform
        self.axis = axis
        self.flip_last = flip_last

    def load(self):
        if self.arr_data is None:
            self.arr_data = self.nii.get_fdata()
            self.affine = self.nii.affine
            if self.axis != -1:
                dims = len(self.arr_data.shape)
                axis_switch = [self.axis] + [i for i in range(dims) if i != self.axis]
                print(dims)
                print(axis_switch)
                print(self.arr_data.shape)

                assert (
                    dims > self.axis
                ), f"chose an axis that exist or -1 not {self.axis} for a {dims} dimensional array"

                self.arr_data = np.transpose(self.arr_data, tuple(axis_switch))
            if self.flip_last:
                self.arr_data = np.swapaxes(self.arr_data, -1, -2)

    def save(self, arr: np.ndarray, file: str):
        if self.flip_last:
            arr = np.swapaxes(arr, -1, -2)
        if self.axis != -1:
            dims = len(arr.shape)
            axis_switch = [i for i in range(1, dims)]
            axis_switch.insert(self.axis, 0)
            print(dims)
            print(axis_switch)
            print(arr.shape)
            arr = np.transpose(arr, tuple(axis_switch))

        img = nib.Nifti1Image(arr, self.affine)
        print(arr.shape)
        if not file.endswith(".nii.gz"):
            file += ".nii.gz"
        # save
        nib.save(img, file)

    def get_copy(self):
        self.load()
        return self.arr_data.copy()

    def __getitem__(self, index):
        self.load()
        if self._transform is not None:
            return self._transform(self.arr_data(index))
        return self.arr_data[index]

    def __len__(self):
        return self.arr_data.shape[0]

    def __str__(self) -> str:
        self.load()
        return f"Nii_dataset: \n file = {self.file}\n axis = {self.axis}\n shape = {self.nii.shape}"


if __name__ == "__main__":
    in_files = "C:/Users/rober/Desktop/"

    out_file = "C:/Users/rober/Desktop/9x9_vergleich_test.nii.gz"
    # axis is the axis you recive for looping over the data set
    for i, nii_ds in enumerate(nii_Datasets(root=in_files, axis=0)):
        print(nii_ds)
        arr = nii_ds.get_copy() * 0
        for j, layer in enumerate(nii_ds):
            arr[j] = layer
            if j % 20 == 0:
                arr[j] = 100
        # Do a smarter way on produciont out_file
        nii_ds.save(arr, out_file)
