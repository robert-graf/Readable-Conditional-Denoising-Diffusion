import glob
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from torch.nn import functional as F
from math import floor, ceil
import torch
from torch import nn
import os

from torch import Tensor
from utils.data_augmentation_3D import SpatialFlip, ColorJitter3D, PadIfNecessary
import nibabel as nib


def pad(x, mod: int):
    padding = []
    for dim in reversed(x.shape[1:]):
        padding.extend([0, (mod - dim % mod) % mod])
    x = F.pad(x, padding)
    return x


def pad_size(x: Tensor, target_shape):
    padding = []
    for in_size, out_size in zip(reversed(x.shape[-3:]), reversed(target_shape)):
        to_pad_size = max(0, out_size - in_size) / 2.0
        padding.extend([ceil(to_pad_size), floor(to_pad_size)])
    x_ = (
        F.pad(x.unsqueeze(0).unsqueeze(0), padding, mode="constant").squeeze(0).squeeze(0)
    )  # mode – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
    # print("Padding", x_.shape, x.shape, padding)
    return x_


def random_crop(x, target_shape):
    # for in_size, out_size, dim in reversed(zip(x.shape[1:], target_shape, [1,2,3])):
    to_remove = max(0, x.shape[-1] - target_shape[-1]) / 2.0
    if to_remove != 0:
        x = x[..., floor(to_remove) : -ceil(to_remove)]
    to_remove = max(0, x.shape[-2] - target_shape[-2]) / 2.0
    if to_remove != 0:
        x = x[..., floor(to_remove) : -ceil(to_remove), :]
    to_remove = max(0, x.shape[-3] - target_shape[-3]) / 2.0
    if to_remove != 0:
        x = x[..., floor(to_remove) : -ceil(to_remove), :, :]
    return x


class MRI2CT_3D_Dataset(Dataset):
    def __init__(
        self,
        root="/media/data/robert/datasets/TGD_3D/",
        factor: int = 2,
        flip: bool = False,
        padding="constant",  # constant, edge, reflect or symmetric
        mri_transform=transforms.Compose([ColorJitter3D(brightness_min_max=(0.8, 1.2), contrast_min_max=(0.8, 1.2))]),
        # counter_example=False,
        # counter_example_v2=False,
        image_dropout=0,
        condition_types=["mri"],  # Unused
        train=True,
        manual_downsample=True,
    ):
        if train:
            root_ct = os.path.join(root, "ct", "train")
            root_mri = os.path.join(root, "mri", "train")
        else:
            root_ct = os.path.join(root, "ct", "test")
            root_mri = os.path.join(root, "mri", "test")
        self.files_ct = sorted(glob.glob(root_ct + "/*/*.nii*")) + sorted(glob.glob(root_ct + "*/*.nii*"))
        assert len(self.files_ct) > 0, f"No file found in {root_ct},{len(self.files_ct)}"
        self.files_mri = sorted(glob.glob(root_mri + "/*/*.nii*")) + sorted(glob.glob(root_mri + "*/*.nii*"))
        base_name = [os.path.basename(i) for i in self.files_ct]
        self.files_mri = [i for i in self.files_mri if os.path.basename(i) in base_name]
        base_name = [os.path.basename(i) for i in self.files_mri]
        self.files_ct = [i for i in self.files_ct if os.path.basename(i) in base_name]
        assert len(self.files_ct) > 0, f"No file found in {root_ct},{len(self.files_ct)}"

        self.factor = factor
        self.flip = flip
        self.padding = padding
        self.mri_transform = mri_transform
        self.image_dropout = image_dropout
        self.train = train
        self.manual_downsample = False  # manual_downsample
        if self.manual_downsample:
            self.target_shape = (24, 2**5 * 4, 2**5 * 4)
            self.factor_shape = (1, factor, factor)
        else:
            self.target_shape = (24, factor, factor)

        # self.pad = PadIfNecessary(3)
        self.spacial_flip = SpatialFlip(dims=(1, 2, 3), auto_update=False)

    def load_file(self, file_ct, file_mri):
        ct = nib.load(file_ct)
        ct_data = ct.get_fdata()
        ct_affine = ct.affine

        mri = nib.load(file_mri)
        mri_data = mri.get_fdata()
        mri_affine = mri.affine

        return ct_data, mri_data, ct_affine, mri_affine

    @torch.no_grad()
    def transform(self, ct, mri):
        # Normalize to -1, 1
        ct /= 1000
        ct = ct.clip(-1, 1)
        mri = mri / np.max(mri)
        mri = mri.clip(0, 1e6)

        # Transform to tensor
        ct = functional.to_tensor(ct)
        mri = functional.to_tensor(mri)

        ## Padding
        ct = pad_size(ct, self.target_shape)
        mri = pad_size(mri, self.target_shape)
        ## Random crop
        ct = random_crop(ct, self.target_shape)
        mri = random_crop(mri, self.target_shape)

        if not self.mri_transform is None:
            mri = self.mri_transform(mri)

        # old_shape = ct.shape
        if self.manual_downsample:
            ct = F.avg_pool3d(ct.unsqueeze_(0), self.factor_shape, stride=self.factor_shape)
            mri = F.avg_pool3d(mri.unsqueeze_(0), self.factor_shape, stride=self.factor_shape)

        # Random flipping
        if self.flip and random.random() > 0.5:
            self.spacial_flip.update()
            ct = self.spacial_flip(ct)
            mri = self.spacial_flip(mri)

        if self.image_dropout > 0 and self.train and self.image_dropout > random.random():
            mri = mri * 0

        mri = mri * 2 - 1
        # print("mri0", mri.min(), mri.max())

        # print("DATASET", ct.shape, mri.shape, self.target_shape)
        if self.manual_downsample:
            return ct.to(torch.float32), mri.to(torch.float32)

        return ct.to(torch.float32).unsqueeze_(0), mri.to(torch.float32).unsqueeze_(0)

    def __getitem__(self, index):
        item_A, item_B, *_ = self.load_file(self.files_ct[index % len(self)], self.files_mri[index % len(self)])

        a = self.transform(item_A, item_B)
        # print("mri", a[1].min(), a[1].max())
        # print(self.files_ct[index % len(self)], self.files_mri[index % len(self)])
        return a

    def __len__(self):
        return len(self.files_ct)
