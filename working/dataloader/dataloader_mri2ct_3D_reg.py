import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
from pathlib import Path
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from torch.nn import functional as F
from math import floor, ceil
import torch
import pandas
from torch import Tensor
from utils.data_augmentation_3D import SpatialFlip, ColorJitter3D
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


def random_crop(target_shape: tuple[int, int, int], *arrs: Tensor):
    sli = [slice(None), slice(None), slice(None)]
    for i in range(3):
        z = max(0, arrs[0].shape[-i] - target_shape[-i])
        if z != 0:
            r = random.randint(0, z)
            r2 = r + target_shape[-i]
            sli[-i] = slice(r, r2 if r2 != arrs[0].shape[-i] else None)

    return tuple(a[..., sli[0], sli[1], sli[2]] for a in arrs)


class MRI2CT_3D_Dataset(Dataset):
    def __init__(
        self,
        root="/media/data/robert/datasets/spinegan_T2w_all_reg_iso/",
        sub_lists=["registration"] + [f"registration_deformed/{i}" for i in range(20)],
        flip: bool = False,
        padding="constant",  # constant, edge, reflect or symmetric
        mri_transform=transforms.Compose([ColorJitter3D(brightness_min_max=(0.8, 1.2), contrast_min_max=(0.8, 1.2))]),
        image_dropout=0.0,
        train=True,
        target_shape=(32, 128, 128),
    ):
        # TODO make a train.csv with all patients
        # "spinegan0002/target-301_from-201"
        self.root = Path(root)
        self.sub_lists = sub_lists
        if train:
            self.pd = pandas.read_csv(Path(self.root, "train.csv"))
        else:
            self.pd = pandas.read_csv(Path(self.root, "val.csv"))

        self.flip = flip
        self.padding = padding
        self.mri_transform = mri_transform
        self.image_dropout = image_dropout
        self.train = train
        self.spacial_flip = SpatialFlip(dims=(2, 3), auto_update=False)
        self.target_shape = target_shape

    def load_file(self, id):
        id1 = id % len(self.pd)
        id2 = id // len(self.pd)
        name = self.root / Path(self.sub_lists[id2], self.pd.iloc[id1]["name"])
        ct_f = next(name.glob("sub-spinegan*_ses-*_sequ-*_reg-*_ct.nii.gz"))
        mri_f = next(name.glob("sub-spinegan*_ses-*_sequ-*_reg-*_dixon.nii.gz"))
        ct_nib: nib.Nifti1Image = nib.load(str(ct_f))
        mri_nib: nib.Nifti1Image = nib.load(str(mri_f))
        return ct_nib.get_fdata(), mri_nib.get_fdata()

    @torch.no_grad()
    def transform(self, ct, mri):
        # Normalize to -1, 1
        ct /= 1000
        ct = ct.clip(-1, 1)
        mri = mri / np.max(mri)
        mri = mri.clip(0, 1e6)

        # Transform to tensor
        ct = Tensor(ct)
        mri = Tensor(mri)

        ## Padding
        ct = pad_size(ct, self.target_shape)
        mri = pad_size(mri, self.target_shape)
        # Coordinate-encoding
        shape = ct.shape
        l1 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
        l2 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
        l3 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
        l1 = Tensor(l1).permute(2, 0, 1)
        l2 = Tensor(l2).permute(0, 2, 1)
        l3 = Tensor(l3)
        assert l1.shape == l2.shape, (l1.shape, l2.shape)
        assert l3.shape == l2.shape, (l3.shape, l2.shape)
        assert ct.shape == l2.shape, (ct.shape, l2.shape)
        assert mri.shape == l2.shape, (mri.shape, l2.shape)
        ## Random crop
        ct, mri, l1, l2, l3 = random_crop(self.target_shape, ct, mri, l1, l2, l3)

        if not self.mri_transform is None:
            mri = self.mri_transform(mri)

        # Random flipping
        if self.flip and random.random() > 0.5:
            self.spacial_flip.update()
            ct = self.spacial_flip(ct)
            mri = self.spacial_flip(mri)

        if self.image_dropout > 0 and self.train and self.image_dropout > random.random():
            mri = mri * 0

        mri = mri * 2 - 1
        out = ct, mri, l1, l2, l3
        out = tuple(a.to(torch.float32).unsqueeze_(0) for a in out)
        return out[0], torch.cat(out[1:], 0)

    def __getitem__(self, index):
        item_A, item_B = self.load_file(index)
        a = self.transform(item_A, item_B)
        return a

    def __len__(self):
        return len(self.pd) * len(self.sub_lists)


if __name__ == "__main__":
    ds = MRI2CT_3D_Dataset()
    print(len(ds))
    print(tuple(i.shape for i in ds[1]))
