from __future__ import annotations
import glob
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.nn import functional as NF
from math import floor, ceil
import torch
import os

from utils.data_augmentation_3D import ColorJitter3D, SpatialFlip


class Vertebra_Dataset(Dataset):
    def __init__(
        self,
        root,
        flip=True,
        mri_transform=transforms.Compose([ColorJitter3D(brightness_min_max=(0.8, 1.2), contrast_min_max=(0.8, 1.2))]),
        # image_dropout=0,
        condition_types=["MRI"],
        train=True,
        size=96,
    ):

        if train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "val")
        if isinstance(root, list):
            self.files = []
            for i in root:
                self.files += sorted(glob.glob(i + "/*/*.npz")) + sorted(glob.glob(i + "/*.npz"))
        else:
            self.files = sorted(glob.glob(root + "/*/*.npz")) + sorted(glob.glob(root + "/*.npz"))
        assert len(self.files) > 0, f"No file found in {root},{len(self.files)}"
        self.flip = flip
        self.spacial_flip = SpatialFlip(dims=(1, 2), auto_update=False)
        if train:
            self.mri_transform = mri_transform
        else:
            self.mri_transform = None
        # self.image_dropout = image_dropout
        self.condition_types = condition_types
        self.train = train
        self.size = size

    def get_conditional_channel_size(self):
        return len(self.condition_types)

    def load_file(self, name) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if name.endswith(".npz"):
            f = np.load(name)
            ct_image_slice = f["CT"].astype("f")  # range [-1-1]
            mr_t1_image_slice = f["T1"].astype("f")  # range [-1-1]
            segmentation_slice = f["SG"].astype("f")  # integer
            vert_idx = int(f["vert_idx"].astype("f"))
            f.close()
            return ct_image_slice, mr_t1_image_slice, segmentation_slice, vert_idx
        assert False, "Expected a .npz file"

    @torch.no_grad()
    def transform(self, ct, mri, seg):
        second_img = []

        if "CT" in self.condition_types:
            ct_ = TF.to_tensor(ct)
            second_img.append(ct_)
        if "MRI" in self.condition_types:
            mri_ = TF.to_tensor(mri)
            if not self.mri_transform is None:
                # print(mri_.shape, torch.cat([mri_, mri_, mri_], dim=1).shape)
                mri_ = self.mri_transform(mri_)
            second_img.append(mri_)
        if "SG" in self.condition_types:
            seg_ = TF.to_tensor(seg)
            second_img.append(seg_)
        if "CT_4" in self.condition_types:
            raise NotImplementedError()
        if "CT_2" in self.condition_types:
            raise NotImplementedError()
        if "target_SG" in self.condition_types:
            target = TF.to_tensor(seg)
        elif "target_MRI" in self.condition_types:
            target = TF.to_tensor(mri)
        elif "target_CT" in self.condition_types:
            target = TF.to_tensor(ct)
        # Sensable Defaults
        elif "CT" in self.condition_types:
            target = TF.to_tensor(mri)
        else:
            target = TF.to_tensor(ct)
        # if self.image_dropout > 0 and self.train and self.image_dropout > random.random():
        #    second_img[0] = second_img[0] * 0
        for id, i in enumerate(second_img):
            assert second_img[0].shape == i.shape, f"Shape mismatch {second_img[0].shape} {id.shape} "
        second_img = torch.cat(second_img, dim=0)
        # Padding

        target.unsqueeze_(0)
        second_img.unsqueeze_(0)

        d, w, h = target.shape[-3], target.shape[-2], target.shape[-1]
        max_wh = self.size
        dp = max((max_wh // 2 - d) / 2, 0)
        hp = max((max_wh - w) / 2, 0)
        vp = max((max_wh - h) / 2, 0)
        padding = (int(floor(vp)), int(ceil(vp)), int(floor(hp)), int(ceil(hp)), int(floor(dp)), int(ceil(dp)))
        # print(target.shape, second_img.shape, padding)
        target = torch.nn.functional.pad(target, pad=padding, mode="constant")
        second_img = torch.nn.functional.pad(second_img, pad=padding, mode="constant")

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(target, output_size=(self.size, self.size))
        target = TF.crop(target, i, j, h, w)
        second_img = TF.crop(second_img, i, j, h, w)

        dp = -ceil(min((max_wh // 2 - d) / 2, 0))

        if dp != 0:
            target = target[..., dp : dp + max_wh // 2, :, :]
            second_img = second_img[..., dp : dp + max_wh // 2, :, :]

        # Random horizontal flipping
        if self.flip and random.random() > 0.5:
            self.spacial_flip.update()
            target = self.spacial_flip(target)
            second_img = self.spacial_flip(second_img)
        # Normalize to -1, 1
        ### Already normalized
        # target = target * 2 - 1
        # second_img = second_img * 2 - 1

        # Polling
        target = NF.avg_pool3d(target, (2, 2, 2), stride=2)
        second_img = NF.avg_pool3d(second_img, (2, 2, 2), stride=2)

        return target, second_img

    def __getitem__(self, index):
        item_A, item_B, Seg, Vert_idx = self.load_file(self.files[index % len(self.files)])
        a, b = self.transform(item_A, item_B, Seg)
        return a, b, Vert_idx

    def __len__(self) -> int:
        return len(self.files)
