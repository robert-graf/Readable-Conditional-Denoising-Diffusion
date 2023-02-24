from __future__ import annotations

import glob
import os
from pathlib import Path
import random
from math import ceil, floor

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.nn import functional as NF
from torch.utils.data import Dataset


class Wopathfx_Dataset(Dataset):
    def __init__(
        self,
        root: str | list[str] = "/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/dataset/traindata/",
        size: int | tuple[int, int] = 320,
        vflip=False,
        hflip=True,
        padding="constant",  # constant, edge, reflect or symmetric
        mri_transform=transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2)]),
        image_dropout=0,
        condition_types=["MRI"],
        train=True,
        gauss=False,
        norm=False,
    ):

        if not isinstance(root, list):
            root = [root]

        for i in root:
            self.files = []
            if train == "test":
                i = os.path.join(str(i), "test")
            elif train:
                i = os.path.join(str(i), "train")
            else:
                i = os.path.join(str(i), "val")
            self.files += sorted(glob.glob(i + "/*/*.npz")) + sorted(glob.glob(i + "/*.npz"))
        assert len(self.files) > 0, f"No file found in {root},{len(self.files)}"
        if isinstance(size, int):
            size = (size, size)

        self.size = size
        self.vflip = vflip
        self.hflip = hflip
        self.padding = padding
        self.gauss = gauss
        if train:
            self.mri_transform = mri_transform
        else:
            self.mri_transform = None
        self.image_dropout = image_dropout
        condition_types = [i if i != "MRI" else "T1" for i in condition_types]
        self.condition_types = condition_types

        self.train = train
        self.norm = norm

    def get_conditional_channel_size(self):
        return max(len(self.condition_types) - 1, 1)

    def load_file(self, name):
        dict_mods = {}
        if name.endswith(".npz"):
            f = np.load(name)
            for k, v in f.items():  # type: ignore
                dict_mods[k] = v.astype("f")
                if self.norm:
                    dict_mods[k] /= max(float(np.max(dict_mods[k])), 0.0000001)
            f.close()  # type: ignore
            return dict_mods
        assert False, "Expected a .npz file"

    def gauss_filter(self, img_data) -> torch.Tensor | np.ndarray:
        if self.gauss:
            to_tensor = False
            if isinstance(img_data, torch.Tensor):
                img_data = img_data.detach().cpu().numpy()
                to_tensor = True
            from scipy import ndimage

            out: np.ndarray = ndimage.median_filter(img_data, size=3)  # type: ignore
            if to_tensor:
                return torch.from_numpy(out)
            return out
        return img_data

    @torch.no_grad()
    def transform(self, dict_mods):
        condition_types = self.condition_types

        if len(condition_types) == 1:
            assert "CT" in dict_mods
            ct = self.gauss_filter(dict_mods["CT"])
            target = TF.to_tensor(ct)
        else:
            key = condition_types[0]
            condition_types = condition_types[1:]
            target = dict_mods[key]
            target = TF.to_tensor(target)
            if key != "CT" and key != "SG" and self.mri_transform is not None:
                target = self.mri_transform(torch.cat([target, target, target], dim=0))[1:2]

        second_img_list: list[torch.Tensor] = []
        for key in condition_types:
            img = TF.to_tensor(dict_mods[key])
            if "CT" == key:
                img = self.gauss_filter(img)
            # elif "CT_4" == key:
            #    ct_4 = TF.to_tensor(ct)
            #    ct_4 = NF.avg_pool2d(ct_4.unsqueeze_(0), 4, stride=4).squeeze()
            #    pixel_shuffle = torch.nn.PixelShuffle(4)
            #    ct_4 = pixel_shuffle(ct_4.unsqueeze(0).repeat(4**2, 1, 1).unsqueeze(0)).squeeze(0).squeeze(0)
            #    ct_4 = NF.pad(ct_4, [0, ct.shape[1] - ct_4.shape[1], 0, ct.shape[0] - ct_4.shape[0]])
            #    second_img_list.append(ct_4.unsqueeze_(0))
            # elif "CT_2" == key:
            #    ct_4 = TF.to_tensor(ct)
            #    ct_4 = NF.avg_pool2d(ct_4.unsqueeze_(0), 2, stride=2).squeeze()
            #    pixel_shuffle = torch.nn.PixelShuffle(2)
            #    ct_4 = pixel_shuffle(ct_4.unsqueeze(0).repeat(2**2, 1, 1).unsqueeze(0)).squeeze(0).squeeze(0)
            #    ct_4 = NF.pad(ct_4, [0, ct.shape[1] - ct_4.shape[1], 0, ct.shape[0] - ct_4.shape[0]])
            #    second_img_list.append(ct_4.unsqueeze_(0))
            elif "SG" == key:
                pass
            else:  # MRI
                if self.mri_transform is not None:
                    img = self.mri_transform(torch.cat([img, img, img], dim=0))[1:2]
            second_img_list.append(img)  # type: ignore

        if self.image_dropout > 0 and self.train and self.image_dropout > random.random():
            second_img_list[0] = second_img_list[0] * 0
        for id, i in enumerate(second_img_list):
            assert second_img_list[0].shape == i.shape, f"Shape mismatch {second_img_list[0].shape} {i.shape} "

        second_img = torch.cat(second_img_list, dim=0)
        # Padding
        w, h = target.shape[-2], target.shape[-1]
        hp = max((self.size[0] - w) / 2, 0)
        vp = max((self.size[1] - h) / 2, 0)
        padding = [int(floor(vp)), int(floor(hp)), int(ceil(vp)), int(ceil(hp))]

        target = TF.pad(target, padding, padding_mode=self.padding)
        second_img = TF.pad(second_img, padding, padding_mode=self.padding)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(target, output_size=self.size)
        target = TF.crop(target, i, j, h, w)
        second_img = TF.crop(second_img, i, j, h, w)

        # Random horizontal flipping
        if self.hflip and random.random() > 0.5:
            target = TF.hflip(target)
            second_img = TF.hflip(second_img)

        # Random vertical flipping
        if self.vflip and random.random() > 0.5:
            target = TF.vflip(target)
            second_img = TF.vflip(second_img)

        # Normalize to -1, 1
        target = target * 2 - 1
        second_img = second_img * 2 - 1

        return target, second_img

    def __getitem__(self, index):
        dict_mods = self.load_file(self.files[index % len(self.files)])
        return self.transform(dict_mods)

    def __len__(self):
        return len(self.files)
