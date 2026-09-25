from __future__ import annotations
from pathlib import Path
import sys


file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))


import glob
import numpy as np
from torch.utils.data import Dataset
from math import floor, ceil
import os
import nibabel as nib

from dataloader.Wrapper_datasets import Wrapper_Image2Image

"""This data set combines many datasets for the universal ae.    """

from dataloader.dataloader_mri2ct import Wopathfx_Dataset
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision import transforms

dataset_names = (
    "night2day",
    "edges2handbags",
    "edges2shoes",
    "facades",
    "maps",
    # "MNIST",
    "horse2zebra",
    # "ae_photos",
    "apple2orange",
    "summer2winter_yosemite",
    "cezanne2photo",
    "ukiyoe2photo",
    "vangogh2photo",
    "grumpifycat",
)
dataset_names_single = ("afhq",)
# fx_T1w,


class Universal_dataset(Dataset):
    def __init__(
        self,
        train=True,
        size=256,
    ):
        self.datasets = []
        self.start = [0]
        self.dims = []
        self.name = []
        self.len = []

        ds = Wopathfx_Dataset(size=size, root="/media/data/robert/datasets/fx_T1w/", train=train, condition_types=["CT", "MRI"])  # type: ignore
        general_wrapper_info = {
            "image_dropout": 0,
            "size": 256,
            "inpainting": None,
            "compute_mean": False,
        }

        from loader.load_dataset import getDataset

        self.datasets.append(Wrapper_Image2Image(ds, **general_wrapper_info))
        self.dims.append(1)
        self.name.append("fx_t1w")
        self.start.append(len(ds))

        ds, i = getDataset(
            train=train, dataset="afhq", size=size, set_size="resize", compute_mean=False, flip=False
        )  # learning_type=""
        self.start.append(self.start[-1] + len(ds))
        self.datasets.append(ds)
        self.dims.append(i)
        self.name.append("afhq")
        print(len(ds))
        for ds_name in dataset_names:
            ds, i = getDataset(
                train=train, dataset=ds_name, size=size, set_size="resize", compute_mean=False, flip=False
            )  # learning_type=""
            self.start.append(self.start[-1] + len(ds))

            self.datasets.append(ds)
            self.dims.append(i)
            self.name.append(ds_name)
            print(len(ds))
            ds, i = getDataset(
                train=train, dataset=ds_name, size=size, set_size="resize", compute_mean=False, flip=True
            )  # learning_type=""
            self.start.append(self.start[-1] + len(ds))

            self.datasets.append(ds)
            self.dims.append(i)
            self.name.append(ds_name)

        for j, ds in enumerate(self.datasets):
            self.len += [j for _ in range(self.start[j], self.start[j + 1])]
        self.convert_tensor = transforms.ToTensor()

        print("total", len(self.len))

    def __getitem__(self, index):
        ds_idx = self.len[index]
        ds = self.datasets[ds_idx]
        dim = self.dims[ds_idx]
        start = self.start[ds_idx]
        # print(ds_idx, dim)
        # i = img["target"]

        img = ds[index - start]

        if dim == 1:
            import torch

            if isinstance(img["target"], torch.Tensor):
                return {"target": img["target"]}
            else:
                return {"target": self.convert_tensor(img["target"])}
        else:
            return {"target": rgb_to_grayscale(img["target"])}

    def __len__(self):
        return len(self.len)


if __name__ == "__main__":
    ds = Universal_dataset()
    print(list(zip(ds.name, ds.start)))

    for i in ds.start[:-1]:
        x = ds[i]

        assert "target" in x
