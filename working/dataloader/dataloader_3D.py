import math
import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
import random
import time
from math import ceil, floor
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nio
import nibabel.processing as nip
import numpy as np
import pandas
import torch
import torchvision.transforms as transforms
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from utils.data_augmentation_3D import ColorJitter3D, SpatialFlip

# Defining relationships
# 0 Pixel matching
# 1 Space matching
# 2 affine registation needed
#
# Ship
# $Ship1 0 *_T1w *_T2w
# $Ship2 0 *rec-in_mevibe *rec_fat_mevibe *rec_water_mevibe
# $Ship3 1 $Ship1 $Ship2 *T2star
#
# CT -1000 1000
# Rest 0 - X

# Assume same space
# Phase Path CT     T2w       T2w
# train path target condition condition ...


def pad(x, mod: int):
    padding = []
    for dim in reversed(x.shape[1:]):
        padding.extend([0, (mod - dim % mod) % mod])
    x = F.pad(x, padding)
    return x


def pad_size(x: Tensor, target_shape, mode="constant"):
    padding = []
    for in_size, out_size in zip(reversed(x.shape[-3:]), reversed(target_shape)):
        to_pad_size = max(0, out_size - in_size) / 2.0
        padding.extend([ceil(to_pad_size), floor(to_pad_size)])
    x_ = (
        F.pad(x.unsqueeze(0).unsqueeze(0), padding, mode=mode).squeeze(0).squeeze(0)
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


class SameSpace_3D_Dataset(Dataset):
    def __init__(
        self,
        root,
        condition_types,
        target_shape,
        flip: bool = False,
        padding="reflect",  # constant, edge, reflect or symmetric
        mri_transform=transforms.Compose([ColorJitter3D(brightness_min_max=(0.8, 1.2), contrast_min_max=(0.8, 1.2))]),
        train=True,
        output_channels=1,
    ):
        # TODO make a train.csv with all patients
        # "spinegan0002/target-301_from-201"
        self.condition_types = [x for x in condition_types if x != "linespace"]
        self.output_channels = output_channels

        self.root = Path(root)
        if ";" in str(root):
            self.pd = pandas.concat([pandas.read_excel(r) for r in str(root).split(";")])
        elif self.root.is_file():
            if self.root.name.endswith(".csv"):
                self.pd = pandas.read_csv(root)
            else:
                self.pd = pandas.read_excel(root)
            self.root = self.root.parent
            if "resampled" in self.root.name:
                self.root = self.root.parent
        else:
            self.pd = pandas.read_csv(Path(self.root, "train.csv"))

        self.pd = self.pd[self.pd["Phase"] == ("train" if train else "val")]
        assert len(self.pd) != 0, self.pd
        self.flip = flip
        self.padding = padding
        self.mri_transform = mri_transform
        self.train = train
        self.spacial_flip = SpatialFlip(dims=(2, 3), auto_update=False)
        self.target_shape = target_shape
        self.count = 10
        self.t_max = 0

    def load_file(self, id, keys=["CT", "T2w"]):
        self.count -= 1

        id1 = id % len(self.pd)
        out = []
        files = []
        for key in keys:
            # f = #self.root /
            assert isinstance(self.pd.iloc[id1][key], str) or not math.isnan(self.pd.iloc[id1][key]), "detected empty entry\n" + str(
                self.pd.iloc[id1]
            )
            if "Path" in self.pd.iloc[id1]:
                p = self.pd.iloc[id1]["Path"]
            else:
                p = ""
            if not str(p).startswith("/"):
                p = Path(self.root, p)
            f = Path(p, self.pd.iloc[id1][key])
            if not f.exists():
                f = str(f).replace("rawdata", "derivatives")
            if not Path(f).exists():
                f = str(f).replace("derivatives", "rawdata")
            if not Path(f).exists():
                f = str(f).replace("rawdata", "resampled")
            if not Path(f).exists():
                f = str(f).replace("rawdata", "translated/rawdata")
            if not Path(f).exists():
                f = str(f).replace("rawdata", "translated/derivatives")
            if not Path(f).exists():
                assert False, f"{f} does not exit"
            if str(f).endswith("nii.gz"):
                # print(f)
                nii: nib.Nifti1Image = nib.load(str(f))  # type: ignore
                aff = nii.affine
                ornt_fr = nio.io_orientation(aff)
                try:
                    arr: np.ndarray = nii.get_fdata()
                except EOFError:
                    print("EOF-ERROR", f)
                    Path(f).unlink()
                    return self.load_file(id + 1, keys=keys)
                except Exception as e:
                    print(f)
                    print(f)
                    print(f)
                    print(e)
                    print(f)
                    print(f)
                    print(f)
                    return self.load_file(id + 1, keys=keys)
                ornt_to = nio.axcodes2ornt(("R", "I", "P"))
                ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)

                arr = nio.apply_orientation(arr, ornt_trans)
                aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
                new_aff = np.matmul(aff, aff_trans)
                if len(out) == 0:
                    inital = nib.Nifti1Image(arr, new_aff)
                elif arr.shape != out[0].shape:
                    print("resampled", arr.shape, inital.shape, "\n", files)
                    nii = nip.resample_from_to(nib.Nifti1Image(arr, new_aff), inital, order=3, cval=0)
                    arr = nii.get_fdata()
            else:
                arr1 = np.load(str(f))
                arr = arr1.copy().astype(np.float32)
                del arr1

            if str(key).upper() == "CT":
                arr /= 1000
                arr = arr.clip(-1, 1)
            else:
                print(f) if np.max(arr) == 0 else None
                arr = arr / np.max(arr)
                arr = arr.clip(0, 1e6)

            out.append(arr)
            files.append(f)

        return out

    @torch.no_grad()
    def transform(self, items_in):
        # Transform to tensor
        items = map(Tensor, items_in)

        ## Padding
        items = list(map(lambda x: pad_size(x, self.target_shape, self.padding), items))
        # Coordinate-encoding
        shape = items[0].shape
        l1 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
        l2 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
        l3 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
        l1 = Tensor(l1).permute(2, 0, 1)
        l2 = Tensor(l2).permute(0, 2, 1)
        l3 = Tensor(l3)
        assert l1.shape == l2.shape, (l1.shape, l2.shape)
        assert l3.shape == l2.shape, (l3.shape, l2.shape)
        assert shape == l2.shape, (shape, l2.shape)
        items.append(l1)
        items.append(l2)
        items.append(l3)
        ## Random crop
        items = list(random_crop(self.target_shape, *items))

        for i, (x, y) in enumerate(zip(items, self.condition_types)):
            if y in [
                "MRI",
                "T1",
                "t1",
                "T2",
                "t2",
                "T1GD",
                "FLAIR",
                "water",
                "fat",
                "T1w",
                "T2w",
            ]:
                items[i] = self.mri_transform(x)

            if y.upper() != "CT":
                items[i] = items[i] * 2 - 1

        # Random flipping
        if self.flip and random.random() > 0.5:
            self.spacial_flip.update()
            items = list(map(self.spacial_flip, items))
        out = tuple(a.to(torch.float32).unsqueeze_(0) for a in items)
        try:
            if self.output_channels == 1:
                return out[0], torch.cat(out[1:], 0)
            else:
                return torch.cat(out[: self.output_channels], 0), torch.cat(out[self.output_channels :], 0)
        except:
            print([o.shape for o in out])
            print([o.shape for o in out])
            exit()

    def __getitem__(self, index):
        try:
            t = time.time()
            list_of_items = self.load_file(index, self.condition_types)
            t2 = time.time()
            # for i in list_of_items:
            #    print(i.min(), i.max())
            a = self.transform(list_of_items)
            self.t_max = max(self.t_max, t2 - t)
            if index % 100 == 0:
                print(self.t_max)

            return a
        except Exception as e:
            print(index, str(e))
            print(self.pd.iloc[index])
            return self.__getitem__(index + 1)
            # raise e

    def __len__(self):
        return len(self.pd)
