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
import nibabel.processing as nip
import nibabel.orientations as nio


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
        data_frame: pandas.DataFrame,
        target_shape=[128, 128, 128],
        keys_in=[],
        keys_out=[],
        flip: bool = False,
        padding="reflect",  # constant, edge, reflect or symmetric
        mri_transform=transforms.Compose([ColorJitter3D(brightness_min_max=(0.8, 1.2), contrast_min_max=(0.8, 1.2))]),
        train=True,
        chaos_mode=False,
        opt=None,
    ):
        """

        Args:
            data_frame (pandas.DataFrame): contains "Path","Phase" and other columns.
                                           Phase is either "train", "val", "test"
                                           a row indicates pairs of files. Other columns are sub paths (starting from "Path") pointing to fils (nii.gz)
            target_shape (_type_): x,y,z of the batches
            key: list[str]|None: names of columns that should be used
            flip (bool, optional): _description_. Defaults to False.
            padding (str, optional): _description_. Defaults to "reflect".
            mri_transform (_type_, optional): _description_. Defaults to transforms.Compose([ColorJitter3D(brightness_min_max=(0.8, 1.2), contrast_min_max=(0.8, 1.2))]).
            train (bool, optional): _description_. Defaults to True.
        """
        if "Phase" in data_frame:
            self.pd = data_frame[data_frame["Phase"] == ("train" if train else "val")]
        else:
            print("use all data from dataframe (no row 'Phase' with train/val split)")
            self.pd = data_frame
        if keys_in is None or len(keys_in) == 0:
            chaos_mode = True
            keys_in = list(self.pd.columns.values)
            print(keys_in)
            keys_in = [a for a in keys_in if a not in ["Path", "Phase", "Unnamed: 0", "Unnamed: 1", "Unnamed: 2"]]
            opt.input_rows = keys_in
        if keys_out is None or len(keys_out) == 0:
            if chaos_mode:
                keys_out = keys_in
            else:
                keys_out = [a for a in keys_in if a not in ["Path", "Phase", "Unnamed: 0", "Unnamed: 1", "Unnamed: 2", *keys_in]]
                opt.output_rows = keys_out
        if train:
            print("Using the following keys as input:", keys_in)
            print("Using the following keys as output:", keys_out)
            assert len(keys_out) == 1 or chaos_mode, f"{keys_out}; only one input is currently possible"
            if chaos_mode:
                print("output is chosen randomly")
        self.keys_in = keys_in
        self.keys_out = keys_out
        self.flip = flip
        self.padding = padding
        self.mri_transform = mri_transform
        self.train = train
        self.spacial_flip = SpatialFlip(dims=(-1, -2), auto_update=False)
        self.chaos_mode = chaos_mode
        if target_shape == None:
            target_shape = (16, 128, 128)
        self.target_shape = target_shape
        self.file_keys = self.keys_out + self.keys_in

    def load_file(self, row, keys=["CT", "T2w"]):
        out = []
        for key in keys:
            # print(key, row[key], row.get("Path", ""))
            f = Path(row.get("Path", ""), row[key])

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

            nii: nib.Nifti1Image = nib.load(str(f))  # type: ignore
            aff = nii.affine
            ornt_fr = nio.io_orientation(aff)

            arr: np.ndarray = nii.get_fdata()
            ornt_to = nio.axcodes2ornt(("R", "P", "I"))
            ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)

            arr = nio.apply_orientation(arr, ornt_trans)
            aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
            new_aff = np.matmul(aff, aff_trans)
            if len(out) == 0:
                initial = nib.Nifti1Image(arr, new_aff)
            elif arr.shape != out[0].shape:
                print("resampeled", arr.shape, initial.shape, "You should reconsider resample it before training")
                nii = nip.resample_from_to(nib.Nifti1Image(arr, new_aff), initial, order=3, cval=0)
                arr = nii.get_fdata()
            if str(key).upper() == "CT":
                arr /= 1000
                arr = arr.clip(-1, 1)
            else:
                arr = arr / np.max(arr)
                arr = arr.clip(0, 1e6)
            # if self.count >= 0:
            #    print(f.name, arr.min(), arr.max())

            out.append(arr)

        return out

    @torch.no_grad()
    def transform(self, items_in, condition_types):
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
        assert len(items) == len(condition_types) + 3

        ## Random crop
        items = list(random_crop(self.target_shape, *items))

        for i, (x, y) in enumerate(zip(items, condition_types)):
            if y in ["MRI", "T1", "t1", "T2", "t2", "T1GD", "FLAIR", "water", "fat", "T1w", "T2w"]:
                items[i] = self.mri_transform(x)

            if y.upper() != "CT":
                items[i] = items[i] * 2 - 1

        # Random flipping
        if self.flip and random.random() > 0.5:
            self.spacial_flip.update()
            items = list(map(self.spacial_flip, items))
        out = list(a.to(torch.float32).unsqueeze_(0) for a in items)
        try:
            return out[0], torch.cat(out[1:], 0)
        except:
            print([o.shape for o in out])
            print([o.shape for o in out])
            exit()

    def __getitem__(self, index):
        try:
            if self.chaos_mode:
                id1 = index % len(self.pd)
                row: pandas.Series = self.pd.iloc[id1]
                keys = row.dropna().keys()
                keys = [a for a in keys if a in self.keys_in]
                file_keys = random.choices(keys, k=1)
                list_of_items = self.load_file(row, keys=file_keys)
                return self.transform(list_of_items, file_keys)
                # return *a  # , keys.index(file_keys[0])

            else:
                id1 = index % len(self.pd)
                row: pandas.Series = self.pd.iloc[id1]
                list_of_items = self.load_file(row, keys=self.file_keys)
                a, b = self.transform(list_of_items, self.file_keys)
                return a, b

        except EOFError:
            print("EOF-ERROR", self.pd.iloc[index])
            # Path(f).unlink()
            return self.__getitem__(index + 1)
        except RuntimeError as e:
            print("RuntimeError", self.pd.iloc[index])
            return self.__getitem__(2 * index + 1)
            raise e

    def __len__(self):
        return len(self.pd)
