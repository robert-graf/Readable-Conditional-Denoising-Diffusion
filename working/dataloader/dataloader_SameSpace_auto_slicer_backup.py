import math
import sys
from pathlib import Path
import zipfile

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
import nibabel.processing as nip
import nibabel.orientations as nio
import math

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
    for in_size, out_size in zip(reversed(x.shape[-2:]), reversed(target_shape)):
        to_pad_size = max(0, out_size - in_size) / 2.0
        padding.extend([ceil(to_pad_size), floor(to_pad_size)])
    # print(x.shape, target_shape)
    x_ = (
        F.pad(x.unsqueeze(0).unsqueeze(0), padding, mode=mode).squeeze(0).squeeze(0)
    )  # mode – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
    # print("Padding", x_.shape, x.shape, padding)
    return x_


def random_crop(target_shape: tuple[int, int], *arrs: Tensor):
    sli = [slice(None), slice(None)]
    for i in range(2):
        z = max(0, arrs[0].shape[-i] - target_shape[-i])
        if z != 0:
            r = random.randint(0, z)
            r2 = r + target_shape[-i]
            sli[-i] = slice(r, r2 if r2 != arrs[0].shape[-i] else None)

    return tuple(a[..., sli[0], sli[1]] for a in arrs)


class SameSpace_2D_Dataset(Dataset):
    def __init__(
        self,
        buffer_slices: str,
        root: str,
        condition_types,
        target_shape,
        flip: bool = False,
        padding="reflect",  # constant, edge, reflect or symmetric
        mri_transform=transforms.Compose([ColorJitter3D(brightness_min_max=(0.8, 1.2), contrast_min_max=(0.8, 1.2))]),
        train=True,
        output_channels=1,
    ):
        assert Path(buffer_slices).parent.exists(), buffer_slices
        # assert ";" not in str(root), "Not yet supported"
        self.slice_buffer = Path(buffer_slices, ("train" if train else "val"))
        self.slice_buffer.mkdir(exist_ok=True, parents=True)
        # TODO make a train.csv with all patients
        # "spinegan0002/target-301_from-201"
        self.condition_types = [x for x in condition_types if x != "linespace"]
        self.output_channels = output_channels
        self.buffer_root = None
        self.root = Path(root)
        if ";" in str(root):
            self.df = pandas.concat([pandas.read_excel(r) for r in str(root).split(";")])
        elif self.root.is_file():
            self.buffer_root = str(self.root).rsplit(".")[0] + ("_train" if train else "_val") + "_buffer.xlsx"
            if self.root.name.endswith(".csv"):
                self.df = pandas.read_csv(root)
            else:
                self.df = pandas.read_excel(root)
            self.root = self.root.parent
            if "resampled" in self.root.name:
                self.root = self.root.parent
        else:
            self.df = pandas.read_csv(Path(self.root, "train.csv"))

        self.df = self.df[self.df["Phase"] == ("train" if train else "val")]
        self.df["num_slices"] = pandas.Series(dtype="int")
        assert len(self.df) != 0, self.df
        self.flip = flip
        self.padding = padding
        self.mri_transform = mri_transform
        self.train = train
        self.spacial_flip = SpatialFlip(dims=(2, 3), auto_update=False)
        self.target_shape = target_shape
        self.count = 10
        s = 0
        if self.buffer_root is not None and Path(self.buffer_root).exists():
            self.df = pandas.read_excel(self.buffer_root)
        else:
            for id1 in range(len(self.df)):
                l = list(self.slice_buffer.glob(f"{id1:09}_*.npz"))
                # print("files", l)
                if len(l) == 0:
                    continue
                else:
                    self.df.iloc[id1, self.df.columns.get_loc("num_slices")] = len(l)
                s += len(l)
                print("Find crops", s, end="\r")
            if self.buffer_root is not None:
                self.df.to_excel(self.buffer_root)

        self.lens_dict = {}
        for i, (r, row) in enumerate(self.df.iterrows()):
            if pandas.isna(row.num_slices):
                continue
            self.lens_dict[i] = row.num_slices

        self.blacklist = []
        self._len = None
        self._len_df = len(self.df)

    def load_3D_file(self, id, keys=["CT", "T2w"]):
        self.count -= 1

        from BIDS import NII

        id1 = id % self._len_df
        out = []
        files = []
        for key in keys:
            # f = #self.root /
            assert isinstance(self.df.iloc[id1][key], str) or not math.isnan(self.df.iloc[id1][key]), "detected empty entry\n" + str(
                self.df.iloc[id1]
            )
            p = self.df.iloc[id1]["Path"]
            if not str(p).startswith("/"):
                p = Path(self.root, p)
            f = Path(p, self.df.iloc[id1][key])
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
                return None
                # assert False, f"{f} does not exit"
            # print(f)
            nii: nib.Nifti1Image = nib.load(str(f))  # type: ignore
            aff = nii.affine
            ornt_fr = nio.io_orientation(aff)
            try:
                arr: np.ndarray = nii.get_fdata()
            except EOFError:
                print("EOF-ERROR", f)
                Path(f).unlink()
                return None  # self.load_file(id + 1, keys=keys)
            except Exception as e:
                print(f)
                print(f)
                print(f)
                print(e)
                print(f)
                print(f)
                print(f)
                return None  # self.load_file(id + 1, keys=keys)
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

    def load_file_npz(self, name, keys):
        dict_mods = []
        if str(name).endswith(".npz"):
            try:
                f = np.load(name)
            except zipfile.BadZipFile:
                print(name)
            for k in keys:  # type: ignore
                dict_mods.append(f[k].astype("f").copy())

            f.close()  # type: ignore
            return dict_mods
        assert False, "Expected a .npz file"

    def load_file(self, id, keys=["CT", "T2w"]):
        slice_buffer = self.slice_buffer

        # Test if num of slices in know
        # print(self.df.iloc[id1].num_slices, pandas.isna(self.df.iloc[id1].num_slices))
        if id not in self.lens_dict:  # pandas.isna(self.df.iloc[id].num_slices):
            l = list(slice_buffer.glob(f"{id:09}_*.npz"))
            # print("files", l)
            if len(l) == 0:
                print("load_3D_file and save as 2D", "  ", end="\r")
                arrs = self.load_3D_file(id, keys)
                if arrs is None:
                    self.__getitem__(random.randint(0, len(self)))
                    return
                num_slices = arrs[0].shape[0]
                # self.df.iloc[id, self.df.columns.get_loc("num_slices")] = num_slices
                self.lens_dict[id] = num_slices
                # slice and save
                for i in range(num_slices):
                    print("load_3D_file and save as 2D", i, "/", num_slices, "  ", end="\r")
                    d = {}
                    for key, a in zip(keys, arrs):
                        d[key] = a[i]
                    np.savez_compressed(Path(slice_buffer, f"{id:09}_{i:03}.npz"), **d)
                print("load_3D_file and save as 2D - END  ", end="\r")
            else:
                # self.df.iloc[id, self.df.columns.get_loc("num_slices")] = len(l)
                self.lens_dict[id] = len(l)
            self._len = None
        # num_slices = int(self.df.iloc[id].num_slices)
        num_slices = self.lens_dict[id]
        selected_slice = random.randint(0, num_slices - 1)
        return self.load_file_npz(Path(slice_buffer, f"{id:09}_{selected_slice:03}.npz"), keys)

    @torch.no_grad()
    def transform(self, items_in):
        # Transform to tensor
        items = map(Tensor, items_in)

        ## Padding
        items = list(map(lambda x: pad_size(x, self.target_shape[-2:], self.padding), items))
        # Coordinate-encoding
        # shape = items[0].shape
        # l1 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
        # l2 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
        # l3 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
        # l1 = Tensor(l1).permute(2, 0, 1)
        # l2 = Tensor(l2).permute(0, 2, 1)
        # l3 = Tensor(l3)
        # assert l1.shape == l2.shape, (l1.shape, l2.shape)
        # assert l3.shape == l2.shape, (l3.shape, l2.shape)
        # assert shape == l2.shape, (shape, l2.shape)
        # items.append(l1)
        # items.append(l2)
        # items.append(l3)
        ## Random crop
        items = list(random_crop(self.target_shape[-2:], *items))

        for i, (x, y) in enumerate(zip(items, self.condition_types)):
            if y in ["MRI", "T1", "t1", "T2", "t2", "T1GD", "FLAIR", "water", "fat", "T1w", "T2w"]:
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
        index = index % self._len_df
        if index in self.blacklist:
            return self.__getitem__(random.randint(0, len(self)))
        try:
            list_of_items = self.load_file(index, self.condition_types)
            if list_of_items is None:
                self.blacklist.append(index)
                return self.__getitem__(random.randint(0, len(self)))
            # for i in list_of_items:
            #    print(i.min(), i.max())
            for i in list_of_items:
                if len(i.shape) == 1:
                    print(index, "wrong shape", [i.shape for i in list_of_items])
                    return self.__getitem__(random.randint(0, len(self)))
            return self.transform(list_of_items)

        except Exception as e:
            if index in self.blacklist:
                return self.__getitem__(random.randint(0, len(self)))
            else:
                print(index, str(e))
                print("ERRRRROR", self.df.iloc[index])
                # return self.__getitem__(index + 1)
            raise e

    def __len__(self):
        if self._len is None:
            self._len = max(self._len_df, int(sum(self.lens_dict.values())))
        return self._len
