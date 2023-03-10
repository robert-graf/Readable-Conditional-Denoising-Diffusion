from __future__ import annotations
import glob
import json
import random
import os
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch import Tensor
import warnings

# class ImageDataset(Dataset):
#    def __init__(
#        self, root="D:/data_public/datasets/monet2photo", root2=None, transforms_=None, unaligned=False, mode="train"
#    ):
#        self.transform = transforms.Compose(transforms_)
#        self.unaligned = unaligned
#        # if root2 == None:
#        #    self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
#        #    self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
#        # else:
#        self.files_A = sorted(glob.glob(root + "/*.*"))
#        if root2 != None and unaligned:
#            self.files_B = sorted(glob.glob(root2 + "/*.*"))
#        else:
#            root_B = root.replace("A", "B").replace("ct", "m$r$i$").replace("mri", "ct").replace("m$r$i$", "mri")
#            if not unaligned:
#                self.files_B = sorted(glob.glob(root2 + "/*.*"))
#            else:
#                count = 0
#                self.files_B = []
#                for f in self.files_A:
#                    name = f.replace("A", "B").replace("ct", "m$r$i$").replace("mri", "ct").replace("m$r$i$", "mri")
#                    if os.path.isfile(name):
#                        self.files_B.append(name)
#                    else:
#                        self.files_A.remove(f)
#                        count += 1
#                print("Non-paired images", count, "of", len(self.files_A))
#
#            print("Autogenerated root_B:", root_B)
#
#        assert len(self.files_A) > 0, f"No file found in {root},{len(self.files_A)}"
#        assert len(self.files_B) > 0, f"No file found in {root2},{len(self.files_B)}"
#
#    def __getitem__(self, index):
#        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
#
#        if self.unaligned:
#            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
#        else:
#            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
#            # name = self.files_A[index % len(self.files_A)].replace('A','B').replace('ct','m$r$i$').replace('mri','ct').replace('m$r$i$','mri')
#            # if not os.path.isfile(name):
#            #    del self.files_A[index]
#            #    print(f"Removed missing paired file: {name}")
#            #    return self.__getitem__(index-1)
#            #
#            # item_B = self.transform(Image.open(name))
#            # item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
#
#        return item_A, item_B
#
#    def __len__(self):
#        return max(len(self.files_A), len(self.files_B))


class DummyDataset(Dataset):
    def __init__(self, num_sample=25, ds=None, timesteps=1):
        self.num_sample = num_sample
        self.ds = ds

    def __getitem__(self, index):
        if self.ds is None:
            return torch.zeros(1), torch.zeros(1)

        return self.ds[random.randint(0, len(self.ds) - 1)]

    def __len__(self):
        return self.num_sample


from pathlib import Path


class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder,
        size=128,
        exts=["jpg", "jpeg", "png"],
        train: bool | None = True,
        _transform=None,
        compute_mean=True,
        flip=True,
        set_size="resize",
        paths: list[Path] | None = None,
    ) -> None:
        super().__init__()
        self.folder = folder
        self.train = train
        if train is None:
            pass
        elif train:
            folder = os.path.join(folder, "train")
        else:
            folder = os.path.join(folder, "val")
        self.image_size = size
        if paths is None:
            self.paths = [p for ext in exts for p in Path(folder).glob(f"**/*.{ext}")]
            if len(self.paths):
                # print("Dataset: search for subfolder")
                word = "train" if train else "val"
                self.paths = [p for ext in exts for p in Path(folder).glob(f"**/*.{ext}") if word in str(p)]
            self.paths += [p for ext in exts for p in Path(folder).glob(f"*.{ext}")]
        else:
            self.paths = paths
        self.set_size = set_size
        if compute_mean:
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.compute_mean_and_std()
        elif _transform is None:
            self.normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

        if _transform:
            self.transform = transforms.Compose([_transform, transforms.ToTensor()])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size) if set_size == "resize" else transforms.RandomCrop(size),
                    transforms.RandomHorizontalFlip() if flip else torch.nn.Identity(),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.normalize),
                ]
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")

        return self.transform(img)

    def compute_mean_and_std(self):
        import numpy as np
        from tqdm import tqdm

        loader = torch.utils.data.DataLoader(self, batch_size=32, num_workers=0, shuffle=False)
        mean = 0.0
        mean_sq = 0.0
        count = 0

        for _, data in tqdm(enumerate(loader), total=len(loader)):
            data: torch.Tensor
            mean += data.sum([0, 2, 3])
            mean_sq += (data**2).sum([0, 2, 3])
            count += np.prod(data.shape) / data.shape[1]
        assert count != 0, f"No dataset in {self.folder}"
        total_mean = mean / count
        total_var: torch.Tensor = (mean_sq / count) - (total_mean**2)  # type: ignore
        total_std = torch.sqrt(total_var)
        self.normalize = {"mean": total_mean.tolist(), "std": total_std.tolist()}
        return self.normalize


from dataloader.aligned_dataset import get_params, get_transform


class AlignedDataset2(Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/val'.
    """

    def __init__(
        self, root, train, size, compute_mean=True, flip=False, exts=["jpg", "jpeg", "png"], set_size="crop", color=True
    ):
        """Initialize this dataset class."""
        self.train = train
        self.paired = True
        self.compute_mean = compute_mean
        if train:
            self.folderA = os.path.join(root, "A", "train")
            self.folderB = os.path.join(root, "B", "train")
            if not os.path.exists(self.folderA):
                self.paired = False
                self.folderA = os.path.join(root, "trainA")
                self.folderB = os.path.join(root, "trainB")
                assert os.path.exists(
                    self.folderA
                ), "Could not find train_folder. Change the config so you dont load the aligned dataset or chang the data set to /A/train/*.png and /B/train/*.png or trainA and trainB"
                print("Unpaired mode")
        else:
            self.folderA = os.path.join(root, "A", "val")
            self.folderB = os.path.join(root, "B", "val")
            if not os.path.exists(self.folderA):
                self.paired = False
                self.folderA = os.path.join(root, "valA")
                self.folderB = os.path.join(root, "valB")
                if not os.path.exists(self.folderA):
                    self.folderA = os.path.join(root, "trainA")
                    self.folderB = os.path.join(root, "trainB")
                assert os.path.exists(self.folderA), "Could not find val_folder"
                print("Unpaired mode")

        self.A_paths = []
        self.B_paths = []

        for ext in exts:
            self.A_paths += sorted(glob.glob(self.folderA + f"/*.{ext}"))
            self.B_paths += sorted(glob.glob(self.folderB + f"/*.{ext}"))
        self.input_nc = 3 if color else 1
        self.output_nc = 3 if color else 1
        self.flip = flip
        self.size = size
        self.set_size = set_size
        if compute_mean:
            f = os.path.join(root, f"mean_and_std_{'b2a' if flip else 'a2b'}.json")
            if os.path.exists(f):
                with open(f, "r") as f:
                    self.normalize = json.load(f)
                # Example shape:
                # self.normalize = {
                #    "target": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                #    "conditional": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                # }
            else:
                from dataloader.dataloader_basic import SingleImageDataset

                print("compute mean and std of target")
                i = torch.nn.Identity()
                folder = self.folderA if flip else self.folderB
                a = SingleImageDataset(folder, -1, train=True, compute_mean=False, _transform=i)
                target_mean_std = a.compute_mean_and_std()
                print(target_mean_std)
                print("compute mean and std of condition")
                folder = self.folderB if flip else self.folderA
                a = SingleImageDataset(folder, -1, train=True, compute_mean=False, _transform=i)
                condition_mean_std = a.compute_mean_and_std()
                print(condition_mean_std)
                self.normalize = {"target": target_mean_std, "conditional": condition_mean_std}
                with open(f, "w") as f:
                    json.dump(self.normalize, f)

    def __getitem__(self, index) -> tuple[Tensor, Tensor, str, str]:
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a tuple that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
        # read a image given a random integer index
        A_path: str = self.A_paths[index]
        if self.paired:
            B_path: str = self.B_paths[index]
        else:
            B_path: str = self.B_paths[random.randint(0, len(self.B_paths) - 1)]

        A = Image.open(A_path).convert("RGB")  # type: ignore
        B = Image.open(B_path).convert("RGB")  # type: ignore
        # apply the same transform to both A and B
        transform_params = get_params(self, A.size)

        if self.flip:
            t = B
            B = A
            A = t
            t = A_path
            A_path = B_path
            B_path = t
        target_transform = get_transform(
            self, transform_params, grayscale=(self.input_nc == 1), mean_key="target", set_size=self.set_size
        )
        cond_transform = get_transform(
            self, transform_params, grayscale=(self.output_nc == 1), mean_key="conditional", set_size=self.set_size
        )

        A: Tensor = target_transform(A)
        B: Tensor = cond_transform(B)
        return A, B, A_path, B_path

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.A_paths)


class LabeledDataset(Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/val'.
    """

    def __init__(
        self,
        root,
        train,
        size,
        compute_mean=True,
        flip=False,
        set_size="crop",
        exts=["jpg", "jpeg", "png"],
    ) -> None:
        """Initialize this dataset class."""
        self.train = train
        self.ab_mode = False
        sub_folder = ""
        self.compute_mean = compute_mean
        if flip:
            warnings.warn(
                "--flip inverts prediction direction A->B to B->A; LabeledDataset does not support this, because it only has one image."
            )
        if train:
            self.folderA = os.path.join(root, "train")
            if not os.path.exists(self.folderA):
                if not os.path.exists(os.path.join(root, "A")):
                    self.folderA = os.path.join(root, "trainA")
                    assert os.path.exists(self.folderA), "Could not find train_folder"
                    self.ab_mode = True
                    self.folderA = os.path.join(root)
                    print("AB mode")
                else:
                    self.ab_mode = True
                    sub_folder = "/train"
                    print("AB2 mode")
        else:
            self.folderA = os.path.join(root, "val")
            if not os.path.exists(self.folderA):
                if not os.path.exists(os.path.join(root, "A")):
                    self.folderA = os.path.join(root, "trainA")
                    assert os.path.exists(self.folderA), "Could not find train_folder"
                    self.ab_mode = True
                    self.folderA = os.path.join(root)
                    print("AB mode")
                else:
                    self.ab_mode = True
                    sub_folder = "/val"
                    print("AB2 mode")

        self.A_paths = []
        self.ids = {}
        classes_folder = []
        if not self.ab_mode:
            classes_folder = sorted(Path(self.folderA).iterdir())
        else:
            if not train:
                classes_folder = [i for i in sorted(Path(self.folderA).iterdir()) if i.name.startswith("val")]
            if len(classes_folder) == 0:
                classes_folder = [
                    i for i in sorted(Path(self.folderA).iterdir()) if i.name.startswith("train") or sub_folder != ""
                ]

        for class_name in classes_folder:
            if not class_name.is_dir():
                continue
            key = class_name.name
            self.ids[key] = len(self.ids)
            paths = []
            for ext in exts:
                paths += sorted(glob.glob(str(class_name) + sub_folder + f"/*.{ext}"))
            self.A_paths += [(i, self.ids[key]) for i in paths]
        self.input_nc = 3 if compute_mean else 1
        self.output_nc = 3 if compute_mean else 1
        self.size = size
        self.set_size = set_size
        if compute_mean:
            f = os.path.join(root, f"mean_and_std_cls.json")
            if os.path.exists(f):
                with open(f, "r") as f:
                    self.normalize = json.load(f)
                # Example shape:
                # self.normalize = {
                #    "target": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                #    "conditional": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                # }
            else:
                from dataloader.dataloader_basic import SingleImageDataset

                print("compute mean and std of target")
                i = torch.nn.Identity()
                a = SingleImageDataset(
                    self.folderA, -1, train=None, compute_mean=False, _transform=i, paths=[i[0] for i in self.A_paths]
                )
                target_mean_std = a.compute_mean_and_std()
                print(target_mean_std)
                self.normalize = {"target": target_mean_std}
                with open(f, "w") as f:
                    json.dump(self.normalize, f)

    def __getitem__(self, index):
        # read a image given a random integer index
        A_path, i = self.A_paths[index]
        A = Image.open(A_path).convert("RGB")
        # apply the same transform to both A and B
        transform_params = get_params(self, A.size)
        target_transform = get_transform(
            self, transform_params, grayscale=(self.input_nc == 1), mean_key="target", set_size=self.set_size
        )

        A = target_transform(A)
        return A, i

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
