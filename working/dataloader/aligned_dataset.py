# from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/aligned_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import glob

import torchvision.transforms as transforms
import random
import numpy as np
import os
from PIL import Image
from functools import partial
import json
import torch


def crop(AB, t=None):
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))
    if t is None:
        return A, B
    if t == "A":
        return A
    if t == "B":
        return B


class AlignedDataset(Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/val'.
    """

    def __init__(
        self,
        root,
        train=True,
        size=256,
        color=True,
        flip=False,
        compute_mean=True,
        set_size="crop",
        exts=["jpg", "jpeg", "png"],
    ):
        """Initialize this dataset class."""
        self.train = train
        if train:
            self.dir_AB = os.path.join(root, "train")  # get the image directory
        else:
            self.dir_AB = os.path.join(root, "val")  # get the image directory
        self.AB_paths = []
        for ext in exts:
            self.AB_paths += sorted(glob.glob(self.dir_AB + f"/*.{ext}"))
        self.input_nc = 3 if color else 1
        self.output_nc = 3 if color else 1
        self.flip = flip
        self.size = size
        self.compute_mean = compute_mean

        self.set_size = set_size
        self.convert_tensor = transforms.ToTensor()

        if color and compute_mean:
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
                a = SingleImageDataset(
                    root, -1, train=True, _transform=partial(crop, t="B" if flip else "A"), compute_mean=False
                )
                target_mean_std = a.compute_mean_and_std()
                print(target_mean_std)
                print("compute mean and std of condition")
                a = SingleImageDataset(
                    root, -1, train=True, _transform=partial(crop, t="A" if flip else "B"), compute_mean=False
                )
                condition_mean_std = a.compute_mean_and_std()
                print(condition_mean_std)
                self.normalize = {"target": target_mean_std, "conditional": condition_mean_std}
                with open(f, "w") as f:
                    json.dump(self.normalize, f)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        # split AB image into A and B
        A, B = crop(AB)
        # apply the same transform to both A and B
        transform_params = get_params(self, A.size)

        if self.flip:
            t = B
            B = A
            A = t

        target_transform = get_transform(
            self, transform_params, grayscale=(self.input_nc == 1), mean_key="target", set_size=self.set_size
        )
        cond_transform = get_transform(
            self, transform_params, grayscale=(self.output_nc == 1), mean_key="conditional", set_size=self.set_size
        )
        # if isinstance(A, Image.Image):
        #    A = self.convert_tensor(A)
        #    B = self.convert_tensor(B)
        A = target_transform(A)
        B = cond_transform(B)
        return A, B, AB_path

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


def get_params(self, size):
    w, h = size

    x = random.randint(0, np.maximum(0, w - self.size))
    y = random.randint(0, np.maximum(0, h - self.size))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip, "crop_size": w}


def get_transform(
    self, params=None, grayscale=False, convert=True, mean_key="target", set_size="crop"
):  # method=transforms.InterpolationMode.BICUBIC,
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    # if "resize" in opt.preprocess:
    #    o_size = [opt.load_size, opt.load_size]
    #    transform_list.append(transforms.Resize(o_size, method))
    # elif "scale_width" in opt.preprocess:
    #    transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    #
    # if "crop" in opt.preprocess:
    # if params is None:
    #   transform_list.append(transforms.RandomCrop(opt.crop_size))
    if set_size == "resize":
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, self.size)))
    elif set_size == "crop":
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params["crop_pos"], self.size)))

    # if opt.preprocess == "none":
    #    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    # if not opt.no_flip:
    #    if params is None:
    #        transform_list.append(transforms.RandomHorizontalFlip())
    #    elif params["flip"]:
    # transform_list.append(transforms.Lambda(lambda img: __flip(img, params["flip"])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale or not self.compute_mean:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            # The image net normalization is imported or the image goes from light to dark during training.
            transform_list += [transforms.Normalize(**self.normalize[mean_key])]
            # transform_list += [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, size, method=3):

    ow, oh = img.size
    if ow == size and oh >= size:
        return img
    w = size
    h = int(max(size * oh / ow, size))
    return img.resize((w, h), method)
    # return img.resize((size, size), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:

        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to "
            "(%d, %d). This adjustment will be done to all images "
            "whose sizes are not multiples of 4" % (ow, oh, w, h)
        )
        __print_size_warning.has_printed = True
