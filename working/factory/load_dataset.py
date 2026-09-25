from __future__ import annotations
import os
from pathlib import Path
import typing
from torchvision.datasets import MNIST
import argparse
from torch.utils.data import Dataset
from typing import Any, Optional, Tuple
from dataloader.Wrapper_datasets import *
from factory.arguments import Train_Option
from utils.utils_diffusion import get_option
from dataloader.dataloader_ae import Universal_dataset

# In this folders the software looks for datasets
paths_searched = ["datasets", "../datasets", "../../datasets", "../../../datasets"]

# remap names to folders, when they don't match
map_folder_name = {
    "wopathfx": "2022_06_21_T1_CT_wopathfx/dataset/traindata/",
    "tgd": "TGD_2D_dash/",
    "vertebra": "wopathfx_3D_vert/",
    "vertebra_cls": "wopathfx_3D_vert/",
}
#

dataset_names_img2img = ["cityscapes", "night2day", "edges2handbags", "edges2shoes", "facades", "maps"]
dataset_names_label2img = ["MNIST"]
dataset_names_costume = [
    "wopathfx",
    "tgd",
    "vertebra_cls",
    "vertebra",
    "bailiang",
    "gbm",
    "fx_T1w",
    "spinegan_T2w",
    "spinegan_T2w_1p_reg",
    "spinegan_T2w_no_reg",
    "nako_upscale_t1_t2",
    "nako_t2",
    "nako_upscale_t1_t2_inv",
    "spinegan_T2w_3D",
    "spinegan_T2w_epic",
]
dataset_names = dataset_names_costume + dataset_names_img2img + dataset_names_label2img
dataset_names_found = []

for p in paths_searched:
    p = Path(p)
    if p.exists():
        for f in p.iterdir():
            if f.is_dir():
                f_name = f.name
                if f_name in dataset_names:
                    continue
                if f_name + "/" in map_folder_name.values():
                    continue
                # class-label or pure generation
                if Path(f, "train").exists() and Path(f, "val").exists():
                    dataset_names_found.append(f_name)
                    dataset_names.append(f_name)
                    # print(f)
                # paired
                if Path(f, "A/train").exists() and Path(f, "B/train").exists():
                    dataset_names_found.append(f_name)
                    dataset_names.append(f_name)
                # unpaired
                if Path(f, "trainA").exists() and Path(f, "trainB").exists():
                    dataset_names_found.append(f_name)
                    dataset_names.append(f_name)
                    # print(f)
from torchvision import transforms


def get_transforms(size, tf, validation=False):
    from utils.utils_cut import SquarePad

    if tf == "crop":
        set_size = transforms.RandomCrop(size)
    elif tf == "resize":
        set_size = transforms.Resize(size)
    else:
        raise NotImplementedError(tf)
    #### Define transformation. ####
    if not validation:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                SquarePad(size),
                set_size,
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                SquarePad(size),
                set_size,
                transforms.Normalize((0.5), (0.5)),
            ]
        )


def get_potential_roots(dataset_folder, offset=0):
    if dataset_folder in map_folder_name:
        dataset_folder = map_folder_name[dataset_folder]
    for p in paths_searched:
        p = os.path.join(p, dataset_folder)
        if os.path.exists(p):
            root = p

            return root
    return os.path.join(paths_searched[offset], dataset_folder)


def getDataset(
    opt: Train_Option | None = None,
    train=False,
    dataset: str | None = None,  # type: ignore
    size: int = 256,  # type: ignore
    is_3D=False,  # type: ignore
    offset=0,
    inpainting=None,  # type: ignore
    set_size="crop",
    compute_mean=True,
    learning_type="unconditional",
    flip=False,
) -> Tuple[Wrapper_Dataset, int]:
    # Defaults
    image_dropout = 0

    condition_types: list[str] = ["T1"]

    if opt is not None:
        dataset: str = opt.dataset
        if not train and opt.dataset_val is not None:
            dataset = opt.dataset_val
        size: int = opt.size
        is_3D: bool = opt.volumes
        inpainting: str | None = get_option(opt, "inpainting", None)
        set_size = opt.transform
        # dataset_folder = dataset
        image_dropout = opt.image_dropout
        learning_type = opt.learning_type
        condition_types: list[str] = get_option(opt, "condition_types", None)
        if condition_types is None:
            condition_types: list[str] = get_option(opt, "condition_typs", ["T1"])
        flip = get_option(opt, "flip", False)
        if learning_type == "upscale":
            opt.__setattr__("conditional", True)
    scale_factor = opt.scale_factor if opt is not None else 4

    assert dataset is not None

    if dataset in map_folder_name:
        dataset_folder = map_folder_name[dataset]
    else:
        dataset_folder = dataset
    root = get_potential_roots(dataset_folder, offset=offset)
    print(f"Dataset: {root}")
    if is_3D:
        print("Using a 3D dataset")
        return getDataset3D(opt, train, dataset, root, size, image_dropout, condition_types, inpainting)
    general_wrapper_info = {
        "image_dropout": image_dropout,
        "size": size,
        "inpainting": inpainting,
        "compute_mean": compute_mean,
    }
    general_ds_info = {
        "train": train,
        "size": size,
        "compute_mean": compute_mean,
        "set_size": set_size,
        "flip": flip,
    }
    #### xlsx datasets ####
    general_wrapper_info = {"image_dropout": image_dropout, "size": size, "inpainting": inpainting}
    if dataset.endswith("xlsx"):
        from dataloader.dataloader_SameSpace_auto_slicer import SameSpace_2D_Dataset

        assert opt is not None
        assert opt.buffer_slices != ""
        assert isinstance(opt.buffer_slices, str)
        ds = SameSpace_2D_Dataset(
            opt.buffer_slices,
            root=dataset,
            condition_types=condition_types,
            train=train,
            target_shape=opt.target_shape3D,
            output_channels=opt.output_channels,
        )

        return Wrapper_Image2Image(ds, **general_wrapper_info), 1
    ###MNIST#####################################################################################################################
    if dataset == "MNIST":
        transforms_ = get_transforms(size, set_size, validation=train)
        ds = MNIST(root=root, transform=transforms_, download=True, train=train)

        if learning_type == "label":
            return (Wrapper_Label2Image(ds, num_classes=10, **general_wrapper_info), 1)
        elif learning_type is None or learning_type == "unconditional":
            return (Wrapper_Unconditional(ds, **general_wrapper_info), 1)
        elif learning_type == "upscale":
            return (Wrapper_Upscale(ds, factor=scale_factor, **general_wrapper_info), 1)

        else:
            assert False, learning_type + "is not supported"
    ###Image2Image#####################################################################################################################
    if dataset in dataset_names_img2img:
        from dataloader.aligned_dataset import AlignedDataset

        ds = AlignedDataset(root, **general_ds_info)
        if learning_type == "image":
            return (Wrapper_Image2Image(ds, **general_wrapper_info), 3)
        elif learning_type is None or learning_type == "unconditional":
            return (Wrapper_Unconditional(ds, **general_wrapper_info), 3)
        elif learning_type == "upscale":
            return (Wrapper_Upscale(ds, factor=scale_factor, **general_wrapper_info), 3)
        else:
            assert False, learning_type + "is not supported"
    ###Costume datasets######################################################################################################################
    if not dataset in dataset_names_costume:
        assert learning_type is not None, str(learning_type) + " is not supported"

    from dataloader.dataloader_basic import SingleImageDataset

    if dataset == "nako_upscale_t1_t2" or dataset == "nako_upscale_t1_t2_inv":
        from dataloader.dataloader_mri2ct import Wopathfx_Dataset

        assert opt is not None
        ds = Wopathfx_Dataset(
            size=(opt.size, opt.size_w),
            root=root,
            train=train,
            condition_types=condition_types,
            norm=True,
        )
        return Wrapper_Image2Image(ds, **general_wrapper_info), 1
    elif dataset == "nako_t2":
        pass
        from dataloader.dataloader_mri2ct import Wopathfx_Dataset

        assert opt is not None
        ds = Wopathfx_Dataset(
            size=opt.size,
            root="/DATA/robert/NAKO/T2w/",
            train=train,
            condition_types=condition_types,
            norm=True,
        )
        return Wrapper_Unconditional(ds, **general_wrapper_info), 1
    #
    elif dataset == "spinegan_T2w_epic":
        from dataloader.dataloader_mri2ct import Wopathfx_Dataset

        ds = Wopathfx_Dataset(
            size=size,
            root="/DATA/NAS/datasets_processed/spine/dataset-SpineGAN/spinegan_T2w_all_reg_iso",
            train=train,
            condition_types=condition_types,
        )
        return Wrapper_Image2Image(ds, **general_wrapper_info), 1

    elif dataset in dataset_names_costume:
        from dataloader.dataloader_mri2ct import Wopathfx_Dataset

        ds = Wopathfx_Dataset(size=size, root=root, train=train, condition_types=condition_types)
        return Wrapper_Image2Image(ds, **general_wrapper_info), 1
    ###Automatic datasets####################################################################################################################
    print(learning_type, train)
    if learning_type is None or learning_type == "unconditional" or learning_type == "upscale":
        if Path(root, "A").exists():
            if not flip:
                root = Path(root, "A")
            elif Path(root, "B").exists():
                root = Path(root, "B")
        if Path(root, "trainA").exists():
            if train:
                if not flip:
                    root = Path(root, "trainA")
                elif Path(root, "trainB").exists():
                    root = Path(root, "trainB")
                general_ds_info["train"] = None
            else:
                if not flip:
                    root = Path(root, "testA")
                elif Path(root, "testB").exists():
                    root = Path(root, "testB")
                general_ds_info["train"] = None
        root = str(root)
        print(root)

        ds = SingleImageDataset(root, **general_ds_info)
        if learning_type == "upscale":
            return (Wrapper_Upscale(ds, factor=scale_factor, **general_wrapper_info), 3)
        return Wrapper_Unconditional(ds, **general_wrapper_info), 3

    if learning_type is None or learning_type == "image":
        from dataloader.dataloader_basic import AlignedDataset2

        ds = AlignedDataset2(root, **general_ds_info)
        return Wrapper_Image2Image(ds, **general_wrapper_info), 3
    if learning_type == "label":
        from dataloader.dataloader_basic import LabeledDataset

        ds = LabeledDataset(root, **general_ds_info)
        return Wrapper_Label2Image(ds, num_classes=len(ds.ids), **general_wrapper_info), 3
    if learning_type == "label_image":
        assert False, "not implemented"
    assert False, "not implemented"


######## 3D ######


def getDataset3D(
    opt: Train_Option, train, dataset: str, root, size, image_dropout, condition_types, inpainting
) -> Tuple[Wrapper_Dataset, int]:
    general_wrapper_info = {"image_dropout": image_dropout, "size": size, "inpainting": inpainting}
    if dataset == "spinegan_T2w_3D":
        from dataloader.dataloader_mri2ct_3D_reg import MRI2CT_3D_Dataset

        assert opt is not None
        ds = MRI2CT_3D_Dataset(root="/media/data/robert/datasets/spinegan_T2w_all_reg_iso/", train=train, target_shape=opt.target_shape3D)
        return Wrapper_Image2Image(ds, **general_wrapper_info), 1

    from dataloader.dataloader_3D import SameSpace_3D_Dataset

    ds = SameSpace_3D_Dataset(
        root=dataset, condition_types=condition_types, train=train, target_shape=opt.target_shape3D, output_channels=opt.output_channels
    )

    return Wrapper_Image2Image(ds, **general_wrapper_info), 1
