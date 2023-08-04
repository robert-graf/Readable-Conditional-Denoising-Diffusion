from __future__ import annotations
from typing import Protocol, runtime_checkable
from typing_extensions import TypeGuard
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import random
import numpy as np


class Diffusion_Dataset(ABC, Dataset):
    # Returns a dict with target and at least on of 'condition', 'label', 'embedding'
    @abstractmethod
    def __getitem__(self, index) -> dict[str, np.ndarray]:
        ...

    @abstractmethod
    def has_condition(self) -> bool:
        ...

    @abstractmethod
    def has_label(self) -> bool:
        ...

    @abstractmethod
    def has_embedding(self) -> bool:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


@runtime_checkable
class Labeled_Dataset(Protocol):
    num_classes: int


keys_dropout = ["condition", "embedding"]
inpainting_choice = ["random_ege", "condition_perlin"]
from typing import Protocol, TypedDict, Union
from torch import Tensor

Batch_Dict = TypedDict(
    "Batch_Dict",
    target=Tensor,
    condition=Union[Tensor, None],
    embedding=Union[Tensor, None],
    label=Union[Tensor, None],
    mask=Union[Tensor, None],
    mask_cond=Union[Tensor, None],
)


class Wrapper_Dataset(Diffusion_Dataset):
    def __init__(self, data_set, image_dropout=0, size=0, inpainting: str | None = None, compute_mean=True):
        self.data_set = data_set
        self.image_dropout = image_dropout
        self.train = data_set.train
        self.inpainting = inpainting
        self.size = size

        if compute_mean and hasattr(data_set, "normalize"):
            self.normalize = data_set.normalize
        else:
            self.normalize = {"mean": [0.5], "std": [0.5]}

    # Returns a dict with 'target' and optionally 'condition', 'label', 'embedding'
    @abstractmethod
    def getitem(self, index) -> Batch_Dict:
        ...

    def __getitem__(self, index) -> Batch_Dict:
        # To use "classifier free guidance" we have to partially train on unconditional images.
        # --image_dropout = 0.5 forces 50 % to be unconditional
        if self.image_dropout > 0 and self.train and self.image_dropout > random.random():
            dic = self.getitem(index)
            for l in keys_dropout:
                if l in dic:
                    dic[l] = dic[l] * 0
            if "label" in dic:
                # replace real class with the "any class"-label
                dic["label"] = self.data_set.num_classes - 1
        else:
            dic = self.getitem(index)
        if self.inpainting is None:
            return dic
        elif self.inpainting == "random_ege":
            # print('inpainting - random_ege')
            # 40% chance to do nothing
            side = int(random.random() * 6)  # [0,5]
            mask: Tensor = np.zeros_like(dic["target"])  # type: ignore
            assert self.size != 0
            mask_height = int((random.random() * 0.4 + 0.1) * self.size) + 1
            if side == 0:
                mask[..., :mask_height] = 1
            elif side == 1:
                mask[..., -mask_height:] = 1
            elif side == 2:
                mask[..., :mask_height, :] = 1
            elif side == 3:
                mask[..., -mask_height:, :] = 1
            dic["mask"] = mask
            return dic
        elif self.inpainting == "condition_perlin":
            assert self.size != 0
            import utils.perlin as perlin

            if random.random() > 0.5:
                dic["mask_cond"] = (
                    perlin.rand_perlin_2d_mask((self.size, self.size), random.choice([2, 4, 8, 16]), (0.4, 0.7)).unsqueeze(0).numpy()
                )
            else:
                dic["mask_cond"] = np.ones_like(dic["target"])  # type: ignore
            return dic
        else:
            assert False, "Inpainting does not exits:" + self.inpainting

    def has_condition(self) -> bool:
        return False

    def has_label(self) -> bool:
        return isinstance(self, Labeled_Dataset)

    def get_label_count(self) -> int:
        assert isinstance(self, Labeled_Dataset)
        return self.num_classes

    def get_embedding_size(self):
        raise NotImplementedError()

    def has_embedding(self) -> bool:
        return False

    def get_conditional_channel_size(self) -> int:
        if hasattr(self.data_set, "get_conditional_channel_size"):
            return self.data_set.get_conditional_channel_size()
        # -1 same as input
        return -1

    def __len__(self) -> int:
        return len(self.data_set)


class Wrapper_Image2Image(Wrapper_Dataset):
    def __init__(self, data_set, image_dropout, **args) -> None:
        super().__init__(data_set, image_dropout, **args)

    def getitem(self, index):
        target, condition, *_ = self.data_set[index]
        return {"target": target, "condition": condition}

    def has_condition(self):
        return True


class Wrapper_Label2Image(Wrapper_Dataset):
    def __init__(self, data_set, num_classes, image_dropout=0, is_image_conditional=False, **args):
        super().__init__(data_set, **args)
        self.num_classes = num_classes
        if image_dropout != 0:
            # add dummy class that is any class
            self.num_classes += 1
        self.is_image_conditional = is_image_conditional
        assert self.has_label()

    def getitem(self, index):
        if self.is_image_conditional:
            target, condition, label, *_ = self.data_set[index]
            return {"target": target, "label": label, "condition": condition}

        target, label, *_ = self.data_set[index]
        return {"target": target, "label": label}

    def has_condition(self) -> bool:
        return self.is_image_conditional


class Wrapper_Embedding2Image(Wrapper_Dataset):
    def __init__(self, data_set, embedding_size, **args) -> None:
        self.embedding_size = embedding_size
        super().__init__(data_set, **args)

    def getitem(self, index):
        target, embedding, *_ = self.data_set[index]
        return {"target": target, "embedding": embedding}

    def has_embedding(self) -> bool:
        return True

    def get_embedding_size(self):
        return self.embedding_size


class Wrapper_Unconditional(Wrapper_Dataset):
    def __init__(self, data_set, **args):
        super().__init__(data_set, **args)

    def getitem(self, index):
        target, *_ = self.data_set[index]
        return {"target": target}

    def get_conditional_channel_size(self):
        return 0
