import types
from dataclasses import asdict, dataclass, field
from enum import Enum
from inspect import signature
from typing import Type, get_args, get_origin

from configargparse import ArgumentParser


def translation_type(enum: Type[Enum]) -> list[str]:
    choice = []
    for v in enum:
        choice.append(v.name)
    return choice


@dataclass()
class Option_to_Dataclass:
    # C = Literal[tuple(range(100))]
    from configargparse import ArgumentParser

    @classmethod
    def get_opt(cls, parser: None | ArgumentParser = None, config=None):
        keys = []
        if parser is None:
            p: ArgumentParser = ArgumentParser()
            p.add_argument("-config", "--config", is_config_file=True, default=config, help="config file path")
        else:
            p = parser

        # fetch the constructor's signature
        parameters = signature(cls).parameters
        cls_fields = sorted({field for field in parameters})

        # split the kwargs into native ones and new ones
        def n(s):
            return str(s).replace("<class '", "").replace("'>", "")

        for name in cls_fields:
            key = "--" + name
            if key in keys:
                continue
            else:
                keys.append(key)
            default = parameters[name].default
            annotation = parameters[name].annotation
            if get_origin(annotation) == types.UnionType:
                for i in get_args(annotation):
                    if i == types.NoneType:
                        default = None
                    else:
                        annotation = i
            # print(type(annotation))
            if annotation == bool:
                if default:
                    p.add_argument(key, action="store_false", default=True)
                else:
                    p.add_argument(key, action="store_true", default=False)
            elif isinstance(default, Enum) or issubclass(annotation, Enum):
                p.add_argument(key, default=default, choices=translation_type(annotation))
            elif get_origin(annotation) == list or get_origin(annotation) == tuple:
                for i in get_args(annotation):
                    if i == types.NoneType:
                        default = None
                    else:
                        annotation = i
                p.add_argument(key, nargs="+", default=default, type=annotation, help="List of " + n(annotation))
            else:
                # print(annotation, key, default, annotation)
                p.add_argument(key, default=default, type=annotation, help=n(annotation))
        return p

    @classmethod
    def from_kwargs(cls, **kwargs):
        # fetch the constructor's signature
        parameters = signature(cls).parameters
        cls_fields = {field for field in parameters}
        # split the kwargs into native ones and new ones
        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in cls_fields:
                if isinstance(parameters[name].default, Enum):
                    try:
                        val = parameters[name].annotation[val]
                    except KeyError as e:
                        print(f"Enum {type(parameters[name].default)} has no {val}")
                        exit(1)
                native_args[name] = val
            else:
                new_args[name] = val
        ret = cls(**native_args)
        # ... and add the new ones by hand
        for new_name, new_val in new_args.items():
            setattr(ret, new_name, new_val)
        return ret


@dataclass
class Train_Option(Option_to_Dataclass):
    # Training
    # try in this order 0.002,0.0002,0.00002

    lr: float = 0.0002

    batch_size: int = 1
    max_epochs: int = 15
    num_cpu: int = 16
    exp_name: str = "NAME"
    size: int = 256
    size_w: int = -1
    transpose_preview: bool = False
    # Dataset
    dataset: str = "maps"
    flip: bool = True
    # Options: crop, resize
    transform: str = "crop"
    # Options: unconditional, image
    learning_type: str | None = None
    dataset_val: str | None = None
    model_name: str = "unet"

    image_dropout: float = 0
    scale_factor: int = 4  # Only used in upscaling
    num_validation_images: int = 8
    gpus: list[int] | None = None
    legacy_reload = False
    new: bool = False
    prevent_nan: bool = False

    def print(self) -> None:
        from pprint import pprint

        d = asdict(self)
        rest = {}
        lambda_list = {}
        net_D = {}
        net_G = {}
        training_keys = [
            "lr",
            "batch_size",
            "max_epochs",
            "decay_epoch",
            "start_epoch",
            "cpu",
            "gpus",
            "num_cpu",
            "new",
        ]
        training = {}
        dataset_keys = ["dataset", "dataset_val", "size", "flip", "transform", "learning_type", "condition_types"]
        dataset = {}
        for key, value in d.items():
            if "lambda" in key:
                lambda_list[key] = value
            elif "net_D" in key:
                net_D[key] = value
            elif "net_G" in key:
                net_G[key] = value
            elif key in training_keys:
                training[key] = value
            elif key in dataset_keys:
                dataset[key] = value
            else:
                rest[key] = value
        print(training)
        print(dataset)
        print(lambda_list) if len(lambda_list) != 0 else None
        print(net_G) if len(net_G) != 0 else None
        print(net_D) if len(net_D) != 0 else None
        pprint(rest, sort_dicts=False, width=200)
