# from __future__ import annotations

import configargparse as argparse
from configargparse import ArgumentParser
from typing import List, Literal, Optional
from inspect import signature


##### DEFAULTS ####
def parseTrainParam(parser: ArgumentParser | None = None, config=None):  # type: ignore
    if parser is None:
        parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-config", "--config", is_config_file=True, default=config, help="config file path")

    parser.add_argument("-lr", "--lr", type=float, default=0.001, help="Learning rate of the network")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-epoch", "--max_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("-n_cpu", "--num_cpu", type=int, default=16, help="Number of cpus")

    parser.add_argument("-en", "--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("-s_epoch", "--start_epoch", type=int, default=0)

    parser.add_argument("-cpu", "--cpu", action="store_true", default=False, help="move to CPU (poorly supported)")
    help = "Don't continue with the same --exp_name model."
    parser.add_argument("-new", "--new", action="store_true", default=False, help=help)
    help = "Image size"
    parser.add_argument("-s", "--size", type=int, default=help)
    help = "Image width (size becomes height); Not supported by most data sets"
    parser.add_argument("--size_w", type=int, default=-1, help=help)
    parser.add_argument("-gpus", "--gpus", nargs="+", default=None, type=int, help="GPU ids, (default id=0)")
    help = "Sets the 'detect_anomaly' flag of pytorch-lighting. Slows down training but the loss will not become NaN "
    parser.add_argument("-no_nan", "--prevent_nan", action="store_true", default=False, help=help)
    parser.add_argument("--transpose_preview", action="store_true", default=False)
    help = "Some pytorch versions are bugged. Error: Use this flag when you see this: 'If capturable=False, state_steps should not be CUDA tensors.'. This prevents the normal loading of the model. This flags loads the model ONLY"
    parser.add_argument("-legacy_reload", "--legacy_reload", action="store_true", default=False, help=help)  # TODO Implement in CUT
    return parser


def __parseParm_A2B_res(parser: ArgumentParser):
    parser.add_argument("-net_D_depth", "--net_D_depth", type=int, default=3)
    parser.add_argument("-net_D_channel", "--net_D_channel", type=int, default=64)
    parser.add_argument("-net_G_depth", "--net_G_depth", type=int, default=9)
    parser.add_argument("-net_G_channel", "--net_G_channel", type=int, default=64)
    parser.add_argument("-net_G_downsampling", "--net_G_downsampling", type=int, default=2)
    parser.add_argument("-net_G_drop_out", "--net_G_drop_out", type=float, default=0.5)


def parseParam_cut(parser: Optional[ArgumentParser] = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    from loader.load_model_cut import name_list

    parser.add_argument("-m", "--model_name", type=str, default="resnet", choices=name_list, help="specify generator architecture")
    parser.add_argument("--lambda_GAN", type=float, default=1.0, help="weight for GAN loss: GAN(G(X))")
    parser.add_argument("--lambda_NCE", type=float, default=1.0, help="weight for NCE loss: NCE(G(X), X)")
    parser.add_argument("--lambda_ssim", type=float, default=1.0, help="weight for SSIM loss: 1-ssim(X,G(X))")
    help = "weight for all paired loss: total = paired(X,G(X))"
    parser.add_argument("--lambda_paired", type=float, default=10, help=help)
    parser.add_argument("--nce_layers", type=str, default="0,4,8,12,16", help="compute NCE loss on which layers")
    parser.add_argument("--netF_nc", type=int, default=256)
    parser.add_argument("--nce_T", type=float, default=0.07, help="temperature for NCE loss")
    parser.add_argument("--num_patches", type=int, default=256, help="number of patches per layer")
    parser.add_argument("--nce_idt", action="store_true", default=False)
    parser.add_argument("--decay_epoch", type=int, default=-1)
    parser.add_argument(
        "--mode",
        type=str,
        default="cut",
        choices=["cut", "pix2pix", "paired_cut"],
        help="cut: unpaired training;\n paired_cut: paired training (+L1-loss)\n pix2pix: paired training with out contrastive loss",
    )

    __parseParm_A2B_res(parser)
    return parser


def parseDiffusion(parser: Optional[ArgumentParser] = None):
    from loader.load_model_diffusion import name_list

    if parser is None:
        parser = argparse.ArgumentParser()
    help = "Use L2 instead of L1 loss"
    parser.add_argument("-L2_loss", "--L2_loss", action="store_true", default=False, help=help)
    help = "This is only concert of the additional image as conditional, not label/embedding"
    parser.add_argument("-c", "--conditional", action="store_true", default=False, help=help)
    help = "Timestep linear"
    parser.add_argument("-l", "--linear", action="store_true", default=False, help=help)
    parser.add_argument("-ido", "--image_dropout", type=float, default=0)
    parser.add_argument("-v", "--volumes", action="store_true", default=False)
    parser.add_argument("-channels", "--channels", type=int, default=64)
    help = "A comma separated list of factors. The list length determents the network depth. The value is the channel (the argument --channels) increase at depth n by the factor of the value n."
    parser.add_argument("-dim_mults", "--dim_multiples", type=str, default="1, 2, 4, 8", help=help)
    parser.add_argument("-lv", "--learned_variance", action="store_true", default=False)
    help = "specify generator architecture"
    parser.add_argument("-m", "--model_name", type=str, default="unet", choices=name_list, help=help)
    help = "Improving Diffusion Model Efficiency Through Patching https://arxiv.org/abs/2207.04316; 1 = deactivated; Can be used for bigger images to reduce size."
    parser.add_argument("-patch_size", "--patch_size", type=int, default=1, help=help)
    help = "Number of diffusion steps (DDPM)"
    parser.add_argument("-timesteps", "--timesteps", type=int, default=1000, help=help)
    help = "Default: Predicting Noise, If set: Predicting Image"
    parser.add_argument("--image_mode", action="store_true", default=False, help=help)
    help = "weight for SSIM loss: 1-ssim(X,G(X)) for image mode"
    parser.add_argument("--lambda_ssim", type=float, default=0.0, help=help)

    # __parseParm_A2B_res(parser)
    # parseParam_img(parser)
    return parser


from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import get_origin, get_args, Type
import types


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
                    p.add_argument(key, action="store_true", default=False)
                else:
                    p.add_argument(key, action="store_false", default=True)
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


@dataclass
class Cut_Option(Train_Option):
    # Train
    decay_epoch: int = 30
    lambda_GAN: float = 1.0
    lambda_NCE: float = 1.0
    lambda_paired: float = 10.0
    lambda_ssim: float = 1

    nce_idt: bool = True
    start_epoch: int = 0
    cpu: bool = False
    new: bool = False
    # modes: cut, pix2pix, paired_cut
    mode: str = "cut"
    # Model
    # model_name Options (cut): resnet, base_unet, unet, style
    model_name: str = "resnet"
    ## Contrastive
    nce_layers: str = "0,4,8,12,16"
    netF_nc: int = 256
    nce_T: float = 0.07
    num_patches: int = 256
    ## Discriminator
    net_D_depth: int = 3
    net_D_channel: int = 64
    ## Generator
    net_G_depth: int = 9
    net_G_channel: int = 64
    net_G_downsampling: int = 2
    net_G_drop_out: float = 0.5
    condition_types: list[str] = field(default_factory=list)  # Used for medical images only


@dataclass
class Diffusion_Option(Train_Option):
    # Train
    L2_loss: bool = False
    conditional: bool = False  # This is only concert of the additional image as conditional, not label/embedding
    linear: bool = False
    volumes: bool = False
    channels: int = 64
    dim_multiples: str = "1, 2, 4, 8"
    learned_variance: bool = False
    patch_size: int = 1
    timesteps: int = 1000
    image_mode = False  # False: Noise is outputted by the diffusion model , True: The image itself is produced
    # latent_diffusion_mode = False
    # Internal type
    normalize: dict[str, list[float] | dict[str, list[float]]] | None = None
    lambda_ssim: float = 0.0


def get_latest_Checkpoint(opt, version="*", log_dir_name="lightning_logs", best=False, verbose=True) -> str | None:
    import glob
    import os

    ckpt = "*"
    if best:
        ckpt = "*best*"
    print() if verbose else None
    checkpoints = None

    if isinstance(opt, str) or not opt.new:
        if isinstance(opt, str):
            checkpoints = sorted(glob.glob(f"{log_dir_name}/{opt}/version_{version}/checkpoints/{ckpt}.ckpt"), key=os.path.getmtime)
        else:
            checkpoints = sorted(
                glob.glob(f"{log_dir_name}/{opt.exp_name}/version_{version}/checkpoints/{ckpt}.ckpt"),
                key=os.path.getmtime,
            )

        if len(checkpoints) == 0:
            checkpoints = None
        else:
            checkpoints = checkpoints[-1]
        print("Reload recent Checkpoint and continue training:", checkpoints) if verbose else None
    else:
        return None

    return checkpoints


@dataclass
class AutoencoderKL_Option(Train_Option):
    # override defaults
    lr: float = 4.5e-6
    # DDconfig
    ch: int = 128
    out_ch: int = 1
    in_channels: int = 1
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    attn_resolutions: list[int] = field(default_factory=list)
    dropout: float = 0.0
    resamp_with_conv: bool = True
    z_channels: int = 4
    double_z: bool = True
    use_linear_attn: bool = False
    attn_type: str = "vanilla"
    ### AE KL
    embed_dim: int = 4
    image_key = "target"
    colorize_n_labels: int | None = None
    ###
    normalize: dict[str, list[float] | dict[str, list[float]]] | None = None


def parseAE(parser: ArgumentParser):
    default = AutoencoderKL_Option()
    parser.add_argument("-ch", "--ch", type=int, default=default.ch)

    parser.add_argument("--ch_mult", nargs="+", default=default.ch_mult, type=int)
    parser.add_argument("--num_res_blocks", type=int, default=default.num_res_blocks)
    parser.add_argument("--z_channels", type=int, default=default.z_channels)
    parser.add_argument("--embed_dim", type=int, default=default.embed_dim)
    parser.add_argument("--dropout", type=float, default=default.dropout)
    return parser


if __name__ == "__main__":
    p = AutoencoderKL_Option.get_opt()
    print(AutoencoderKL_Option.from_kwargs(**p.parse_args().__dict__))
