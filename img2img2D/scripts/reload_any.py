from __future__ import annotations
from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

from loader.arguments import get_latest_Checkpoint


from configargparse import ArgumentParser

from dataclasses import dataclass, asdict, field
from inspect import signature
from CUT import CUT
from diffusion import Diffusion
from diffusion_adversarial import Diffusion_Adversarial
from typing import Optional, Type, Union, TypeGuard, TYPE_CHECKING
import torch, os
from utils.nii_utils import nii2arr
from enum import auto, Enum
from torch import Tensor


@dataclass()
class SynDiffNet(Diffusion):
    args_syndiff: Syndiff_Option

    def cuda(self) -> SynDiffNet:
        return self


try:
    syndiff_path = "../SynDiff/"
    if Path(syndiff_path, "test.py").exists():
        Path(syndiff_path, "test.py").rename(Path(syndiff_path, "SynDiff_test.py"))
    sys.path.append(syndiff_path)
    from backbones.ncsnpp_generator_adagn import NCSNpp as SynDiffNet  # type: ignore
    from SynDiff_test import load_checkpoint, Posterior_Coefficients, sample_from_model, get_time_schedule  # type: ignore
except Exception as e:
    print(e)
    print("Could not load 'SynDiff', Download SynDiff. Rename test.py to SynDiff_test.py")


class ComparableEnum(Enum):
    def __eq__(self, other):
        if isinstance(other, Enum):
            if other.__class__.__name__ != self.__class__.__name__:
                return False
            return self.value == other.value
        return False


class TranslationType(ComparableEnum):
    V_STITCHING = auto()
    TOP = auto()


class ResamplingType(ComparableEnum):
    NATIVE = auto()
    SAGITTAL = auto()
    ISO = auto()


def translation_type(enum: Type[Enum] = TranslationType) -> list[str]:
    choice = []
    for v in enum:
        choice.append(v.name)
    return choice


@dataclass
class Reload_Any_Option:
    exp_name: str
    root: str
    version: str = "*"
    cut: bool = False
    ddpm: bool = False
    ddim: bool = False
    syndiff: bool = False
    adversarial: bool = False
    majority_vote: bool = False
    guidance_w: float = 1
    bs: int = 12
    test = False
    new: bool = False
    factor: int = 10
    timesteps: int = 1000
    eta: float = 1.0
    keep_cdt: bool = False
    override: bool = False
    cdt_from: str | Reload_Any_Option | None = None
    match_string: str = "*-1_t2.nii*"  # "+-1_T2w.nii.gz"
    T1: bool = False
    out: str = None  # type: ignore
    translationType: TranslationType = TranslationType.V_STITCHING
    resample: ResamplingType = ResamplingType.SAGITTAL
    keep_resampled: bool = False
    log_dir_name = "logs_diffusion"

    @classmethod
    def from_kwargs(cls, **kwargs):
        # fetch the constructor's signature
        cls_fields = {field for field in signature(cls).parameters}

        # split the kwargs into native ones and new ones
        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in cls_fields:
                native_args[name] = val
            else:
                new_args[name] = val

        # use the native ones to create the class ...
        ret = cls(**native_args)

        # ... and add the new ones by hand
        for new_name, new_val in new_args.items():
            setattr(ret, new_name, new_val)
        if ret.out is None:
            ret.out = ret.root
        if "resample" in kwargs:
            ret.resample = ResamplingType[kwargs["resample"]]
        if "translationType" in kwargs:
            ret.translationType = TranslationType[kwargs["translationType"]]
        if "match_string" in kwargs:
            ret.match_string = str(kwargs["match_string"]).replace("+", "*")
        return ret

    def get_result_folder_name(self) -> str:
        result_folder = "result_" + self.exp_name
        if self.ddim or self.adversarial:
            result_folder += f"_ddim_eta-{self.eta:.1f}_{self.timesteps}"
        elif self.ddpm:
            result_folder += "_ddpm"
        if self.guidance_w != 1:
            result_folder += f"_w={int(self.guidance_w)}"
        return result_folder

    def get_cdt_from_name(self) -> str:
        assert self.cdt_from is not None
        if isinstance(self.cdt_from, Reload_Any_Option):
            return self.cdt_from.get_result_folder_name()
        result_folder = "result_" + self.cdt_from
        return result_folder

    def get_folder_rawdata(self) -> Path:
        return Path(self.get_folder(), "rawdata")

    def get_folder_derivatives(self) -> Path:
        return Path(self.get_folder(), "derivatives")

    def get_folder(self) -> Path:
        rawdata_folder = Path(f"{self.out}/{self.get_result_folder_name()}")
        return rawdata_folder


@dataclass
class Syndiff_Option:
    image_size = 256
    size = 256
    # exp = "exp_syndiff"
    num_channels = 2
    num_channels_dae = 64
    ch_mult: list[int] = field(default_factory=lambda: [1, 1, 2, 2, 4, 4])  # field(default_factory=list)
    num_timesteps = 4
    num_res_blocks = 2
    # batch_size = 1
    embedding_type = "positional"
    z_emb_dim = 256
    contrast1 = "CT"
    contrast2 = "T1"
    which_epoch = 500
    # gpu_chose = 0
    path = "../SynDiff/output/"
    # input_path = "../datasets/fx_T1w/"
    # output_path output/
    not_use_tanh = False
    attn_resolutions: list[int] = field(default_factory=lambda: [16])
    dropout = 0
    resamp_with_conv = True
    conditional = True
    fir = True
    fir_kernel: list[int] = field(default_factory=lambda: [1, 3, 3, 1])
    skip_rescale = True
    resblock_type = "biggan"
    progressive = "none"
    progressive_input = "residual"
    progressive_combine = "sum"
    nz = 100
    n_mlp = 3
    beta_min = 0.1
    beta_max = 20
    use_geometric = False
    centered = True


args_syndiff = Syndiff_Option()


def get_option_reload(no_model_picking: bool = False):
    parser = ArgumentParser()
    parser.add_argument("-config", "--config", is_config_file=True, help="config file path")
    if not no_model_picking:

        group = parser.add_mutually_exclusive_group()
        group.add_argument("--cut", action="store_true")
        group.add_argument("--ddpm", action="store_true")
        group.add_argument("--ddim", action="store_true")
        group.add_argument("--syndiff", action="store_true")
        group.add_argument("--majority_vote", action="store_true")
        group.add_argument("--adversarial", action="store_true")

        parser.add_argument("-en", "--exp_name", type=str)
        parser.add_argument("-v", "--version", default="*", type=str)
        parser.add_argument("-w", "--guidance_w", default=1, type=int, help="only ddim/ddpm")
        parser.add_argument("-t", "--timesteps", default=1, type=int, help="only ddim")
        parser.add_argument("--eta", default=1, type=float, help="only ddim")
        parser.add_argument("-bs", "--bs", default=12, type=int)
    else:
        parser.add_argument("--exp_name", default="no_name_required")
        parser.add_argument("--T1", action="store_true")

    parser.add_argument("--root", default="/media/data/robert/datasets/fx_T1w/", type=str)
    parser.add_argument("--out", default=None, type=str)
    parser.add_argument("--cdt_from", default=None, type=str)

    parser.add_argument("--keep_cdt", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--match_string", default="*-1_t2.nii*", type=str)
    parser.add_argument("--translationType", default="V_STITCHING", choices=translation_type())
    parser.add_argument("--resample", default="SAGITTAL", choices=translation_type(enum=ResamplingType))
    parser.add_argument("--keep_resampled", action="store_true", default=False)

    return parser


def get_option_test(parser: ArgumentParser):
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--factor", default=1, type=int)

    return parser


def get_option(parser: ArgumentParser):
    opt = parser.parse_args()
    print(opt)
    return Reload_Any_Option.from_kwargs(**opt.__dict__)


Models = Union[CUT, Diffusion, SynDiffNet, Diffusion_Adversarial]
# from BIDS.nii_utils


def get_model(opt: Reload_Any_Option, no_reload=False, verbose=True) -> tuple[Models, Path]:
    model = None
    if opt.majority_vote:
        return None, None  # type: ignore
    checkpoint = get_latest_Checkpoint(
        opt.exp_name, log_dir_name=opt.log_dir_name, best=False, version=opt.version, verbose=verbose
    )
    if checkpoint is not None and no_reload:
        return None, checkpoint  # type: ignore
    if opt.cut:
        assert checkpoint is not None, f"{opt.exp_name} - version {opt.version}, does not exist"
        model = CUT.load_from_checkpoint(checkpoint, strict=False)
    elif opt.adversarial:
        assert checkpoint is not None, f"{opt.exp_name} - version {opt.version}, does not exist"
        model = Diffusion_Adversarial.load_from_checkpoint(checkpoint, strict=False)
    elif opt.syndiff:
        # Initializing and loading network

        exp_path = os.path.join(args_syndiff.path, opt.exp_name)
        print(exp_path)
        checkpoint = os.path.join(exp_path, "content.pth")
        if not no_reload:
            gen_diffusive_1 = SynDiffNet(args_syndiff).cuda()
            try:
                checkpoint_dict = torch.load(checkpoint, map_location="cuda:0")["gen_diffusive_1_dict"]

            except Exception as ex:
                checkpoint_file = exp_path + "/{}_{}.pth"
                epoch_chosen = 5000
                for epoch_chosen in range(args_syndiff.which_epoch, 0, -10):
                    checkpoint = checkpoint_file.format("gen_diffusive_1", str(epoch_chosen))
                    if Path(checkpoint).exists():
                        print("Failed to read content.pth, use now:", checkpoint)
                        break
                checkpoint_dict = torch.load(checkpoint, map_location="cuda:0")
                # load_checkpoint(
                #    checkpoint_file, gen_diffusive_1, "gen_diffusive_1", epoch=str(epoch_chosen), device="cuda:0"
                # )
            for key in list(checkpoint_dict.keys()):
                checkpoint_dict[key[7:]] = checkpoint_dict.pop(key)  # remove module.
            gen_diffusive_1.load_state_dict(checkpoint_dict)
            model = gen_diffusive_1
            if opt.root == "/media/data/robert/datasets/spinegan_T2w":
                con_type = ["CT", "T2"]
            else:
                con_type = ["CT", "T1"]
            args_syndiff.condition_types = con_type  # type: ignore
            args_syndiff.num_cpu = 16  # type: ignore
            model.opt = args_syndiff  # type: ignore
    else:
        assert checkpoint is not None, f"'{opt.exp_name}' - version {opt.version}, does not exist"
        # c = torch.load(checkpoint)
        # if "discriminator.model.0.weight" in c:
        #    model = Diffusion.load_from_checkpoint(checkpoint, strict=False)
        # else:
        model = Diffusion.load_from_checkpoint(checkpoint, strict=False)
    if model is not None:
        model.cuda()
        model.eval()
    assert checkpoint is not None, f"{opt.exp_name} - version {opt.version}, does not exist"
    return model, Path(checkpoint)  # type: ignore


def is_CUT(model, opt) -> TypeGuard[CUT]:
    return isinstance(model, CUT)


def is_DDIM(model, opt: Reload_Any_Option) -> TypeGuard[Diffusion]:
    if opt.ddim:
        return isinstance(model, Diffusion)
    else:
        return False


def is_DDPM(model, opt: Reload_Any_Option) -> TypeGuard[Diffusion]:
    if opt.ddpm:
        return isinstance(model, Diffusion)
    else:
        return False


def is_syndiff(model, opt: Reload_Any_Option) -> TypeGuard[SynDiffNet]:
    if opt.syndiff:
        try:
            return isinstance(model, SynDiffNet)
        except:
            print("Could not load 'SynDiff.backbones.ncsnpp_generator_adagn'")
            pass
    return False


def is_adversarial_diffusion(model, opt: Reload_Any_Option) -> TypeGuard[Diffusion_Adversarial]:
    if opt.adversarial:
        try:
            return isinstance(model, Diffusion_Adversarial)
        except:
            pass
    return False


def get_image(cond: Tensor, model: Models, opt: Reload_Any_Option) -> Tensor:
    assert isinstance(cond, Tensor), type(cond)
    assert cond.min() >= -1, f"[{cond.min()} - {cond.max()}]"
    assert cond.min() <= 1, f"[{cond.min()} - {cond.max()}]"

    if is_CUT(model, opt):
        out = (model.forward(cond) + 1) / 2  # type: ignore
    elif is_DDPM(model, opt):
        out = model.forward(cond.shape[0], x_conditional=cond, guidance_w=opt.guidance_w)
        out.clamp_(0, 1)
    elif is_DDIM(model, opt) or is_adversarial_diffusion(model, opt):
        out = model.forward_ddim(
            cond.shape[0],
            range(0, 1000, 1000 // opt.timesteps),
            x_conditional=cond,
            w=opt.guidance_w,
            eta=opt.eta,
            progressbar=False,
        )
        out = out[0].cpu()
        out.clamp_(0, 1)
    elif is_syndiff(model, opt):
        device = cond.device
        pos_coeff = Posterior_Coefficients(args_syndiff, device)
        T = get_time_schedule(args_syndiff, device)
        x1_t = torch.cat((torch.randn_like(cond), cond), 1)
        # diffusion steps
        fake_sample1 = sample_from_model(pos_coeff, model, args_syndiff.num_timesteps, x1_t, T, args_syndiff)
        out = (fake_sample1 + 1) / 2
        # fake_sample1 = crop(fake_sample1)
        out.clamp_(0, 1)

    else:
        assert False, "Model not supported\n\n" + str(opt) + "\n" + str(model._get_name())
    return out


import nibabel as nib
import numpy as np

from utils.nii_utils import v_idx2name


def dice(im1, im2, value: Optional[int] = None, sift=0):
    im1 = np.asarray(im1).copy()
    im2 = np.asarray(im2).copy()
    if value is not None:
        # print(
        #    v_idx2name[value], v_idx2name[value + sift], value, sift, np.unique(im1), np.unique(im2)
        # ) if sift * sift > 1 else None
        im1[im1 != value] = 0
        im2[im2 != value + sift] = 0

    im1[im1 != 0] = 1
    im2[im2 != 0] = 1
    im1 = im1.astype(bool)
    im2 = im2.astype(bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    q = im1.sum() + im2.sum()
    if q == 0:
        return -1  # for filtering
    return 2.0 * intersection.sum() / q


partial = {
    "mask_case12.nii.gz": 8,
    "mask_case134.nii.gz": 8,
    "mask_case143.nii.gz": 8,
    "mask_case164.nii.gz": 8,
    "mask_case203.nii.gz": 8,
    "mask_case215.nii.gz": 8,
    "mask_case32.nii.gz": 8,
    "mask_case48.nii.gz": 8,
    "mask_case5.nii.gz": 8,
    "mask_case55.nii.gz": 8,
    "mask_case57.nii.gz": 8,
    "mask_case75.nii.gz": 8,
    "mask_case76.nii.gz": 8,
    "mask_case79.nii.gz": 8,
    "mask_case82.nii.gz": 8,
    "mask_case107.nii.gz": 9,
    "mask_case109.nii.gz": 9,
    "mask_case130.nii.gz": 9,
    "mask_case187.nii.gz": 9,
    "mask_case202.nii.gz": 9,
    "mask_case22.nii.gz": 9,
    "mask_case29.nii.gz": 9,
    "mask_case37.nii.gz": 9,
    "mask_case41.nii.gz": 9,
    "mask_case6.nii.gz": 9,
    "mask_case66.nii.gz": 9,
    "mask_case68.nii.gz": 9,
    "mask_case72.nii.gz": 9,
    "mask_case73.nii.gz": 9,
    "mask_case81.nii.gz": 9,
    "mask_case85.nii.gz": 9,
    "mask_case98.nii.gz": 9,
}


def compute_dice(file_seg, file_seg_gt, id, result, mapping={}, mapping_pred={}):
    file_seg_gt = Path(file_seg_gt)
    if not file_seg_gt.exists():
        print("Ground truth does not exist", file_seg_gt)
        return

    nii: nib.Nifti1Image = nib.load(file_seg_gt)

    im_gt = nii2arr(nii)
    im_gt[im_gt == -1024] = 0
    im_gt[im_gt >= 64512] = 0  # wraparound
    if file_seg_gt.name in partial:  # Remove partial volumes
        im_gt[im_gt >= partial[file_seg_gt.name]] = 0
    if len(mapping) != 0:
        im_gt_c = im_gt.copy()
        for ky, vl in mapping.items():
            im_gt[im_gt_c == ky] = vl

    vert_ids = list(np.unique(im_gt))[1:]  # [1:] -> ignore 0
    # No predicted file, set all dice to 0
    if not Path(file_seg).exists():
        result[0][id] = 0
        for i in np.unique(vert_ids):
            assert i != 0
            result[int(i)][id] = 0
        return

    nii: nib.Nifti1Image = nib.load(file_seg)
    im_pred = nii2arr(nii)
    im_pred[im_pred < 0] = 0  # -1024 is set for no image present...
    im_pred[im_pred >= 64512] = 0  # wraparound
    if len(mapping_pred) != 0:
        im_pred_c = im_pred.copy()
        for ky, vl in mapping_pred.items():
            im_pred[im_pred_c == ky] = vl
    # Run Evaluation
    for i in np.unique(vert_ids):
        assert i != 0

        dice_vert1 = dice(im_gt, im_pred, value=i, sift=1)  # Labels dont match perfectly...
        dice_vert2 = dice(im_gt, im_pred, value=i, sift=0)
        dice_vert3 = dice(im_gt, im_pred, value=i, sift=-1)

        if i == 19:
            # print(Path(file_seg).name)
            dice_vert4 = dice(im_gt, im_pred, value=i, sift=6)
            # print(max(dice_vert1, dice_vert2, dice_vert3), dice_vert4)
            dice_vert3 = max(dice_vert4, dice_vert3)
        dice_vert = max(dice_vert1, dice_vert2, dice_vert3)
        if dice_vert < 0.5:
            dice_vert1 = dice(im_gt, im_pred, value=i, sift=2)
            dice_vert2 = dice(im_gt, im_pred, value=i, sift=-2)
            dice_vert = max(dice_vert1, dice_vert2, dice_vert)

        result[int(i)][id] = dice_vert
    dice_global = dice(im_pred, im_gt)
    # print("global", dice_global)
    result[0][id] = dice_global


def run_docker(folder_name: Path, verbose=True):
    from BIDS.bids_utils import run_cmd

    # docker run -v /media/data/NAKO/MRT/test/new_docker2/result_paper_T2_pix2pix/rawdata/:/data christianpayer/verse19 /predict.sh --user 1000

    # docker run --user 1000 -v /media/data/robert/test/MRSpineSeg_ours/result_paper_T1_pix2pix_sa-unet/:/data anjanys/bonescreen-segmentor:main /src/process_spine.py
    code = run_cmd(
        [
            "docker",
            "run",
            "--user",
            str(os.getuid()),  # $UID
            "-v",
            f"{folder_name}:/data",
            "anjanys/bonescreen-segmentor:main",
            "/src/process_spine.py",
        ],
        verbose=verbose,
    )
    return code
