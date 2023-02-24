from argparse import Namespace
from CUT import CUT
from loader.arguments import Cut_Option
from utils.utils_diffusion import get_option

name_list = ["resnet", "unet", "style", "base_unet"]

from models.diffusion_unet import Unet
from torch.nn import Module


def load_generator(opt: Cut_Option, in_channel, lightning_module: CUT) -> Module:
    if not hasattr(opt, "model_name") or opt.model_name == name_list[0]:  # Default 2D case
        from models.cut_model import Generator

        return Generator(in_channel, in_channel, **opt.__dict__)

    if opt.model_name == name_list[1]:
        from models.cut_diffusion_unet import Unet_CUT

        return Unet_CUT(opt.net_G_channel, channels=in_channel, learned_variance=False, resnet_block_groups=8)
    if opt.model_name == name_list[2]:
        from models.cut_stylegan import StyleGAN2Generator

        return StyleGAN2Generator(in_channel, in_channel, **opt.__dict__)
    if opt.model_name == name_list[3]:
        from models.cut_unet import UNet

        return UNet(in_channel, in_channel, **opt.__dict__)

    raise NotImplementedError()
