from argparse import Namespace
from CUT import CUT
from factory.arguments import Cut_Option
from models.cut_model import Discriminator, Discriminator3D
from utils.utils_diffusion import get_option

name_list = ["resnet", "unet", "style", "base_unet"]

from models.diffusion_unet import Unet
from torch.nn import Module


def load_generator(opt: Cut_Option, in_channel, lightning_module: CUT) -> Module:
    # 3D
    num_cond = len(opt.condition_types) - opt.output_channels
    if opt.volumes:
        print("3D Unet")
        from models.diffusion_unet3D import Unet

        return Unet(
            dim=opt.channels,
            dim_mults=get_option(opt, "dim_multiples", (1, 2, 4, 8), separated_list=True),
            channels=in_channel,
            conditional_dimensions=num_cond - 1,  # additional conditions
            learned_variance=False,
            conditional_label_size=0,
            conditional_embedding_size=0,
            out_dim=opt.output_channels * in_channel,
        )  # type: ignore

    # 2D
    if not hasattr(opt, "model_name") or opt.model_name == name_list[0]:  # Default 2D case
        from models.cut_model import Generator

        return Generator(in_channel * num_cond, in_channel, **opt.__dict__)

    elif opt.model_name == name_list[1]:
        from models.cut_diffusion_unet import Unet_CUT

        return Unet_CUT(opt.net_G_channel, channels=in_channel * num_cond, learned_variance=False, resnet_block_groups=8)
    elif opt.model_name == name_list[2]:
        from models.cut_stylegan import StyleGAN2Generator

        return StyleGAN2Generator(in_channel * num_cond, in_channel, **opt.__dict__)
    elif opt.model_name == name_list[3]:
        from models.cut_unet import UNet

        return UNet(in_channel * num_cond, in_channel, **opt.__dict__)

    raise NotImplementedError()


def load_discriminator(opt: Cut_Option, in_channel, use_paired) -> Module:
    num_cond = len(opt.condition_types)
    # 3D
    if opt.volumes:
        return Discriminator3D(
            in_channel * num_cond if use_paired else in_channel * opt.output_channels,
            depth=opt.net_D_depth,
            channels=opt.net_D_channel,
        )
    return Discriminator(
        in_channel * num_cond if use_paired else in_channel,
        depth=opt.net_D_depth,
        channels=opt.net_D_channel,
    )
