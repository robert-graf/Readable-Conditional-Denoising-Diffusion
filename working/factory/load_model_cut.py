from factory.arguments import CycleGAN_Option
from models.cut_model import Discriminator, Discriminator3D
from utils.utils_diffusion import get_option

name_list = ["resnet", "unet", "style", "base_unet"]

from torch.nn import Module


def load_generator(opt: CycleGAN_Option, in_channel, reverse_gan=False) -> Module:
    # reverse_gan: instead of 1 -> n the prediction is n-> 1
    # 3D
    if not hasattr(opt, "output_channels"):
        opt.output_channels = 1
        # opt.target_shape3D = None
    # print(opt.__dict__, in_channel)
    num_cond = len(opt.condition_types) - opt.output_channels
    in_c = in_channel * num_cond
    out_c = in_channel * opt.output_channels
    if reverse_gan:
        i = in_c
        in_c = out_c
        out_c = i
        num_cond = opt.output_channels

    if getattr(opt, "volumes", False):
        print("3D Unet")
        from models.diffusion_unet3D import Unet

        return Unet(
            dim=opt.channels,
            dim_mults=get_option(opt, "dim_multiples", (1, 2, 4, 8), separated_list=True),
            channels=in_channel,
            conditional_dimensions=num_cond - 1,  # additional conditions images beyond the first one
            learned_variance=False,
            conditional_label_size=0,
            conditional_embedding_size=0,
            out_dim=out_c,
        )  # type: ignore

    # 2D
    if not hasattr(opt, "model_name") or opt.model_name == name_list[0]:  # Default 2D case
        from models.cut_model import Generator

        return Generator(in_c, out_c, **opt.__dict__)

    elif opt.model_name == name_list[1]:
        from models.cut_diffusion_unet import Unet_CUT

        return Unet_CUT(opt.net_G_channel, channels=in_c, learned_variance=False, resnet_block_groups=8)
    elif opt.model_name == name_list[2]:
        from models.cut_stylegan import StyleGAN2Generator

        return StyleGAN2Generator(in_c, out_c, **opt.__dict__)
    elif opt.model_name == name_list[3]:
        from models.cut_unet import UNet

        return UNet(in_c, out_c, **opt.__dict__)

    raise NotImplementedError()


def load_discriminator(opt: CycleGAN_Option, in_channel, use_paired, input_disc=False) -> Module:
    if not hasattr(opt, "output_channels"):
        opt.output_channels = 1
    if use_paired:
        num_cond = len(opt.condition_types)
    elif input_disc:
        num_cond = opt.output_channels
    else:
        num_cond = len(opt.condition_types) - opt.output_channels
    # 3D
    if getattr(opt, "volumes", False):
        return Discriminator3D(in_channel * num_cond, depth=opt.net_D_depth, channels=opt.net_D_channel)
    return Discriminator(in_channel * num_cond, depth=opt.net_D_depth, channels=opt.net_D_channel)
