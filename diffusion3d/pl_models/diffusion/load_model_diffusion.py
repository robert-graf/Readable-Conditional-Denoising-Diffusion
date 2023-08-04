from pl_models.diffusion.diffusion_utils import get_option
from models.diffusion_unet import Unet
from models.diffusion_unet3D import Unet as Unet3D
from utils import arguments


def load_model(opt: arguments.Diffusion_Option, in_channel, lightning_module) -> Unet | Unet3D:

    conditional_label_size = opt.conditional_label_size
    conditional_embedding_size = 0
    learned_variance = hasattr(opt, "learned_variance") and opt.learned_variance

    lightning_module.conditional_dimensions = opt.conditional_dimensions
    lightning_module.conditional_label_size = conditional_label_size
    lightning_module.conditional_embedding_size = conditional_embedding_size

    # self.generator = load_generator(opt, version='A2B')
    if opt.volumes:  # 3D
        return Unet3D(
            dim=opt.channels,
            dim_mults=get_option(opt, "dim_multiples", (1, 2, 4, 8), separated_list=True),
            channels=in_channel,
            conditional_dimensions=opt.conditional_dimensions,
            learned_variance=learned_variance,
            conditional_label_size=conditional_label_size,
            conditional_embedding_size=conditional_embedding_size,
        )  # type: ignore
    else:
        return Unet(
            dim=opt.channels,
            dim_mults=get_option(opt, "dim_multiples", (1, 2, 4, 8), separated_list=True),
            channels=in_channel,
            conditional_dimensions=opt.conditional_dimensions,
            learned_variance=learned_variance,
            conditional_label_size=conditional_label_size,
            conditional_embedding_size=conditional_embedding_size,
        )  # type: ignore
