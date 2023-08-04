from utils.utils_diffusion import get_option

name_list = ["unet"]  # , "p_unet", "splice", "adversarial", "book"]
from models.diffusion_unet import Unet


def load_model(opt, in_channel, lightning_module) -> Unet:

    #### Increase input size by the conditional input #####
    if opt.conditional:
        if hasattr(opt, "conditional_channel_size"):
            conditional_dimensions = opt.conditional_channel_size
            print("conditional_dimensions", conditional_dimensions)
        #### LEGACY CODE ####
        elif not hasattr(opt, "condition_typs") or len(opt.condition_typs) == 0:
            print("Conditional")
            conditional_dimensions = in_channel
            setattr(opt, "conditional_dimensions", conditional_dimensions)
        else:
            print(opt.condition_typs)
            conditionals = ["MRI", "CT_4", "SG", "CT"]
            conditionals = [i for i in opt.condition_typs if i in conditionals]
            conditional_dimensions = in_channel * len(conditionals)
            print("Conditional", conditionals)
            setattr(opt, "conditional_dimensions", conditional_dimensions)
        #### LEGACY CODE ####
    else:
        conditional_dimensions = 0
    print("conditional_dimensions", conditional_dimensions)
    conditional_label_size = get_option(opt, "conditional_label_size", 0)
    print("conditional_label_size", conditional_label_size)
    conditional_embedding_size = get_option(opt, "conditional_embedding_size", 0)
    print("conditional_embedding_size", conditional_embedding_size)

    #### Increase output size by the learned_variance #####
    learned_variance = hasattr(opt, "learned_variance") and opt.learned_variance
    if not learned_variance:
        print("Warning: You are using the non-learned Variance")
    #
    lightning_module.conditional_dimensions = conditional_dimensions
    lightning_module.conditional_label_size = conditional_label_size
    lightning_module.conditional_embedding_size = conditional_embedding_size

    # self.generator = load_generator(opt, version='A2B')
    if opt.volumes:  # 3D
        # if opt.model_name == name_list[1]:
        #    from models.diffusion_unet3D_progressive import Unet
        # else:
        from models.diffusion_unet3D import Unet

    elif not hasattr(opt, "model_name") or opt.model_name == name_list[0]:  # Default 2D case
        from models.diffusion_unet import Unet
    # elif opt.model_name == name_list[2]:
    #    from models.diffusion_unet_splice import Unet
    # elif opt.model_name == name_list[3]:
    #    from models.diffusion_unet_adversarial import Unet
    # elif opt.model_name == name_list[4]:
    #    from models.diffusion_unet_wordbookenc import Unet
    else:
        from models.diffusion_unet import Unet
    return Unet(
        dim=get_option(opt, "channels", 64),
        dim_mults=get_option(opt, "dim_multiples", (1, 2, 4, 8), separated_list=True),
        channels=in_channel,
        conditional_dimensions=conditional_dimensions,
        learned_variance=learned_variance,
        conditional_label_size=conditional_label_size,
        conditional_embedding_size=conditional_embedding_size,
        patch_size=get_option(opt, "patch_size", 1),
    )  # type: ignore
