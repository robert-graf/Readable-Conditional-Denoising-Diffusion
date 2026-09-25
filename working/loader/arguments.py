from attr import dataclass
from factory.auto_arguments import Option_to_Dataclass
from dataclasses import asdict, field


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
    # scale_factor: int = 4  # Only used in upscaling
    num_validation_images: int = 8
    gpus: list[int] | None = None
    legacy_reload = False
    new: bool = False
    prevent_nan: bool = False
    volumes: bool = False
    dim_multiples: str = "1, 2, 4, 8"
    channels: int = 64
    target_shape3D: list[int] | None = None
    condition_types: list[str] = field(default_factory=list)  # Used for medical images only
    cpu: bool = False
    new: bool = False
    start_epoch: int = 0
    scale_factor = 1
    logs_folder: str = "logs_diffusion3D"
    output_channels: int = 1  # Not supported in 2D Datasets...

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
    # modes: cut, pix2pix, paired_cut
    mode: str = "pix2pix"
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

    # This is only concert of the additional image as conditional, not label/embedding (Diffusion only)
    conditional: bool = False


@dataclass
class Diffusion_Option(Train_Option):
    # Train
    L2_loss: bool = False
    linear: bool = False
    learned_variance: bool = False
    timesteps: int = 1000
    image_mode: bool = False  # False: Noise is outputted by the diffusion model , True: The image itself is produced
    # latent_diffusion_mode = False
    # Internal type
    normalize: dict[str, list[float] | dict[str, list[float]]] | None = None
    lambda_ssim: float = 0.0
    conditional: bool = False
    conditional_channel_size: int = -1


def get_latest_Checkpoint(opt: Train_Option, version="*", log_dir_name="lightning_logs", best=False, verbose=True) -> str | None:
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
