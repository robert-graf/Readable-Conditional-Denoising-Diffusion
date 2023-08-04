from attr import dataclass
from utils.auto_arguments import Option_to_Dataclass
from dataclasses import asdict, field


@dataclass
class Train_Option(Option_to_Dataclass):
    # Training
    experiment_name: str = "NAME"
    # try in this order 0.002,0.0002,0.00002
    lr: float = 0.0002

    batch_size: int = 1
    batch_size_val: int = 1

    num_epochs: int = 150
    num_cpu: int = 16

    target_patch_shape: list[int] | None = None
    flip: bool = True
    transpose_preview = False
    # Options: crop, resize

    # Options: unconditional, image
    # dataset_val: str | None = None

    gpus: list[int] | None = None
    legacy_reload = False
    new: bool = False
    prevent_nan: bool = False
    volumes: bool = False
    dim_multiples: str = "1, 2, 4, 8"
    channels: int = 64
    # condition_types: list[str] | None = None
    cpu: bool = False
    start_epoch: int = 0
    log_dir: str = "logs_diffusion3D"
    model_name: str = "unet"  # No Other implemented
    auto_lr_find = False
    dataset: str = ""
    dataset_val: str = ""


@dataclass
class Diffusion_Option(Train_Option):
    # Train
    L2_loss: bool = False
    linear: bool = False
    learned_variance: bool = False  # No longer supported;
    timesteps: int = 100
    image_mode: bool = False  # False: Noise is outputted by the diffusion model , True: The image itself is produced
    # Internal type
    # normalize: dict[str, list[float] | dict[str, list[float]]] | None = None

    conditional_dimensions: int = 4
    conditional_label_size: int = 2
    image_dropout: float = 0.0

    output_rows: list[str] | None = None
    input_rows: list[str] | None = None

    @property
    def conditional(self):
        return self.conditional_dimensions != 0


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
                glob.glob(f"{log_dir_name}/{opt.experiment_name}/version_{version}/checkpoints/{ckpt}.ckpt"),
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
