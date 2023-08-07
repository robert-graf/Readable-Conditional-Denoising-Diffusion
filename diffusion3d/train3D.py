from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from utils import arguments
from utils.Wrapper_datasets import Batch_Dict
from pl_models.diffusion.diffusion import Diffusion
import utils.make_snap as snap


class Diffusion3D(pl.LightningModule):
    def __init__(self, opt: arguments.Diffusion_Option) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.learning_rate = opt.lr
        opt.conditional_dimensions = len(opt.input_rows) + 3 if opt.input_rows is not None else 0
        self.channels = 1
        self.diffusion_net = Diffusion(opt, channel=self.channels, save_hyperparameters=False)

        self.counter = 0
        self.buffer_target = []
        self.buffer_condition = []
        self.buffer_labels = []
        self.buffer = None
        t = list(range(0, opt.timesteps, max(opt.timesteps // 10, 10)))
        t.append(opt.timesteps - 1)
        self.t = torch.Tensor(t).long()

    def configure_optimizers(self):
        self.opt.lr = self.learning_rate
        return self.diffusion_net.configure_optimizers()

    def training_step(self, train_batch: Batch_Dict, batch_idx):
        self.diffusion_net.constance.to(self.device)
        loss = self.diffusion_net.training_step(train_batch, batch_idx, do_log=True, compute_loss=True)  # type: ignore
        loss: Tensor = loss.mean()
        return loss

    def validation_step(self, batch: Batch_Dict, batch_idx, ddpm=False):
        opt = self.opt
        self.counter += 1

        tb_logger: TensorBoardLogger = self.logger  # type: ignore
        # Possible returns of the dataset
        target: Tensor = batch.get("target")
        x_conditional: Tensor | None = batch.get("condition", None) if opt.conditional else None
        label: Tensor | None = batch.get("label", None)
        # Generate Image

        generated_other: list[Tensor] = []
        if ddpm:
            generated: Tensor = self.diffusion_net.forward(
                target.shape[0],
                x_conditional=x_conditional,
                label=label,
            ).cpu()  # type: ignore
            raise NotImplementedError()
        else:
            generated: Tensor = self.diffusion_net.forward_ddim(
                target.shape[0],
                intermediate=list(range(0, self.diffusion_net.num_timesteps, min(10, self.diffusion_net.num_timesteps))),
                x_conditional=x_conditional,
                label=label,
                w=0,
            )  # type: ignore
            generated = generated[0].cpu()  # type: ignore

        generated = torch.clamp(generated, 0, 1)
        print(end="\r")
        # Log labels/embeddings
        # Print Label
        if label is not None:
            tb_logger.experiment.add_text(
                "generated_images_labels",
                str(label.tolist()),
                self.counter,
            )

        # 3D flattening on dimension, so it works like the 2D case
        generated = generated.swapaxes_(-1, -2).reshape((-1, 1, generated.shape[-2], generated.shape[-1]))
        if x_conditional is not None:
            if self.opt.volumes:
                target = target.swapaxes_(-1, -2).reshape((-1, 1, generated.shape[-2], generated.shape[-1]))
                x_conditional = x_conditional[:, 0].swapaxes_(-1, -2).reshape((generated.shape[0], 1, -1, generated.shape[-1]))

        # Print only generated images
        if x_conditional is None:
            grid = torchvision.utils.make_grid(generated, nrow=4)
            tb_logger.experiment.add_image("generated images", grid, self.counter)
        # Print conditional images and generated images side by side
        else:
            # Map to [0,1]
            target = self.diffusion_net.denormalize(target.cpu())
            # List of images that will be displayed side by side
            stacked_images = [target, generated]
            stacked_images += generated_other
            if x_conditional is not None:
                x_conditional = self.denormalize(x_conditional.cpu(), conditional=True)
                c: int = self.channels
                stacked_images += [x_conditional[:, i : i + c].cpu() for i in range(0, x_conditional.shape[1], c)]
            if opt.transpose_preview:
                stacked_images = [i.transpose(-1, -2) for i in stacked_images]
            try:
                st = torch.stack(stacked_images, dim=0).cpu()
                text = [f"GT {opt.output_rows}", "Predicted", *(opt.input_rows if opt.input_rows is not None else ["Input"])]
                grid = snap.make_grid_with_labels(
                    st.swapaxes(0, 1).reshape((-1, *st.shape[-3:])),
                    text + text + text,
                    nrow=9,
                )

                # grid = torchvision.utils.make_grid(
                #    torch.cat(stacked_images, dim=-1).cpu(),
                #    nrow=2 if not self.opt.volumes else 3,
                # )
                # snap.show(grid)
                tb_logger.experiment.add_image("conditional image", grid, self.counter)
            except Exception as e:
                raise e
                print([s.shape for s in stacked_images])

                exit()

    def denormalize(self, x, conditional=True):
        return self.diffusion_net.denormalize(x, conditional)


def main(opt: arguments.Diffusion_Option, limit_train_batches=1):
    #### Define dataset ####
    from torch.utils.data import DataLoader
    from utils.Wrapper_datasets import Wrapper_Label2Image, Wrapper_Image2Image
    from dataloader_3D import SameSpace_3D_Dataset
    import pandas as pd

    assert Path(opt.dataset).exists(), opt.dataset
    # dataset_val
    train_pd = pd.read_excel(opt.dataset)
    print(len(train_pd), opt.dataset)
    train_ds = Wrapper_Image2Image(
        SameSpace_3D_Dataset(
            train_pd, opt.target_patch_shape, keys_in=opt.input_rows, keys_out=opt.output_rows, flip=opt.flip, train=True, opt=opt
        ),
        image_dropout=opt.image_dropout,
    )
    val_pd = pd.read_excel(opt.dataset_val if opt.dataset_val != "" else opt.dataset)
    val_ds = Wrapper_Image2Image(
        SameSpace_3D_Dataset(val_pd, opt.target_patch_shape, keys_in=opt.input_rows, keys_out=opt.output_rows, train=False), image_dropout=0
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_cpu,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=opt.batch_size_val,
        shuffle=True,
        num_workers=opt.num_cpu,
        persistent_workers=True,
    )

    model = Diffusion3D(opt=opt)

    # Get last checkpoint. If there is non or --new was called this returns None and starts a new model.
    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name=opt.log_dir, best=False)
    ### We do not reload with trainer.fit, as my
    model.load_from_checkpoint(last_checkpoint) if last_checkpoint is not None else None
    last_checkpoint = None

    # Define Last and best Checkpoints to be saved.
    mc_last = ModelCheckpoint(
        filename="{epoch}-{step}-{train_All:.8f}_latest",
        monitor="step",
        mode="max",
        every_n_train_steps=min(500, len(train_loader)),
        save_top_k=3,
    )

    # mc_best = ModelCheckpoint(
    #    monitor="metric_val",
    #    mode="min",
    #    filename="{epoch}-{step}-{train_All:.8f}_best",
    #    every_n_train_steps=len(train_loader) + 1,
    #    save_top_k=2,
    # )
    from pytorch_lightning.callbacks import Callback

    # This sets the experiment name. The model is in /lightning_logs/{opt.exp_nam}/version_*/checkpoints/
    logger = TensorBoardLogger(opt.log_dir, name=opt.experiment_name, default_hp_metric=False)
    limit_train_batches = limit_train_batches if limit_train_batches != 1 else None

    gpus = opt.gpus
    accelerator = "gpu"
    if gpus is None:
        gpus = 1
    elif -1 in gpus:
        gpus = None
        accelerator = "cpu"
    # exit()
    # training
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=gpus,
        num_nodes=1,  # Train on 'n' GPUs; 0 is CPU
        limit_train_batches=limit_train_batches,  # Train only x % (if float) or train only on x batches (if int)
        limit_val_batches=3,
        max_epochs=opt.num_epochs,  # Stopping epoch
        logger=logger,
        callbacks=[mc_last],  # mc_best
        detect_anomaly=opt.prevent_nan,
        auto_lr_find=opt.auto_lr_find,
    )
    if opt.auto_lr_find:
        trainer.tune(
            model,
            train_loader,
            val_loader,
        )
        model.learning_rate *= 0.5
        try:
            next(Path().glob(".lr_find*")).unlink()
        except StopIteration:
            pass
    trainer.fit(model, train_loader, val_loader, ckpt_path=last_checkpoint)


def get_opt(config=None) -> arguments.Diffusion_Option:
    torch.cuda.empty_cache()
    opt = arguments.Diffusion_Option().get_opt(None, config)
    opt = arguments.Diffusion_Option.from_kwargs(**opt.parse_args().__dict__)
    opt.experiment_name = "Diffusion_3D_" + opt.experiment_name
    if opt.target_patch_shape == None:
        opt.target_patch_shape = [16, 128, 128]
        opt.volumes = True
    assert opt.conditional
    # opt.new = True
    return opt


if __name__ == "__main__":
    main(get_opt())
