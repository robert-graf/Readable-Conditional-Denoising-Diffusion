from __future__ import annotations


# %reload_ext tensorboard
# tensorboard --logdir logs_diffusion

import typing

import torch
from torch.utils.data import DataLoader
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from threading import Thread
from dataloader.Wrapper_datasets import Batch_Dict
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger

from loader import arguments
from loader import load_dataset


import time
from diffusion import Diffusion, get_opt
from AutoencoderKL import AutoencoderKL


class Latened_Diffusion(pl.LightningModule):
    def __init__(self, opt: arguments.Diffusion_Option, in_channel=1) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.channels = in_channel
        #### Initialize Models ####
        last_checkpoint = arguments.get_latest_Checkpoint(opt.ae_name, log_dir_name="logs_diffusion", best=True, version="*")  # type: ignore #TODO set best=True
        assert last_checkpoint is not None
        self.ae = AutoencoderKL.load_from_checkpoint(last_checkpoint, strict=False)
        # fix opt to new size and channels
        opt.size = opt.size // (2 ** (len(self.ae.encoder.down) - 1))
        opt.conditional_channel_size *= self.ae.opt.z_channels  # type: ignore

        self.diffusion = Diffusion(opt, in_channel=self.ae.opt.z_channels, easy_color_mode=False)
        self.opt = opt

        if opt.legacy_reload:
            last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name="logs_diffusion", best=False)
            if last_checkpoint is not None:
                lighting_dic = torch.load(last_checkpoint)
                self.load_state_dict(lighting_dic, strict=False)
        # Fix weights
        # for par in self.ae.parameters():
        #    par.requires_grad = False
        self.counter = 0

    @torch.no_grad()
    def forward(
        self,
        batch_size: int,
        encode_target: Tensor | None = None,
        mask_for_inpainting: Tensor | None = None,
        x_conditional: Tensor | None = None,
        **kwargs,
    ) -> Tensor | tuple[Tensor, typing.List[Tensor]]:
        # Encode
        assert mask_for_inpainting is None, "not implemented"
        if encode_target is not None:
            sampler = self.ae.encode(encode_target)
            # sampler.deterministic = True
            encode_target = sampler.sample()
        if x_conditional is not None:
            sampler = self.ae.encode(x_conditional)
            # sampler.deterministic = True
            x_conditional = sampler.sample()
        # DDPM
        out = self.diffusion.forward(batch_size, encode_target=encode_target, x_conditional=x_conditional, **kwargs)
        # Decode
        out_list = None
        if isinstance(out, tuple):
            out, out_list = out

        out = self.ae.decode(out)
        if out_list is not None:
            return out, [self.ae.decode(i) for i in out_list]
        return out

    @torch.no_grad()
    def forward_ddim(
        self,
        batch_size: int,
        intermediate: list[int] | range,  # List of steps visited by DDIM
        x_conditional: Tensor | None = None,
        return_noise=False,
        **kwargs,
    ):
        # Encode
        if x_conditional is not None:
            sampler = self.ae.encode(x_conditional)
            # sampler.deterministic = True
            x_conditional = sampler.sample()
        # DDIM
        out = self.diffusion.forward_ddim(
            batch_size, intermediate, x_conditional=x_conditional, return_noise=return_noise, **kwargs
        )
        # Decode
        noise = None
        if return_noise:
            out, out_list, noise = out  # type: ignore
        else:
            out, out_list = out  # type: ignore
        out_list = [self.ae.decode(i) for i in out_list]
        out = self.ae.decode(out)
        if return_noise:
            return out, out_list, noise
        return out, out_list

    def configure_optimizers(self):
        return self.diffusion.configure_optimizers()

    @torch.no_grad()
    def encode_batch(self, train_batch: Batch_Dict):
        mask_for_inpainting: Tensor | None = train_batch.get("mask", None)
        if mask_for_inpainting is not None:
            raise NotImplementedError()
        x_0 = train_batch.get("target")
        train_batch["target"] = self.ae.encode(x_0).sample()

        if "condition" in train_batch and train_batch["condition"] is not None:
            x_conditional = train_batch["condition"]
            mask_for_inpainting_cond: Tensor | None = train_batch.get("mask_cond", None)
            if mask_for_inpainting_cond is not None:
                x_conditional = x_conditional * mask_for_inpainting_cond
                train_batch["mask_cond"] = None
            train_batch["condition"] = self.ae.encode(x_conditional).sample()
        return train_batch

    def training_step(self, train_batch: Batch_Dict, batch_idx):
        self.diffusion.trainer = self.trainer
        train_batch = self.encode_batch(train_batch)

        return self.diffusion.training_step(train_batch, batch_idx, img_decoder=self.ae)

    def validation_step_by_callback(self, batch: Batch_Dict, batch_idx, logger: TensorBoardLogger | None = None):
        if self.trainer is not None:
            self.diffusion.trainer = self.trainer
        opt = self.opt
        self.counter += 1
        if self.counter % 5 != 0:
            return
        if logger is None:
            tb_logger: TensorBoardLogger = self.logger  # type: ignore
        else:
            tb_logger = logger
        target: Tensor = batch.get("target")
        x_conditional: Tensor | None = batch.get("condition", None) if opt.conditional else None
        label: Tensor | None = batch.get("label", None)
        embedding: Tensor | None = batch.get("embedding", None)
        # mask_for_inpainting: Tensor | None = batch.get("mask", None)
        mask_for_inpainting_cond: Tensor | None = batch.get("mask_cond", None)
        # encode_target: Tensor | None = target if mask_for_inpainting is not None else None

        if mask_for_inpainting_cond is not None:
            x_conditional2 = x_conditional * mask_for_inpainting_cond
        else:
            x_conditional2 = x_conditional

        generated_other: list[Tensor] = []

        generated: Tensor = self.forward(
            opt.num_validation_images, x_conditional=x_conditional2, label=label, embedding=embedding
        ).cpu()  # type: ignore
        print(end="\r")
        try:
            # Print Label
            if label is not None:
                tb_logger.experiment.add_text("generated_images_labels", str(label.tolist()), self.counter)
            if embedding is not None:
                tb_logger.experiment.add_text("generated_images_embedding", str(embedding.tolist()), self.counter)
        except Exception as e:
            print("Warning in text logging:", str(e))

        # 3D flattening on dimension, so it works like the 2D case
        if self.opt.volumes:
            generated = generated.permute(0, 1, 3, 4, 2)
            generated = generated.reshape((-1, 1, generated.shape[-2], generated.shape[-1]))
            if x_conditional is not None:
                if self.opt.volumes:
                    target = target.permute(0, 1, 3, 4, 2).reshape((-1, 1, generated.shape[-2], generated.shape[-1]))
                    x_conditional = x_conditional.permute(0, 1, 3, 4, 2)
                    x_conditional = x_conditional.reshape((-1, 1, generated.shape[-2], generated.shape[-1]))

        # Print only generated images
        if x_conditional is None:  # and mask_for_inpainting is None:
            grid = torchvision.utils.make_grid(generated, nrow=4)
            tb_logger.experiment.add_image("generated images", grid, self.counter)
        # Print conditional images and generated images side by side
        else:
            # Map to [0,1]
            target = self.diffusion.denormalize(target.cpu())
            # List of images that will be displayed side by side
            stacked_images = [target, generated]
            stacked_images += generated_other
            if x_conditional is not None:
                x_conditional = self.diffusion.denormalize(x_conditional.cpu(), conditional=True)
                c = self.channels
                stacked_images += [x_conditional[:, i : i + c].cpu() for i in range(0, x_conditional.shape[1], c)]
            if mask_for_inpainting_cond is not None:
                assert x_conditional2 is not None
                c = self.channels
                x_conditional2 = self.diffusion.denormalize(x_conditional2.cpu(), conditional=True)
                stacked_images += [x_conditional2[:, i : i + c].cpu() for i in range(0, x_conditional2.shape[1], c)]

            # if mask_for_inpainting is not None:
            #    stacked_images.append((mask_for_inpainting.cpu() * target))

            grid = torchvision.utils.make_grid(
                torch.cat(stacked_images, dim=-1).cpu(),
                nrow=2 if not self.opt.volumes else 5,
            )

            if self.opt.volumes:
                grid = torch.nn.functional.interpolate(grid, scale_factor=2, mode="nearest")  # type: ignore

            tb_logger.experiment.add_image("conditional image", grid, self.counter)


def main(opt: arguments.Diffusion_Option, limit_train_batches=1.0):
    #### Define transformation. ####

    #### Define dataset ####
    if opt.volumes:
        setattr(opt, "num_validation_images", 1)
    else:
        setattr(opt, "num_validation_images", 8)
    dataset, in_channel = load_dataset.getDataset(opt, True)
    dataset_val, _ = load_dataset.getDataset(opt, False)
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_cpu, shuffle=True, drop_last=True)
    from dataloader.dataloader_basic import DummyDataset

    dataset_val = DummyDataset(num_sample=opt.num_validation_images, ds=dataset_val)
    dummy_val_loader = DataLoader(
        dataset_val,
        batch_size=opt.num_validation_images,
        num_workers=4,
        drop_last=False,
    )

    # Get information from the dataset
    if opt.conditional:
        assert (
            dataset.has_condition()
        ), "You selected (image) conditional, but the dataset does not support conditioning with an image."
        conditional_channel_size = dataset.get_conditional_channel_size()
        if conditional_channel_size == -1:
            conditional_channel_size = in_channel
        setattr(opt, "conditional_channel_size", conditional_channel_size)
    else:
        setattr(opt, "conditional_channel_size", 0)
    if dataset.has_label():
        setattr(opt, "conditional_label_size", dataset.get_label_count())
    if dataset.has_embedding():
        setattr(opt, "conditional_embedding_size", dataset.get_embedding_size())
    if hasattr(dataset, "normalize"):
        setattr(opt, "normalize", dataset.normalize)
        print("normalize:", opt.normalize)
    else:
        print("normalize: default")
    # opt.image_channels_B = opt.image_channels_A
    model = Latened_Diffusion(opt=opt, in_channel=in_channel)
    # Get last checkpoint. If there is non or --new was called this returns None and starts a new model.
    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name="logs_diffusion", best=True)

    # Define Last and best Checkpoints to be saved.

    mc_last = ModelCheckpoint(
        filename="{epoch}-{step}-{train_All:.4f}_latest",
        monitor="step",
        mode="max",
        every_n_train_steps=min(200, len(dataset)),
        save_top_k=3,
    )

    from pytorch_lightning.callbacks import Callback

    class ValidationCallback(Callback):
        def on_train_epoch_start(self, trainer, pl_module: Latened_Diffusion):
            pl_module.cuda()
            time.sleep(0.5)
            batch = {}
            for k, v in next(iter(dummy_val_loader)).items():
                batch[k] = v.to(pl_module.device)
            # next(iter(dummy_val_loader))
            pl_module.diffusion.thread = Thread(target=pl_module.validation_step_by_callback, args=(batch, 0))
            pl_module.diffusion.thread.start()
            print(end="\r")

        def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: Latened_Diffusion) -> None:
            if not pl_module.diffusion.thread is None:
                pl_module.diffusion.thread.join(timeout=600)
                pl_module.diffusion.thread = None
                print(end="\r")

        def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: Latened_Diffusion, outputs, batch, batch_idx: int
        ) -> None:
            pl_module.diffusion.ema.update()

    opt.print()
    # This sets the experiment name. The model is in /lightning_logs/{opt.exp_nam}/version_*/checkpoints/
    logger = TensorBoardLogger("logs_diffusion", name=opt.exp_name, default_hp_metric=False)
    limit_train_batches = limit_train_batches if limit_train_batches != 1 else None

    gpus = opt.gpus
    accelerator = "gpu"
    if gpus is None:
        gpus = 1
    elif -1 in gpus:
        gpus = []
        accelerator = "cpu"

    # training
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=gpus,
        num_nodes=1,  # Train on 'n' GPUs; 0 is CPU
        limit_train_batches=limit_train_batches,  # Train only x % (if float) or train only on x batches (if int)
        max_epochs=opt.max_epochs,  # Stopping epoch
        logger=logger,
        callbacks=[mc_last, ValidationCallback()],
        resume_from_checkpoint=last_checkpoint,
        detect_anomaly=opt.prevent_nan
        # amp_backend="apex",
        # amp_level="O3",
    )
    trainer.tune(model)
    trainer.fit(
        model,
        train_loader,
        # dummy_val_loader,
    )  # ckpt_path=last_checkpoint


if __name__ == "__main__":
    from configargparse import ArgumentParser

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-ae", "--ae_name", type=str, required=True, help="Experiment name")
    opt = get_opt(parser=parser)

    main(opt)
