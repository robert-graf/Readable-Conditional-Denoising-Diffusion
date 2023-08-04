import glob
from itertools import chain
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import torchvision
from configargparse import ArgumentParser
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from utils import arguments
from utils.Wrapper_datasets import Batch_Dict
from pl_models.diffusion.diffusion import Diffusion
from models.discriminator3D import ClassPatchDiscriminator3D


class StarDiffusion(pl.LightningModule):
    def __init__(self, opt: arguments.Diffusion_Option) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        opt.conditional_label_size
        self.learning_rate = opt.lr

        self.diffusion_net = Diffusion(opt, save_hyperparameters=False)
        self.disc = ClassPatchDiscriminator3D(1, kernel_size=(1, 4, 4))
        self.criterion_class = torch.nn.CrossEntropyLoss()

        self.channels = 1
        self.counter = 0
        self.buffer_target = []
        self.buffer_condition = []
        self.buffer_labels = []
        self.buffer = None
        t = list(range(0, opt.timesteps, max(opt.timesteps // 10, 10)))
        t.append(opt.timesteps - 1)
        self.t = torch.Tensor(t).long()
        self.automatic_optimization = False

    def training_step(self, train_batch: Batch_Dict, batch_idx):
        opt_diffusion, opt_classifier = self.optimizers()
        opt_classifier: torch.optim.Adam
        opt_diffusion: torch.optim.Adam
        opt_diffusion.zero_grad()
        for p in self.disc.parameters():
            p.requires_grad = False
        self.diffusion_net.constants.to(self.device)
        # train_batch needs, label_A, target_A, optional_paired_B

        # normal diffusion step (either learning the output given a label or paired training)
        diffusion_dict = self.diffusion_net.training_step(train_batch, batch_idx, do_log=False, return_all=True)  # type: ignore
        loss: Tensor = diffusion_dict["loss"].mean()

        ### Class disc ###
        x_t: Tensor = diffusion_dict["x_t"]
        t: Tensor = diffusion_dict["t"]
        model_out: Tensor = diffusion_dict["model_out"]
        label: Tensor = train_batch["label"]
        x_t_next = self.diffusion_net.forward_ddim_step_training(x_t, None, t, t_next=abs(t - 1), model_out=model_out)
        pred = self.disc(x_t_next)
        loss_class_dif = self.criterion_class(pred, label)
        ### Cycle ###
        if self.current_epoch > 10:
            x_0: Tensor = train_batch["target"]
            condition: Tensor = train_batch["condition"]  # first dim zeros last 3 dimes space embedding
            label: Tensor = train_batch["label"]  # type: ignore # first dim zeros last 3 dimes space embedding
            self.buffer_target.append(x_0.detach().cpu())
            self.buffer_condition.append(condition.detach().cpu())
            self.buffer_labels.append(label.detach().cpu())
            loss_cycle = self.training_cycle(batch_idx)
        else:
            loss_cycle = 0

        loss_all = loss_cycle + loss + loss_class_dif
        self.log("train_diffusion", loss.detach().cpu().item())
        self.log("train_class_dif", loss_class_dif.detach().cpu().item())
        self.log("train_All", loss_all.detach().cpu().item())
        self.manual_backward(loss_all)
        opt_diffusion.step()
        opt_classifier.zero_grad()
        for p in self.disc.parameters():
            p.requires_grad = True
        pred = self.disc(x_t)
        loss_class_disc = self.criterion_class(pred, label)
        self.log("train_class_disc", loss_class_disc.detach().cpu().item())
        self.manual_backward(loss_class_disc)
        opt_classifier.step()

        # b = x_0.shape[0]
        # Pick output intermediate label
        # rand_label = torch.randint(0, self.opt.conditional_label_size, (b,), device=x_0.device).long()
        # Diffusion Cycle consistency loop
        # t = torch.randint(1, self.diffusion_net.num_timesteps, (b,), device=x_0.device).long()
        # t_next = t - 1
        # noise = torch.randn_like(x_0)
        # Forward | 5_1: x_t ~ sqrt(a_bar_t) x_0 + sqrt(1-alpha_bar_t)*e
        # x_t = self.diffusion_net.constants.mix_img_and_noise(x_0, noise, t)
        # condition[:, [0]] = x_0
        # x_t_next = self.diffusion_net.forward_ddim_step_training(x_t, condition, t, t_next, rand_label, eta=1, add_noise=True)
        # x_0_pred = self.diffusion_net.forward_ddim_step_training(x_t_next, condition, t - 1, t_next - 1, rand_label, eta=1, add_noise=False)
        # TODO class predictor
        # x_t_pred = self.diffusion_net.constants.mix_img_and_noise(x_0_pred, noise, t)

        # loss_cycle = self.diffusion_net.criterion_generator(x_t_pred, x_t).mean()  # img-mode

        # self.log("train_cyc", loss_cycle.detach().cpu().item())
        # try:
        # except Exception:
        #    pass

        return

    def training_cycle(self, batch_idx):

        mixing = 1  # max(0, min(1, self.current_epoch / 10 - 2))
        if len(self.buffer_target) == 11:
            with torch.no_grad():
                target = self.buffer_target.pop(0).to(self.device)
                condition = self.buffer_condition.pop(0).to(self.device)
                label = self.buffer_labels.pop(0)
                target_buffer = torch.cat(self.buffer_target, 0).to(self.device)
                target_condition = torch.cat(self.buffer_condition, 0).to(self.device)
                target_condition[:, [0]] = target_buffer
                t = self.t.detach().to(self.device)
                if self.buffer is None:
                    self.buffer = torch.rand((10,) + target_buffer.shape[1:]).to(self.device)
                    self.rand_label = torch.randint(0, self.opt.conditional_label_size, (10,), device=self.device).long()
                condition[:, [0]] = self.buffer[[0]] * mixing + target * (1 - mixing)
                self.buffer[[-1]] = torch.rand_like(self.buffer[[-1]])
                self.buffer[:-1] = self.buffer[1:].clone()
                self.rand_label[:-1] = self.rand_label[1:].clone()
                if self.opt.conditional_label_size == 2:
                    self.rand_label[[-1]] = 1 - self.buffer_labels[-1]
                else:
                    self.rand_label[[-1]] = (
                        self.buffer_labels[-1] + random.randint(1, self.opt.conditional_label_size - 1)
                    ) % self.opt.conditional_label_size
                # torch.randint(0, self.opt.conditional_label_size, (1,), device=self.device).long()

                self.buffer = self.diffusion_net.forward_ddim_step_training(
                    self.buffer, target_condition, t[1:], t[:-1], self.rand_label, eta=1, add_noise=True
                )
            batch = {"target": target.detach(), "condition": condition.to(self.device).detach(), "label": label.to(self.device).detach()}
            # TODO Buffer and batching

            loss: Tensor = self.diffusion_net.training_step(batch, batch_idx, do_log=False).mean()  # type: ignore
            self.log("train_cyc", loss.detach().cpu().item())
            return loss
        return 0

    def configure_optimizers(self):
        self.opt.lr = self.learning_rate
        optimizer_D = torch.optim.Adam(self.disc.parameters(), lr=self.opt.lr)  # default betas

        return self.diffusion_net.configure_optimizers(), optimizer_D

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
                label.shape[0],
                x_conditional=x_conditional,
                label=label,
            ).cpu()  # type: ignore
            raise NotImplementedError()
        else:
            generated: Tensor = self.diffusion_net.forward_ddim(
                label.shape[0],
                intermediate=list(range(0, self.diffusion_net.num_timesteps, min(10, self.diffusion_net.num_timesteps))),
                x_conditional=x_conditional,
                label=label,
                w=0,
            )
            generated = generated[0].cpu()  # type: ignore
            x_conditional2 = x_conditional.clone()
            x_conditional2[:, [0]] = target
            generated2: Tensor = self.diffusion_net.forward_ddim(
                label.shape[0],
                intermediate=list(range(0, self.diffusion_net.num_timesteps, min(10, self.diffusion_net.num_timesteps))),
                x_conditional=x_conditional2,
                label=1 - label,  # TODO
                w=0,
            )
            generated2 = generated2[0].cpu()  # type: ignore
        generated = torch.clamp(generated, 0, 1)
        generated2 = torch.clamp(generated2, 0, 1)
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
        if self.opt.volumes:
            generated = generated.swapaxes_(-1, -2).reshape((-1, 1, generated.shape[-2], generated.shape[-1]))
            generated2 = generated2.swapaxes_(-1, -2).reshape((-1, 1, generated.shape[-2], generated.shape[-1]))
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
            stacked_images = [target, generated, generated2]
            stacked_images += generated_other
            if x_conditional is not None:
                x_conditional = self.denormalize(x_conditional.cpu(), conditional=True)
                c: int = self.channels
                stacked_images += [x_conditional[:, i : i + c].cpu() for i in range(0, x_conditional.shape[1], c)]
            if opt.transpose_preview:
                stacked_images = [i.transpose(-1, -2) for i in stacked_images]
            try:
                grid = torchvision.utils.make_grid(
                    torch.cat(stacked_images, dim=-1).cpu(),
                    nrow=2 if not self.opt.volumes else 3,
                )
                tb_logger.experiment.add_image("conditional image", grid, self.counter)
            except Exception:
                print([s.shape for s in stacked_images])

                exit()

    def denormalize(self, x, conditional=True):
        return self.diffusion_net.denormalize(x, conditional)


def main(opt: arguments.Diffusion_Option, limit_train_batches=1):
    #### Define dataset ####
    from torch.utils.data import DataLoader
    from utils.Wrapper_datasets import Wrapper_Label2Image
    from dataloader_3D import SameSpace_3D_Dataset
    import pandas as pd

    # df = pd.read_csv("/media/data/robert/datasets/spinegan_T2w_all_reg_iso/train.csv")
    # df["Path"] = df["name"].apply(lambda x: str(Path("/media/data/robert/datasets/spinegan_T2w_all_reg_iso/registration", x)))
    # df["T2w"] = df["Path"].apply(lambda x: list(Path(x).glob("*_dixon.nii.gz"))[0].name)
    # df["CT"] = df["Path"].apply(lambda x: list(Path(x).glob("*_ct.nii.gz"))[0].name)
    # df = df[["CT", "T2w", "Path"]]
    # dfs = [df]
    # for i in range(19):
    #    df2 = df.copy()
    #    df2["Path"] = df["Path"].apply(lambda x: str(x).replace("/registration", f"/registration_deformed/{i}"))
    #    dfs.append(df2)
    # df = pd.concat(dfs).reset_index()[["CT", "T2w", "Path"]]
    # df.to_excel("/media/data/robert/datasets/spinegan_T2w_all_reg_iso/train.xlsx")
    # print(df.head())
    # exit()
    train_pd = pd.read_excel("/media/data/robert/datasets/spinegan_T2w_all_reg_iso/train.xlsx")
    #
    train_ds = Wrapper_Label2Image(
        SameSpace_3D_Dataset(
            train_pd,
            opt.target_patch_shape,
            flip=opt.flip,
            train=True,
            unpaired_mode=True,
        ),
        2,
        is_image_conditional=True,
        argumentation="reconstruction",
    )
    val_pd = pd.read_excel("/media/data/robert/datasets/spinegan_T2w_all_reg_iso/val.xlsx")
    val_ds = Wrapper_Label2Image(
        SameSpace_3D_Dataset(
            val_pd,
            opt.target_patch_shape,
            train=False,
            unpaired_mode=True,
        ),
        2,
        is_image_conditional=True,
        argumentation="reconstruction",
    )
    # exit()
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

    model = StarDiffusion(opt=opt)

    # Get last checkpoint. If there is non or --new was called this returns None and starts a new model.
    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name=opt.log_dir, best=False)
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
    opt.experiment_name = "starDif_" + opt.experiment_name
    if opt.target_patch_shape == None:
        opt.target_patch_shape = [16, 128, 128]
        opt.volumes = True
    assert opt.conditional
    # opt.new = True
    return opt


if __name__ == "__main__":
    main(get_opt())
