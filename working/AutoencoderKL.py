# Source: https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/autoencoder.py
from __future__ import annotations
from typing import TypedDict

from models.autoencoder_model import Encoder, Decoder
from models.autoencoder_kl import DiagonalGaussianDistribution
from models.autoencoder_loss import LPIPSWithDiscriminator
import pytorch_lightning as pl
import torch
from typing import TypedDict
import torch.nn.functional as F
from loader import arguments

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from loader.arguments import AutoencoderKL_Option


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        opt: arguments.AutoencoderKL_Option,
        ckpt_path=None,
        ignore_keys=[],
        monitor="val/rec_loss",
    ):

        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.image_key = opt.image_key
        self.encoder = Encoder(**opt.__dict__)
        self.decoder = Decoder(**opt.__dict__)
        self.loss = LPIPSWithDiscriminator(
            disc_start=0, kl_weight=0.000001, disc_weight=0.5, disc_in_channels=opt.out_ch
        )  # TODO settings
        self.quant_conv = torch.nn.Conv2d(2 * opt.z_channels, 2 * opt.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(opt.embed_dim, opt.z_channels, 1)
        self.embed_dim = opt.embed_dim
        if opt.colorize_n_labels is not None:
            self.register_buffer("colorize", torch.randn(3, opt.colorize_n_labels, 1, 1))

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(ik)
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+log_var
            ae_loss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log("ae_loss", ae_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return ae_loss

        elif optimizer_idx == 1:
            # train the discriminator
            disc_loss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log("disc_loss", disc_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return disc_loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        ae_loss, log_dict_ae = self.loss(
            inputs, reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split="val"
        )

        disc_loss, log_dict_disc = self.loss(
            inputs, reconstructions, posterior, 1, self.global_step, last_layer=self.get_last_layer(), split="val"
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.opt.lr
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            x_rec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert x_rec.shape[1] > 3
                x = self.to_rgb(x)
                x_rec = self.to_rgb(x_rec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = x_rec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))  # type: ignore
        x = F.conv2d(x, weight=self.colorize)  # type: ignore
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


def main(opt: AutoencoderKL_Option, limit_train_batches=1.0):
    #### Define transformation. ####

    #### Define dataset ####
    dataset, in_channel = load_dataset.getDataset(opt, True)
    dataset_val, _ = load_dataset.getDataset(opt, False)
    train_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_cpu,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=opt.batch_size,
        num_workers=opt.num_cpu,
        shuffle=False,
        drop_last=True,
    )
    assert opt.in_channels == in_channel, "Dataset and model channel miss match"
    assert opt.out_ch == in_channel, "Dataset and model channel miss match"
    # Get information from the dataset
    if hasattr(dataset, "normalize"):
        setattr(opt, "normalize", dataset.normalize)
        print("normalize:", opt.normalize)
    else:
        print("normalize: default")
    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name="logs_diffusion", best=True)

    model = AutoencoderKL(opt, ckpt_path=last_checkpoint if not opt.new else None, ignore_keys=[])

    # Define Last and best Checkpoints to be saved.
    mc_best = ModelCheckpoint(
        filename="{epoch}-{step}-{train_All:.4f}_best",
        monitor="ae_loss_step",
        mode="min",
        save_top_k=5,
        verbose=False,
        save_on_train_epoch_end=True,
    )
    mc_last = ModelCheckpoint(
        filename="{epoch}-{step}-{train_All:.4f}_latest",
        monitor="step",
        mode="max",
        every_n_train_steps=min(200, len(dataset)),
        save_top_k=2,
    )
    from models.autoencoder_kl import ImageLogger

    print(opt)
    # This sets the experiment name. The model is in /lightning_logs/{opt.exp_nam}/version_*/checkpoints/
    logger = TensorBoardLogger("logs_diffusion", name=opt.exp_name, default_hp_metric=False)
    # training
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
        callbacks=[mc_last, mc_best, ImageLogger(500, 10, increase_log_steps=False)],
        # resume_from_checkpoint=last_checkpoint,
        detect_anomaly=opt.prevent_nan,
        # amp_backend="apex",
        # amp_level="O3",
        benchmark=True,
        accumulate_grad_batches=2,
    )
    trainer.tune(model)
    trainer.fit(
        model,
        train_loader,
        val_loader,
    )  # ckpt_path=last_checkpoint


from loader import load_dataset


def get_opt(config=None) -> AutoencoderKL_Option:
    torch.cuda.empty_cache()

    opt = arguments.parseTrainParam(config=config)
    opt = arguments.parseAE(opt)

    load_dataset.parseParam_datasets(opt)
    # TODO add here model specific parameters
    # opt = arguments.parseParam_cyk(opt)
    opt = AutoencoderKL_Option.from_kwargs(**opt.parse_args().__dict__)
    return opt


if __name__ == "__main__":
    opt = get_opt()
    # opt.image_dropout = 0
    main(opt)
