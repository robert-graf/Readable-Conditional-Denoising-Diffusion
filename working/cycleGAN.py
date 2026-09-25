# %reload_ext tensorboard
# %tensorboard --logdir lightning_logs/
import glob
import itertools
from collections import deque

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure

import utils.make_snap as snap
from dataloader.Wrapper_datasets import Batch_Dict
from factory import arguments, load_dataset

# from torchmetrics import StructuralSimilarityIndexMeasure  # type: ignore
from models.cut_model import Generator
from models.patchnce import PatchNCELoss
from models.PatchSampleF import PatchSampleF
from utils.utils_cut import LambdaLR, weights_init_normal


class Optimizer(object):
    def __init__(self, pl_module: pl.LightningModule, opt: torch.optim.Adam):
        self.pl_module = pl_module
        self.opt = opt

    def __enter__(self):
        self.pl_module.toggle_optimizer(self.opt)

    def __exit__(self, *args):
        self.opt.step()
        self.opt.zero_grad()
        self.pl_module.untoggle_optimizer(self.opt)


class CycleGAN(pl.LightningModule):
    def __init__(self, opt: arguments.CycleGAN_Option, in_channel) -> None:
        """Implementation of a unpaired/paired GAN. Currently it is not possible to mix this. (use_paired)"""
        super().__init__()
        self.save_hyperparameters()
        self.logger: TensorBoardLogger

        self.opt = opt
        ### MODE SELECT ###
        self.use_contrastive = opt.mode == "cut" or opt.mode == "paired_cut"
        self.use_paired = opt.mode == "pix2pix" or opt.mode == "paired_cut"

        #### Initialize Models ####
        # from loader.load_dmodel import load_generator
        from factory.load_model_cut import load_discriminator, load_generator

        self.channels = in_channel
        self.gan_A2B = load_generator(opt, in_channel)
        self.gan_B2A = load_generator(opt, in_channel, reverse_gan=True)
        self.discriminator_A = load_discriminator(opt, in_channel, self.use_paired)
        self.discriminator_B = load_discriminator(opt, in_channel, self.use_paired, input_disc=True)

        #### Initial Weights ####
        self.gan_A2B.apply(weights_init_normal)
        self.discriminator_A.apply(weights_init_normal)
        self.gan_B2A.apply(weights_init_normal)
        self.discriminator_B.apply(weights_init_normal)

        #### Losses ####
        # Using LSGAN variants hardcoded ([vanilla| lsgan | wgangp]) TODO test Wasserstein GANs
        self.criterion_GAN = torch.nn.MSELoss()

        # if self.use_paired:
        self.criterion_paired = torch.nn.L1Loss()
        ##### contrastive loss ######
        if self.use_contrastive:
            raise NotImplementedError()
            self.nce_layers = [int(i) for i in self.opt.nce_layers.split(",")]
            # normal | xavier | kaiming | orthogonal
            self.patch_SampleF_MLP = PatchSampleF(use_mlp=True, init_type="normal", init_gain=0.02, nc=opt.netF_nc)
            self.criterion_NCE = [PatchNCELoss(opt).to(self.device) for _ in self.nce_layers]
            # A janky way to initialize but this was given to me...
            with torch.no_grad():
                if opt.lambda_NCE > 0.0:
                    a = torch.zeros((opt.batch_size, in_channel, opt.size, opt.size), device=self.device)
                    _ = self.calculate_NCE_loss(a, a)
                if opt.nce_idt and opt.lambda_NCE > 0.0:
                    a = torch.zeros((opt.batch_size, in_channel, opt.size, opt.size), device=self.device)
                    _ = self.calculate_NCE_loss(a, a)

        self.counter = 0
        self.automatic_optimization = False
        self.q = deque(maxlen=1000)

    def forward(self, x: Tensor) -> Tensor:
        print(x.shape, self.opt.volumes)
        return self.gan_A2B(x)

    def configure_optimizers(self) -> tuple[list[torch.optim.Adam], list[torch.optim.lr_scheduler.LambdaLR]]:
        opt = self.opt
        assert opt != None
        if self.use_contrastive:
            raise NotImplementedError()
            para = itertools.chain(self.gan_A2B.parameters(), self.gan_B2A.parameters(), self.patch_SampleF_MLP.parameters())
        else:
            para = itertools.chain(self.gan_A2B.parameters(), self.gan_B2A.parameters())
        optimizer_G_A2B = torch.optim.Adam(para, lr=self.opt.lr, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(self.discriminator_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(self.discriminator_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        if opt.decay_epoch == -1:
            setattr(opt, "decay_epoch", opt.max_epochs // 2)
        lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G_A2B, lr_lambda=LambdaLR(opt.max_epochs, opt.start_epoch, opt.decay_epoch).step
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=LambdaLR(opt.max_epochs, opt.start_epoch, opt.decay_epoch).step
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=LambdaLR(opt.max_epochs, opt.start_epoch, opt.decay_epoch).step
        )

        return [
            optimizer_G_A2B,
            optimizer_D_A,
            optimizer_D_B,
        ], [
            lr_scheduler_G_A2B,
            lr_scheduler_D_A,
            lr_scheduler_D_B,
        ]

    def training_step(self, train_batch: Batch_Dict, batch_idx):
        opt_g, opt_d_A, opt_d_B = self.optimizers()  # type: ignore
        opt_g: torch.optim.Adam
        opt_d_A: torch.optim.Adam
        opt_d_B: torch.optim.Adam
        #### Get batch ####
        real_A = train_batch["condition"]
        real_B = train_batch["target"]

        assert real_A is not None
        assert (
            real_B.shape[1] == opt.output_channels
        ), f"real_a and output_channels are unequal. This feature is not supported every were. Shape:{real_A.shape}, output_channels = {opt.output_channels}"
        #### In case of multiple optimizer fork ###
        # Compute forward and loss. Log loss. return one loss value.
        with Optimizer(self, opt_g):
            loss1, fake_B = self.training_step_G(self.gan_A2B, self.discriminator_B, real_A, real_B, "A2B")
            loss2, fake_A = self.training_step_G(self.gan_B2A, self.discriminator_A, real_B, real_A, "B2A")
            loss_cycle = self.criterion_paired(self.gan_A2B(fake_A), real_B) + self.criterion_paired(self.gan_B2A(fake_B), real_A)
            loss: Tensor = loss1 + loss2 + opt.lambda_cycle_consistency * loss_cycle
            self.q.append(loss.detach().cpu())
            self.manual_backward(loss)
            self.log("train/All", loss.detach().cpu())
            self.log("train/All_avg", sum(self.q) / len(self.q), prog_bar=True)

        # Compute forward and loss. Log loss. return one loss value.
        with Optimizer(self, opt_d_B):
            loss = self.training_step_D(self.discriminator_B, real_B, real_A, fake_B.detach(), "B")
            self.manual_backward(loss)

        # Compute forward and loss. Log loss. return one loss value.
        with Optimizer(self, opt_d_A):
            loss = self.training_step_D(self.discriminator_A, real_A, real_B, fake_A.detach(), "A")
            self.manual_backward(loss)

    def training_step_G(self, gan, discriminator, real_A: Tensor, real_B: Tensor, name):
        opt = self.opt
        fake_B: Tensor = gan(real_A)
        # First, G(A) should fake the discriminator
        ZERO = torch.zeros(1, device=self.device)
        loss_G_GAN = ZERO
        if self.use_paired:
            fake = torch.cat([fake_B, real_A], dim=1)
        else:
            fake = fake_B

        if opt.lambda_GAN > 0.0:
            pred_fake: Tensor = discriminator(fake)
            real_label = torch.ones((pred_fake.shape[0], 1), device=self.device)
            loss_G_GAN = self.criterion_GAN(pred_fake, real_label) * opt.lambda_GAN
        loss_NCE_both = 0
        if self.use_contrastive:
            if real_A.shape == real_B:
                idt: Tensor = gan(real_B)
            else:
                idt: Tensor = gan(torch.cat([real_B, real_A[:, 1:]], 1))

            loss_NCE = ZERO
            if opt.lambda_NCE > 0.0:
                loss_NCE = self.calculate_NCE_loss(real_A, fake_B)

            loss_NCE_both = loss_NCE
            loss_NCE_Y = ZERO
            if opt.nce_idt and opt.lambda_NCE > 0.0:
                loss_NCE_Y = self.calculate_NCE_loss(real_B, idt)
                loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
            self.log(f"train/GAN_{name}", loss_G_GAN.detach() / opt.lambda_GAN)
            self.log(f"train/NCE_{name}", loss_NCE.detach() / opt.lambda_NCE)
            self.log(f"train/NCE_Y_{name}", loss_NCE_Y.detach() / opt.lambda_NCE)

        loss_paired = 0
        loss_ssim = 0
        if self.use_paired:
            loss_paired = self.criterion_paired(real_B, fake_B)
            self.log(f"train/loss_paired_{name}", loss_paired.detach())
            if opt.lambda_ssim > 0.0:
                loss_ssim = opt.lambda_ssim * (
                    1 - structural_similarity_index_measure(real_B + 1, fake_B + 1, data_range=2.0)
                )  # type: ignore
                self.log(f"train/loss_ssim_{name}", loss_ssim.detach())
            loss_paired = opt.lambda_paired * (loss_ssim + loss_paired)
        if opt.lambda_cycle_identity != 0:
            assert (
                real_B.shape[1] == real_A.shape
            ), f"real_a and real_A are unequal in shape. lambda_cycle_identity is not supported every were. Shape:{real_A.shape}, output_channels = {opt.output_channels}"

            # F_a2b(B)->B
            fake_id = gan(real_B)
            loss_id = self.criterion_paired(real_B, fake_id)
            loss_paired += opt.lambda_cycle_identity * loss_id
        loss_G = loss_G_GAN + loss_NCE_both + loss_paired

        return loss_G, fake_B

    def training_step_D(self, discriminator, real_inp, other_img, fake, name) -> Tensor:
        # Fake loss, will be fake_B if unpaired and fake_B||real_A if paired
        pred_fake = discriminator(fake)
        fake_label = torch.zeros((pred_fake.shape[0], 1), device=self.device)
        loss_D_fake = self.criterion_GAN(pred_fake, fake_label).mean()  # is mean really necessary?

        # Real loss
        if self.use_paired:
            real = torch.cat([real_inp, other_img], dim=1)
        else:
            real = real_inp

        pred_real = discriminator(real)
        assert not np.any(np.isnan(pred_real.detach().cpu().numpy()))  # type: ignore
        real_label = torch.ones((pred_real.shape[0], 1), device=self.device)
        loss_D_real = self.criterion_GAN(pred_real, real_label).mean()
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        self.log(f"train/D_real_{name}", loss_D_real.detach())
        self.log(f"train/D_fake_{name}", loss_D_fake.detach())
        return loss_D

    def calculate_NCE_loss(self, src, tgt) -> Tensor:
        raise NotImplementedError()
        n_layers = len(self.nce_layers)
        feat_q = self.forward_GAN_with_Intermediate(tgt, self.nce_layers)
        feat_k = self.forward_GAN_with_Intermediate(src, self.nce_layers)
        feat_k_pool, sample_ids = self.patch_SampleF_MLP(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.patch_SampleF_MLP(feat_q, self.opt.num_patches, sample_ids)
        ZERO = torch.zeros(1, device=self.device)
        total_nce_loss = ZERO
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterion_NCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def forward_GAN_with_Intermediate(self, input, target_layers) -> list[Tensor]:
        raise NotImplementedError()
        if isinstance(self.gan, Generator):  # self.opt.model_name == "resnet"
            if -1 in target_layers:
                target_layers.append(len(self.gan.model))
            assert len(target_layers)

            partial_forward = input
            feats = []
            for layer_id, layer in enumerate(self.gan.model):
                partial_forward = layer(partial_forward)
                if layer_id in target_layers:
                    feats.append(partial_forward)
                else:
                    pass
            return feats
        else:
            _, features = self.gan(input, return_intermediate=True, layers=target_layers)
            return features

    @torch.no_grad()
    def validation_step(self, batch: Batch_Dict, batch_idx):
        real_A = batch["condition"]
        assert real_A is not None

        real_B = batch["target"]
        print(real_B.shape, real_A.shape)
        assert (
            real_B.shape[1] == opt.output_channels
        ), f"real_a and output_channels are unequal. This feature is not supported every were. Shape:{real_A.shape}, output_channels = {opt.output_channels}"
        fake_B: Tensor = self.gan_A2B(real_A)
        fake_A: Tensor = self.gan_B2A(real_B)

        if self.opt.volumes:
            out = (
                [real_B[:, [i]] for i in range(real_B.shape[1])]
                + [fake_B[:, [i]] for i in range(fake_B.shape[1])]
                + [real_A[:, [i]] for i in range(real_A.shape[1])]
                + [fake_A[:, [i]] for i in range(fake_A.shape[1])]
            )  # , fake_id, real_B
            out = [denormalize(i) for i in out]

            # print([a.shape for a in out])
            stacked_images = torch.stack(out, dim=-1).cpu()
            stacked_images = stacked_images[:, :, ::4]
            stacked_images = stacked_images.swapaxes(1, 2)
            stacked_images = stacked_images.reshape((-1,) + stacked_images.shape[-3:])
        else:
            out = []
            for j in range(real_B.shape[0]):
                out += (
                    [real_B[[j], [i]] for i in range(real_B.shape[1])]
                    + [fake_B[[j], [i]] for i in range(fake_B.shape[1])]
                    + [real_A[[j], [i]] for i in range(real_A.shape[1])]
                    + [fake_A[[j], [i]] for i in range(fake_A.shape[1])]
                )  # , fake_id, real_B
            out = [denormalize(i) for i in out]

            stacked_images = torch.stack(out, dim=0).cpu()
            print(stacked_images.shape)
            stacked_images = stacked_images.reshape((-1,) + stacked_images.shape[-3:])
            print(stacked_images.shape)
        # print(grid.shape)
        output_rows = 1
        try:
            st = stacked_images.cpu()
            text = [
                f"GT {opt.condition_types[0]}",
                f"Predicted {opt.condition_types[0]}",
                *(opt.condition_types[1:]),
                *("Predicted" + a for a in opt.condition_types[1:]),
            ]
            for i in range(output_rows - 1):
                text += text
            stacked_images = snap.make_grid_with_labels(st, text, nrow=len(text))
            self.logger.experiment.add_image("A2B", stacked_images, self.counter)
        except Exception as e:
            print(e)
            grid2 = torchvision.utils.make_grid(stacked_images, nrow=output_rows)
            self.logger.experiment.add_image("A2B", grid2, self.counter)
        self.counter += 1


def normalize(tensor) -> Tensor:
    return (tensor * 2) - 1  # map [0,1]->[-1,1]


def denormalize(tensor) -> Tensor:
    return torch.clamp((tensor + 1) / 2, 0, 1)  # map [-1,1]->[0,1]


def reload(name, version="*", best=False):
    if best:
        checkpoints = sorted(glob.glob(f"lightning_logs/{name}/version_{version}/checkpoints/*best*.ckpt"))
    else:
        checkpoints = sorted(glob.glob(f"lightning_logs/{name}/version_{version}/checkpoints/*.ckpt"))
    assert len(checkpoints) != 0, "Did not find the requested network."
    return CycleGAN.load_from_checkpoint(checkpoints[-1])


def main(opt: arguments.CycleGAN_Option, limit_train_batches=128):
    #### Define dataset ####
    opt.exp_name = "cGAN_" + opt.exp_name
    setattr(opt, "num_validation_images", max(opt.batch_size, 16))
    dataset, in_channel = load_dataset.getDataset(opt, True, compute_mean=False)
    dataset_test, _ = load_dataset.getDataset(opt, False, compute_mean=False)
    train_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_cpu,
        shuffle=True,
        drop_last=True,
        # pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    from dataloader.dataloader_basic import DummyDataset

    dummy_val_loader = DataLoader(
        DummyDataset(num_sample=opt.num_validation_images, ds=dataset_test),
        batch_size=opt.num_validation_images,
        num_workers=opt.num_cpu,
        drop_last=False,
    )

    # Get information from the dataset
    setattr(opt, "conditional", True)
    conditional_channel_size = dataset.get_conditional_channel_size()
    if conditional_channel_size == -1:
        conditional_channel_size = in_channel
    setattr(opt, "conditional_channel_size", conditional_channel_size)
    if dataset.has_label():
        setattr(opt, "conditional_label_size", dataset.get_label_count())
    if dataset.has_embedding():
        setattr(opt, "conditional_embedding_size", dataset.get_embedding_size())
    model = CycleGAN(opt, in_channel)
    # Get last checkpoint. If there is non or --new was called this returns None and starts a new model.
    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name="logs_diffusion3D", best=True)

    # Define Last and best Checkpoints to be saved.

    mc_best = ModelCheckpoint(
        filename="{epoch}-{step}-{train_All:.4f}_best",
        monitor="train/All",
        mode="min",
        save_top_k=2,
        verbose=False,
        save_on_train_epoch_end=True,
    )
    mc_last = ModelCheckpoint(
        filename="{epoch}-{step}_{train_All}_latest",  # {train_All:.4}
        monitor="step",
        mode="max",
        every_n_train_steps=min(500, len(dataset)),
        save_top_k=3,
    )
    # print(opt)
    opt.print()
    # This sets the experiment name. The model is in /lightning_logs/{opt.exp_nam}/version_*/checkpoints/
    logger = TensorBoardLogger("logs_diffusion3D", name=opt.exp_name, default_hp_metric=False)
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
        # gpus=gpus,  # torch.cuda.device_count(),
        # num_nodes=num_nodes,  # Train on 'n' GPUs; 0 is CPU
        limit_train_batches=limit_train_batches if limit_train_batches != 1.0 else None,
        # Train only x % (if float) or train only on x batches (if int)
        max_epochs=opt.max_epochs,  # Stopping epoch
        logger=logger,
        callbacks=[mc_last, mc_best],
        detect_anomaly=opt.prevent_nan
        # progress_bar_refresh_rate=0
        # detect_anomaly=True,
    )

    trainer.fit(
        model,
        train_loader,
        dummy_val_loader,
        ckpt_path=last_checkpoint,
    )


def get_opt(config=None):
    torch.cuda.empty_cache()
    opt = arguments.CycleGAN_Option().get_opt(None, config)
    opt = arguments.CycleGAN_Option.from_kwargs(**opt.parse_args().__dict__)
    if not hasattr(opt, "output_channels"):
        opt.output_channels = 1
    return opt


if __name__ == "__main__":
    opt = get_opt()
    if opt.lambda_paired != 0:
        if "y" in input("You currently would run in paired mode! Would you switch to unpaired mode? (y/n)").lower():
            opt.lambda_paired = 0
            opt.mode = "cycleGAN"
    if opt.lambda_cycle_identity != 0:
        if len(opt.condition_types) != 2 * opt.output_channels:
            input(
                "lambda_cycle_identity only functions if input and output have the same number of channels? The lambda_cycle_identity will be set to 0 (OK)"
            )
            opt.lambda_cycle_identity = 0
    # opt.image_dropout = 0
    main(opt, limit_train_batches=1)
