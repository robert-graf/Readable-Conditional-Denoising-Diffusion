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

from dataloader.Wrapper_datasets import Batch_Dict
from factory import arguments, load_dataset

# from torchmetrics import StructuralSimilarityIndexMeasure  # type: ignore
from models.cut_model import Generator
from models.patchnce import PatchNCELoss
from models.PatchSampleF import PatchSampleF
from utils.utils_cut import LambdaLR, weights_init_normal


class CUT(pl.LightningModule):
    def __init__(self, opt: arguments.CycleGAN_Option, in_channel) -> None:
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
        self.gan = load_generator(opt, in_channel)
        self.discriminator = load_discriminator(opt, in_channel, self.use_paired)

        #### Initial Weights ####
        self.gan.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        #### Losses ####
        # Using LSGAN variants hardcoded ([vanilla| lsgan | wgangp]) TODO test Wasserstein GANs
        self.criterion_GAN = torch.nn.MSELoss()

        if self.use_paired:
            self.criterion_paired = torch.nn.L1Loss()
        ##### contrastive loss ######
        if self.use_contrastive:
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
        return self.gan(x)

    def configure_optimizers(self) -> tuple[list[torch.optim.Adam], list[torch.optim.lr_scheduler.LambdaLR]]:
        opt = self.opt
        assert opt != None
        if self.use_contrastive:
            para = itertools.chain(self.gan.parameters(), self.patch_SampleF_MLP.parameters())
        else:
            para = self.gan.parameters()
        optimizer_G = torch.optim.Adam(para, lr=self.opt.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        if opt.decay_epoch == -1:
            setattr(opt, "decay_epoch", opt.max_epochs // 2)
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=LambdaLR(opt.max_epochs, opt.start_epoch, opt.decay_epoch).step
        )
        lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D, lr_lambda=LambdaLR(opt.max_epochs, opt.start_epoch, opt.decay_epoch).step
        )

        return [optimizer_G, optimizer_D], [lr_scheduler_G, lr_scheduler_D]

    def training_step(self, train_batch: Batch_Dict, batch_idx):
        opt_g, opt_d = self.optimizers()  # type: ignore
        opt_g: torch.optim.Adam
        opt_d: torch.optim.Adam
        #### Get batch ####
        real_A = train_batch["condition"]
        real_B = train_batch["target"]

        assert real_A is not None
        assert (
            real_B.shape[1] == opt.output_channels
        ), f"real_a and output_channels are unequal. This feature is not supported every were. Shape:{real_A.shape}, output_channels = {opt.output_channels}"
        #### In case of multiple optimizer fork ###
        # Compute forward and loss. Log loss. return one loss value.
        opt_g.zero_grad()
        loss = self.training_step_G(real_A, real_B, batch_idx)
        self.manual_backward(loss)
        opt_g.step()

        # Compute forward and loss. Log loss. return one loss value.
        opt_d.zero_grad()
        loss = self.training_step_D(real_A, real_B)
        self.manual_backward(loss)
        opt_d.step()

    def training_step_G(self, real_A: Tensor, real_B: Tensor, batch_idx):
        opt = self.opt
        fake_B: Tensor = self.gan(real_A)
        # First, G(A) should fake the discriminator
        ZERO = torch.zeros(1, device=self.device)
        loss_G_GAN = ZERO
        if self.use_paired:
            fake = torch.cat([fake_B, real_A], dim=1)
        else:
            fake = fake_B

        if opt.lambda_GAN > 0.0:
            pred_fake: Tensor = self.discriminator(fake)
            real_label = torch.ones((pred_fake.shape[0], 1), device=self.device)
            loss_G_GAN = self.criterion_GAN(pred_fake, real_label) * opt.lambda_GAN
        loss_NCE_both = 0
        if self.use_contrastive:
            if real_A.shape == real_B:
                idt: Tensor = self.gan(real_B)
            else:
                idt: Tensor = self.gan(torch.cat([real_B, real_A[:, 1:]], 1))

            loss_NCE = ZERO
            if opt.lambda_NCE > 0.0:
                loss_NCE = self.calculate_NCE_loss(real_A, fake_B)

            loss_NCE_both = loss_NCE
            loss_NCE_Y = ZERO
            if opt.nce_idt and opt.lambda_NCE > 0.0:
                loss_NCE_Y = self.calculate_NCE_loss(real_B, idt)
                loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
            self.log("train/GAN", loss_G_GAN.detach() / opt.lambda_GAN)
            self.log("train/NCE", loss_NCE.detach() / opt.lambda_NCE)
            self.log("train/NCE_Y", loss_NCE_Y.detach() / opt.lambda_NCE)

        loss_paired = 0
        loss_ssim = 0
        if self.use_paired:
            loss_paired = self.criterion_paired(real_B, fake_B)
            self.log("train/loss_paired", loss_paired.detach())
            if opt.lambda_ssim > 0.0:
                loss_ssim = opt.lambda_ssim * (
                    1 - structural_similarity_index_measure(real_B + 1, fake_B + 1, data_range=2.0)
                )  # type: ignore
                self.log("train/loss_ssim", loss_ssim.detach())
            loss_paired = opt.lambda_paired * (loss_ssim + loss_paired)
        self.fake_B_buffer = fake.detach()

        loss_G = loss_G_GAN + loss_NCE_both + loss_paired
        self.log("train/All", loss_G.detach().cpu())
        self.q.append(loss_G.detach().cpu())
        self.log("train_avg", sum(self.q) / len(self.q), prog_bar=True)
        return loss_G

    def training_step_D(self, real_A, real_B) -> Tensor:
        # Fake loss, will be fake_B if unpaired and fake_B||real_A if paired
        fake = self.fake_B_buffer
        pred_fake = self.discriminator(fake)

        assert not np.any(
            np.isnan(pred_fake.detach().cpu().numpy())  # type: ignore
        ), "NAN detected! (ʘᗩʘ'), if this happened at the start of your training, than the init is instable. Try again, or change init_type and try again."
        fake_label = torch.zeros((pred_fake.shape[0], 1), device=self.device)
        loss_D_fake = self.criterion_GAN(pred_fake, fake_label).mean()  # is mean really necessary?

        # Real loss
        if self.use_paired:
            real = torch.cat([real_B, real_A], dim=1)
        else:
            real = real_B

        pred_real = self.discriminator(real)
        assert not np.any(np.isnan(pred_real.detach().cpu().numpy()))  # type: ignore
        real_label = torch.ones((pred_real.shape[0], 1), device=self.device)
        loss_D_real = self.criterion_GAN(pred_real, real_label).mean()
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        self.log("train/D_real", loss_D_real.detach())
        self.log("train/D_fake", loss_D_fake.detach())
        return loss_D

    def calculate_NCE_loss(self, src, tgt) -> Tensor:
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

    def validation_step(self, batch: Batch_Dict, batch_idx):
        real_A = batch["condition"]
        real_B = batch["target"]
        print(real_B.shape, real_A.shape)
        assert (
            real_B.shape[1] == opt.output_channels
        ), f"real_a and output_channels are unequal. This feature is not supported every were. Shape:{real_A.shape}, output_channels = {opt.output_channels}"
        fake_B = self.gan(real_A)
        assert real_A is not None
        # fake_id = self.gan(real_B)
        out = (
            [real_B[:, [i]] for i in range(real_B.shape[1])]
            + [fake_B[:, [i]] for i in range(fake_B.shape[1])]
            + [real_A[:, [i]] for i in range(real_A.shape[1])]
        )  # , fake_id, real_B
        out = [denormalize(i) for i in out]

        # print([a.shape for a in out])
        grid = torch.cat(out, dim=-1).cpu()
        if self.opt.volumes:
            grid = grid[:, :, ::4]
        if opt.volumes:
            grid = grid.swapaxes(1, 2)
            grid = grid.reshape((-1,) + grid.shape[-3:])
        # print(grid.shape)
        grid = torchvision.utils.make_grid(grid, nrow=2)
        self.logger.experiment.add_image("A2B", grid, self.counter)
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
    return CUT.load_from_checkpoint(checkpoints[-1])


def main(opt: arguments.CycleGAN_Option, limit_train_batches=128):
    #### Define dataset ####
    opt.exp_name = "CUT_" + opt.exp_name
    setattr(opt, "num_validation_images", max(opt.batch_size, 16))
    dataset, in_channel = load_dataset.getDataset(opt, True, compute_mean=False)
    dataset_test, _ = load_dataset.getDataset(opt, False, compute_mean=False)
    train_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_cpu,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
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
    model = CUT(opt, in_channel)
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

    trainer.fit(model, train_loader, dummy_val_loader, ckpt_path=last_checkpoint)


def get_opt(config=None):
    torch.cuda.empty_cache()
    opt = arguments.CycleGAN_Option().get_opt(None, config)
    opt = arguments.CycleGAN_Option.from_kwargs(**opt.parse_args().__dict__)
    return opt


if __name__ == "__main__":
    opt = get_opt()
    # opt.image_dropout = 0
    main(opt, limit_train_batches=1)
