# %reload_ext tensorboard
# %tensorboard --logdir lightning_logs/


from loader import arguments
import itertools
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl

from models.cut_model import Generator, Discriminator
from models.patchnce import PatchNCELoss
from models.PatchSampleF import PatchSampleF

from utils.utils_cut import LambdaLR, ReplayBuffer, weights_init_normal, SquarePad

from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
import glob
import numpy as np
import math

from threading import Thread
from diffusion import Diffusion


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class CUT(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        #### Initialize Models ####
        # from loader. import load_generator

        self.gan = Generator(1, 1)  # load_generator(opt, version="A2B")
        self.discriminator = Discriminator(1, depth=4, channels=32)
        self.patch_SampleF_MLP = PatchSampleF(use_mlp=True, init_type="xavier", init_gain=0.02, nc=256)  # TODO GPU IDS
        # TODO self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        # TODO self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        #### Initial Weights ####
        self.gan.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        #### Buffers ####
        # self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer(max_size=100)

        #### Losses ####
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE','NCE_Y']
        setattr(self.opt, "nce_layers", "0,4,8,12,16")
        setattr(self.opt, "lambda_NCE", 1.0)
        setattr(self.opt, "num_patches", 256)
        setattr(self.opt, "nce_T", 0.07)
        setattr(self.opt, "nce_idt", False)
        setattr(self.opt, "decay_epoch", opt.max_epochs // 2)
        setattr(self.opt, "lambda_GAN", 1.0)

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(",")]
        # Using LSGAN variants hardcoded ([vanilla| lsgan | wgangp]) TODO test Wasserstein GANs
        self.criterion_GAN = torch.nn.MSELoss()

        self.criterion_Dif = torch.nn.L1Loss()
        self.criterion_NCE = [PatchNCELoss(opt).to(self.device) for _ in self.nce_layers]

        self.resnn = None
        # A janky way to initialize but this was given to me...
        with torch.no_grad():
            if opt.lambda_NCE > 0.0:
                a = torch.zeros((opt.batch_size, 1, opt.size, opt.size), device=self.device)
                _ = self.calculate_NCE_loss(a, a)
            if opt.nce_idt and opt.lambda_NCE > 0.0:
                a = torch.zeros((opt.batch_size, 1, opt.size, opt.size), device=self.device)
                _ = self.calculate_NCE_loss(a, a)
        self.counter = 0

        #### Diffusion
        self.diffusion = Diffusion(opt=opt, in_channel=1)
        self.fake_diffusion = None
        self.thread = None

    def forward(self, x):
        return self.gan(x)

    def configure_optimizers(self):
        opt = self.opt
        assert opt != None
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.gan.parameters(), self.patch_SampleF_MLP.parameters()),
            lr=self.opt.lr,
            betas=(0.5, 0.999),
        )
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=LambdaLR(opt.max_epochs, opt.start_epoch, opt.decay_epoch).step
        )
        lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D, lr_lambda=LambdaLR(opt.max_epochs, opt.start_epoch, opt.decay_epoch).step
        )
        optimizer_Diff = self.diffusion.configure_optimizers()
        return [optimizer_G, optimizer_D, optimizer_Diff, optimizer_Diff], [lr_scheduler_G, lr_scheduler_D]

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        #### Get batch ####
        real_ct = train_batch["target"]
        real_mr = train_batch["condition"]
        mask = train_batch["mask_cond"]

        #### In case of multiple optimizer fork ###
        if optimizer_idx == 0:
            # Compute forward and loss. Log loss. return one loss value.
            return self.training_step_G(real_ct, real_mr, batch_idx)
        elif optimizer_idx == 1:
            # Compute forward and loss. Log loss. return one loss value.
            return self.training_step_D(real_mr)
        elif optimizer_idx == 2:
            # TODO
            # train_batch["target"] = real_ct
            # make image_dropout
            batch_size = self.fake_b_buffered.shape[0]
            train_batch["condition"] = self.fake_b_buffered.clone()
            train_batch["condition"][: batch_size // 2] *= 0

            return self.diffusion.training_step(train_batch, batch_idx)
        elif optimizer_idx == 3:
            # TODO
            # train_batch["target"] = real_ct
            # make image_dropout
            batch_size = self.fake_b_buffered.shape[0]
            train_batch["condition"] = self.fake_b_buffered.clone()
            train_batch["condition"][batch_size // 2 :] *= 0

            return self.diffusion.training_step(train_batch, batch_idx)

    def training_step_G(self, real_A: Tensor, real_B: Tensor, batch_idx) -> Tensor:
        opt = self.opt

        fake: Tensor = self.gan(real_A)
        idt: Tensor = self.gan(real_B)
        assert not np.any(np.isnan(fake.detach().cpu().numpy()))
        assert not np.any(np.isnan(idt.detach().cpu().numpy()))
        # First, G(A) should fake the discriminator
        loss_G_GAN = 0.0
        if opt.lambda_GAN > 0.0:
            pred_fake: Tensor = self.discriminator(fake)
            assert not np.any(np.isnan(pred_fake.detach().cpu().numpy()))
            real_label = torch.ones((pred_fake.shape[0], 1), device=self.device)
            loss_G_GAN = self.criterion_GAN(pred_fake, real_label) * opt.lambda_GAN

        loss_NCE = 0.0
        if opt.lambda_NCE > 0.0:
            loss_NCE: Tensor = self.calculate_NCE_loss(real_A, fake)
            assert not np.any(np.isnan(loss_NCE.detach().cpu().numpy()))

        loss_NCE_both = loss_NCE
        loss_NCE_Y = 0.0
        if opt.nce_idt and opt.lambda_NCE > 0.0:
            loss_NCE_Y: Tensor = self.calculate_NCE_loss(real_B, idt)
            assert not np.any(np.isnan(loss_NCE_Y.detach().cpu().numpy()))
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5

        self.fake_B_buffer.push_and_pop(fake.detach())
        self.fake_b_buffered = fake.detach()

        self.log(f"train_GAN", loss_G_GAN / opt.lambda_GAN)
        self.log(f"train_NCE", loss_NCE / opt.lambda_NCE)
        self.log(f"train_NCE_Y", loss_NCE_Y / opt.lambda_NCE)
        loss_G = loss_G_GAN + loss_NCE_both  # *opt.lambda_NCE

        diffusion_out = self.ddim(real_B)
        if diffusion_out is not None:
            d_real_B, d_fake_A = diffusion_out
            loss_dif = self.criterion_Dif(self.gan(d_fake_A), d_real_B)
            loss_G += loss_dif

        self.log(f"train_All", loss_G)
        return loss_G

    def training_step_D(self, real_B):
        # Fake loss
        fake = self.fake_B_buffer.peek()
        pred_fake = self.discriminator(fake)
        assert not np.any(np.isnan(pred_fake.detach().cpu().numpy()))
        fake_label = torch.zeros((pred_fake.shape[0], 1), device=self.device)
        loss_D_fake = self.criterion_GAN(pred_fake, fake_label).mean()  # is mean really necessary?
        # Real loss
        pred_real = self.discriminator(real_B)
        assert not np.any(np.isnan(pred_real.detach().cpu().numpy()))
        real_label = torch.ones((pred_real.shape[0], 1), device=self.device)
        loss_D_real = self.criterion_GAN(pred_real, real_label).mean()
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        self.log(f"train_D_real", loss_D_real)
        self.log(f"train_D_fake", loss_D_fake)
        return loss_D

    def ddim(self, real_B):
        out = None
        if self.fake_diffusion is not None:
            self.fake_B_buffer.forced_push_info(self.fake_diffusion.detach().cuda())
            out = self.real_diffusion, self.fake_diffusion
            self.real_diffusion = None
            self.fake_diffusion = None
            self.thread.join(timeout=60)
            self.thread = None
            print(end="\r")
            return out
        if self.thread is None:

            def call_ddim(self, real_B):
                self.real_diffusion = real_B
                out, out_list = self.diffusion.forward_ddim(
                    real_B.shape[0], range(0, 1000, 50), 1.0, real_B, progressbar=False
                )
                self.fake_diffusion = self.diffusion.normalize(out)

            self.thread = Thread(target=call_ddim, args=(self, real_B), daemon=True)
            self.thread.start()

        return None

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.forward_GAN_with_Intermediate(tgt, self.nce_layers)
        feat_k = self.forward_GAN_with_Intermediate(src, self.nce_layers)
        feat_k_pool, sample_ids = self.patch_SampleF_MLP(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.patch_SampleF_MLP(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterion_NCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def forward_GAN_with_Intermediate(self, input, target_layers):
        # if self.opt.model_name == "resnet":
        if -1 in target_layers:
            target_layers.append(len(self.gan.model))
        assert len(target_layers)

        partial_forward = input
        feats = []
        for layer_id, layer in enumerate(self.gan.model):
            # print(layer_id, layer)
            partial_forward = layer(partial_forward)
            if layer_id in target_layers:
                # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                feats.append(partial_forward)
            else:
                # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                pass
        return feats
        # else:
        #    _, features = self.gan(input, return_intermediate=True)
        #    return [features[i] for i in target_layers]

    def validation_step(self, batch, batch_idx):

        real_ct = batch["target"]
        real_mr = batch["condition"]
        fake = self.gan(real_ct)

        grid = torchvision.utils.make_grid(torch.cat([real_ct, fake, real_mr], dim=-1).cpu(), nrow=2)
        grid += 1
        grid /= 2
        self.logger.experiment.add_image("MR2CT", grid, self.counter)
        self.counter += 1
        self.counter %= 100


def reload(name, version="*", best=False):
    if best:
        checkpoints = sorted(glob.glob(f"logs_diffusion/{name}/version_{version}/checkpoints/*best*.ckpt"))
    else:
        checkpoints = sorted(glob.glob(f"logs_diffusion/{name}/version_{version}/checkpoints/*.ckpt"))
    assert len(checkpoints != 0), "Did not find the requested network."
    return CUT.load_from_checkpoint(checkpoints[-1])


def get_transforms(opt=None, validation=False):
    #### Define transformation. ####
    if not validation:
        return [
            transforms.ToTensor(),
            SquarePad(opt.size),
            transforms.RandomCrop(opt.size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5), (0.5)),
        ]
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                # SquarePad(opt.size),
                # transforms.RandomCrop(opt.size),
                transforms.Normalize((0.5), (0.5)),
            ]
        )


def main(opt, limit_train_batches=1.0):
    #### Define transformation. ####

    #### Define dataset ####
    if opt.volumes:
        setattr(opt, "num_validation_images", 1)
    else:
        setattr(opt, "num_validation_images", 8)
    dataset, in_channel = load_dataset.getDataset(opt, True)
    dataset_test, _ = load_dataset.getDataset(opt, False)
    train_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_cpu,
        shuffle=True,
        drop_last=True,
    )
    from dataloader.dataloader_basic import DummyDataset

    dummy_val_loader = DataLoader(
        DummyDataset(num_sample=opt.num_validation_images, ds=dataset_test),
        batch_size=opt.num_validation_images,
        num_workers=4,
        drop_last=False,
    )

    # Get information from the dataset
    if opt.conditional:
        assert dataset.has_condition(), "You selected conditional, but the dataset does not support this"
        conditional_channel_size = dataset.get_conditional_channel_size()
        if conditional_channel_size == -1:
            conditional_channel_size = in_channel
        setattr(opt, "conditional_channel_size", conditional_channel_size)
    else:
        setattr(opt, "conditional_channel_size", 0)
    if dataset.has_label():
        setattr(opt, "conditional_label_size", dataset.num_classes)
    if dataset.has_embedding():
        setattr(opt, "conditional_embedding_size", dataset.embedding_size)
    if hasattr(dataset, "normalize"):
        setattr(opt, "normalize", dataset.normalize)
        print("normalize:", opt.normalize)
    else:
        print("normalize: default")
    # opt.image_channels_B = opt.image_channels_A
    model = CUT(opt)
    # Diffusion(opt=opt, in_channel=in_channel)
    # Get last checkpoint. If there is non or --new was called this returns None and starts a new model.
    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name="logs_diffusion", best=True)

    # Define Last and best Checkpoints to be saved.

    mc_best = ModelCheckpoint(
        filename="{epoch}-{step}-{train_All:.4f}_best",
        monitor="train_All",
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
    from pytorch_lightning.callbacks import Callback

    class ValidationCallback(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            pl_module = pl_module.diffusion
            pl_module.cuda()
            batch = {}
            for k, v in next(iter(dummy_val_loader)).items():
                batch[k] = v.to(pl_module.device)

            def validation_step(batch, i, logger):
                pl_module.validation_step_by_callback(batch, i, logger)
                pl_module.validation_step_by_callback(batch, i, logger)
                pl_module.validation_step_by_callback(batch, i, logger)

            # next(iter(dummy_val_loader))
            pl_module.thread = Thread(target=validation_step, args=(batch, 0, logger), daemon=True)
            pl_module.thread.start()
            print(end="\r")

        def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            pl_module = pl_module.diffusion
            if not pl_module.thread is None:
                # pl_module.thread.join(timeout=600)
                pl_module.thread = None
                print(end="\r")

    print(opt)
    # This sets the experiment name. The model is in /lightning_logs/{opt.exp_nam}/version_*/checkpoints/
    logger = TensorBoardLogger("logs_diffusion", name=opt.exp_name, default_hp_metric=False)
    # training
    trainer = pl.Trainer(
        gpus=1,  # torch.cuda.device_count(),
        num_nodes=1,  # Train on 'n' GPUs; 0 is CPU
        limit_train_batches=limit_train_batches if limit_train_batches != 1.0 else None,
        # Train only x % (if float) or train only on x batches (if int)
        max_epochs=opt.max_epochs,  # Stopping epoch
        logger=logger,
        callbacks=[mc_best, mc_last, ValidationCallback()],
        resume_from_checkpoint=last_checkpoint,
        # progress_bar_refresh_rate=0
    )
    model.diffusion.__dict__["logger"] = logger

    trainer.fit(
        model,
        train_loader,
        dummy_val_loader,
    )  # ckpt_path=last_checkpoint
    # recommended lr = 2e-5

    # CUDA_VISIBLE_DEVICES=0 python3 diffusion.py -lr 2e-5 -bs 8 -epoch 1000 -en diffusion_ct -s 192 -ds Img --root /media/data/robert/datasets/TGD_2D/ct/train
    # CUDA_VISIBLE_DEVICES=1 python3 diffusion.py -lr 2e-5 -bs 8 -epoch 1000 -en diffusion_ct -s 104 -ds cat


from loader import load_dataset


def get_opt(config=None):
    torch.cuda.empty_cache()

    opt = arguments.parseTrainParam(config=config)
    opt = arguments.parseDiffusion(opt)

    load_dataset.parseParam_datasets(opt)
    # TODO add here model specific parameters
    # opt = arguments.parseParam_cyk(opt)
    return opt.parse_args()


if __name__ == "__main__":
    opt = get_opt()
    # opt.image_dropout = 0
    main(opt)
