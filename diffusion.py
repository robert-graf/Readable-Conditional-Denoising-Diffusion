from __future__ import annotations


# %reload_ext tensorboard
# tensorboard --logdir logs_diffusion

import itertools
from warnings import warn

import torch
from torch.utils.data import DataLoader
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from threading import Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AutoencoderKL import AutoencoderKL
from dataloader.Wrapper_datasets import Batch_Dict
from utils.utils_cut import weights_init_normal

from pytorch_lightning.loggers import TensorBoardLogger

import utils.losses as losses

import time

from tqdm import tqdm

# pip install ema-pytorch
from ema_pytorch import EMA
import math
import torchvision
from typing import Callable, List, OrderedDict


from torchmetrics.functional import structural_similarity_index_measure
from utils.utils_diffusion import get_option

from utils.diffusion_network_based_guidance import network_based_guidance
from utils.utils_diffusion import Diffusions_Constance


def first(w: float, i: float) -> float:
    return w


class WarnOnlyOnce:
    warnings = set()

    @classmethod
    def warn(cls, message):
        # storing int == less memory then storing raw message
        h = hash(message)
        if h not in cls.warnings:
            # do your warning
            warn(message)
            cls.warnings.add(h)


class Diffusion(pl.LightningModule):
    def __init__(self, opt: arguments.Diffusion_Option, in_channel=1, easy_color_mode=True) -> None:
        super().__init__()
        assert opt.lambda_ssim == 0.0 or opt.image_mode, "SSIM only functions in image mode"
        self.save_hyperparameters()
        self.counter = 0

        ema_decay = 0.995
        ema_update_every = 10
        # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        self.p2_loss_weight_gamma = 0.0
        self.p2_loss_weight_k = 1

        self.num_timesteps: int = opt.timesteps
        # self.gradient_accumulate_every = 2

        self.opt = opt
        #### Initialize Models ####
        from loader.load_model_diffusion import load_model

        self.generator = load_model(opt, in_channel, self)
        self.channels = in_channel

        self.ema = EMA(self.generator, beta=ema_decay, update_every=ema_update_every)
        # self.amp = False  #Use pytorch lightning AMP instead
        # from torch.cuda.amp import GradScaler
        # self.scaler = GradScaler(enabled=self.amp) #Use pytorch lightning scaler instead

        #### Initial Weights ####
        self.generator.apply(weights_init_normal)

        #### Losses ####
        if opt.L2_loss:
            self.criterion_generator = torch.nn.MSELoss(reduction="none")
        else:  # Default
            self.criterion_generator = torch.nn.L1Loss(reduction="none")  # Claim: Less Hallucination, less diversity

        self.constants = Diffusions_Constance("linear" if opt.linear else "cosine", timesteps=self.num_timesteps)
        # The validation is costume multithreaded
        self.thread: Thread | None = None
        self.easy_color_mode = easy_color_mode
        if get_option(opt, "legacy_reload", False):
            # Some pytorch versions can not load the optimizer... This is an ugly fix for resuming training.
            # Set: --legacy_reload for reloading without reloading the optimizer.
            opt.new = False
            last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name="logs_diffusion", best=False)
            opt.new = True
            lighting_dic = torch.load(last_checkpoint)
            state_dict_generator: OrderedDict[str, Tensor] = OrderedDict()
            state_dict_ema: OrderedDict[str, Tensor] = OrderedDict()
            for key, value in lighting_dic["state_dict"].items():
                if "generator." in key:
                    state_dict_generator[key[10:]] = value
                elif "ema." in key:
                    state_dict_ema[key[4:]] = value
                # else:
                #    print(key)
            self.generator.load_state_dict(state_dict_generator)
            self.ema.load_state_dict(state_dict_ema)

    def assert_input(self, x_conditional, label, embedding, guidance_w) -> None:
        ########### Check inputs ####################
        assert self.conditional_dimensions == 0 or x_conditional is not None, "The model requires a conditional input."
        assert (
            x_conditional is None or x_conditional.shape[1] == self.conditional_dimensions
        ), f"The conditional input must have {self.conditional_dimensions} color dimension. Got shape {x_conditional.shape}"
        assert (
            not hasattr(self.generator, "label_emb") or label is not None
        ), "The model requires a label input as a one-hot vector."
        assert self.generator.conditional_embedding_size == 0 or (
            embedding is not None and self.generator.conditional_embedding_size == embedding.shape[-1]
        ), f"The model requires a embedding input with size {self.generator.conditional_embedding_size}."
        if guidance_w != 0:
            from utils.utils_diffusion import get_option

            image_dropout: float = self.opt.image_dropout
            inpainting = get_option(self.opt, "inpainting", [])
            assert image_dropout > 0 or (
                inpainting is not None and len(inpainting) != 0
            ), "For classifier free guidance you need to train the model with image_dropout = 0.5 not 0"

    @torch.no_grad()
    def forward(
        self,
        batch_size: int,
        start_timestep: int | None = None,
        encode_target: Tensor | None = None,
        mask_for_inpainting: Tensor | None = None,
        x_conditional: Tensor | None = None,
        label: Tensor | None = None,
        embedding=None,
        intermediate: List[Tensor] | None = None,
        force_fixed_variance: bool = False,
        guidance_w: float = 0,
        guidance_mapping: Callable[[float, float], float] = first,
        cond_fn=None,
        skip_steps=False,
    ) -> Tensor:  # Tuple[Tensor, typing.List[Tensor]]
        """Generate new Samples from the network

        Args:
            batch_size (int):
                Number of samples generated
            start_timestep (int, optional):
                Do only n time steps. Goes n ot 0. Is used by encode_target as the starting noise. Defaults to None.
            encode_target (Tensor, optional):
                Used for encoding as a initial information when start_timestep is set or used with mask_for_inpainting as the given task.. Defaults to None.
            mask_for_inpainting (Tensor, optional):
                Provide a [0,1] mask for image inpainting. Defaults to None.
            x_conditional (Tensor, optional):
                Condition. Defaults to None.
            label (int, optional):
                Condition. Defaults to None.
            embedding (*, optional):
                Condition. Defaults to None.
            intermediate (List, optional):
                List of int[t]. Those timesteps t are stored and returned in a list. Defaults to None.
            force_fixed_variance (bool, optional):
                Don't use the learned variance. Defaults to False.
            guidance_w (int, optional):
                Classifier-Free Diffusion Guidance https://arxiv.org/abs/2207.12598. Defaults to 0.
            guidance_mapping (function, optional):
                TODO. Defaults to first.
            cond_fn (function, optional):
                Network guidance. You provide a function like in utils.diffusion_network_based_guidance. Defaults to None.
            skip_steps (bool, optional):
                Only use the steps given in intermediate, Skip all others. Defaults to False. Does not produce particularly nice images most of the time.

        Returns:
            Tensor | Tuple[Tensor, typing.List[Tensor]]: \n
                The generated sample. It is denormalized and on CPU. \n
                A list of intermediate steps; NOT denormalized and on CPU. Returned when intermediate != None
        """
        self.assert_input(x_conditional, label, embedding, guidance_w)
        if encode_target is not None:
            assert (
                mask_for_inpainting is not None or start_timestep is not None
            ), "You use a encode_target but dont use a mask for inpainting or a initial timestep "

        if not x_conditional is None and self.opt.conditional:
            shape = list(x_conditional.shape)
            shape[1] = self.channels
        else:
            if self.opt.size_w != -1:
                h, w2 = (self.opt.size, self.opt.size_w)
            else:
                h, w2 = (self.opt.size, self.opt.size)
            shape = (batch_size, self.channels, w2, h)
        # channels = self.opt.image_channels_A
        device = self.device
        # 1: x_0 ~ Normal(0,I)
        x = torch.randn(shape, device=device).to(torch.float32)
        # You can encode the starting image with start_timesteps and encode_target.
        if encode_target is not None and start_timestep is not None:
            assert start_timestep != self.num_timesteps, "Don't insert a start time step if you start from scratch"
            # print('##############################\n''##############################\n''##############################\n''##############################\n''##############################\n')
            # Forward | 5_1: x_t ~ sqrt(a_bar_t) x_0 + sqrt(1-alpha_bar_t)*e
            t = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
            print(t.device, t.shape, x.device, x.shape, encode_target.shape, t, start_timestep)

            x = self.constants.mix_img_and_noise(encode_target.to(device), x, t.cuda())

        out_intermediate_list = []
        if start_timestep is None:
            start_timestep = self.num_timesteps

        # Image Inpainting. Set some information on the final input
        if mask_for_inpainting is not None:
            assert encode_target is not None, "For image inpainting a initial image is needed. Use encode_target  "
            mask_for_inpainting = mask_for_inpainting.to(device)
            x = x * (1 - mask_for_inpainting) + encode_target.to(device) * mask_for_inpainting
        # 2: for t=T,...,1 do
        if skip_steps:
            assert intermediate != None, "skip_steps requires intermediate to be not None"
            steps = list(intermediate)
            start_timestep = len(steps)
        else:
            steps = range(0, start_timestep)
        for t in tqdm(reversed(steps), desc="sampling ddpm", total=start_timestep):
            x = self.forward_single_step(
                x,
                batch_size,
                t,
                intermediate,
                out_intermediate_list,
                mask_for_inpainting,
                x_conditional,
                label,
                embedding,
                force_fixed_variance,
                guidance_w,
                guidance_mapping,
                cond_fn,
            )
        x = self.denormalize(x)

        if len(out_intermediate_list):
            return x, out_intermediate_list  # type: ignore
        return x

    @torch.no_grad()
    def forward_single_step(
        self,
        x,
        batch_size,
        i,
        intermediate,
        out_intermediate_list,
        mask_for_inpainting,
        x_conditional,
        label,
        embedding,
        force_fixed_variance=False,
        w: float = 0,
        guidance_mapping: Callable[[float, float], float] = first,
        cond_fn=None,
    ):
        # https://arxiv.org/abs/2006.11239
        # algorithm 2 sampling inner-for-loop
        device = self.device
        shape = x.shape
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        if label is not None:
            label = label.to(torch.long)
        if embedding is not None:
            embedding = embedding.to(torch.float32)
        # Concat in conditional mode to the input
        if self.opt.conditional:
            assert not x_conditional is None
            x_in = torch.cat([x, x_conditional], dim=1)

            if w != 0:
                x_in_unconditional = torch.cat([x, x_conditional * 0], dim=1)
        else:
            x_in = x
        # Generator predict noise (epsilon).
        # 1: e(x_t,t) is the output of the model and the predicted noise of the image
        e = self.generator(
            x_in.to(torch.float32),
            t.to(torch.float32),
            label=label,
            embedding=embedding,
        ).to(self.constants.accuracy)
        # Compute Variance from prediction or use the fixed variance.
        if self.generator.learned_variance:
            var_pred: Tensor
            e, var_pred = torch.split(e, self.channels, dim=1)
        if self.generator.learned_variance and not force_fixed_variance:
            variance_sqrt = self.constants.predicted_variance_sqrt(var_pred, t, shape)  # type: ignore
        else:
            variance_sqrt = self.constants.fixed_variance_sqrt(t, shape)
        # Classifier-Free Diffusion Guidance
        # https://openreview.net/forum?id=qw8AKxfYbI
        # Normalized classifier-free Guidance
        # https://arxiv.org/pdf/2205.12952.pdf
        # I tried to implement Normalized classifier-free Guidance, but it does not work.
        if w != 0:
            if label is not None:
                WarnOnlyOnce.warn(
                    "There is currently no sensible default for 'no conditions', so if you ONLY train on label, w!=0 should no work"
                    + "\nTODO the dataset already produces a dummy label, but currently there is no communication between here and the ds."
                )
            if embedding is not None:
                embedding = embedding * 0
            e_unconditional = self.generator(
                x_in_unconditional.to(torch.float32),  # type: ignore
                t.to(torch.float32),
                label=label,
                embedding=embedding,
            ).to(self.constants.accuracy)
            if self.opt.image_mode:
                WarnOnlyOnce.warn(
                    "Classifier-Free Diffusion Guidance is not implemented for image_mode in ddpm, use ddim instead."
                )
            if self.generator.learned_variance:
                e_unconditional, _ = torch.split(e_unconditional, self.channels, dim=1)
            # My proposal: make w dependent on t, we w can be big when t is big. (Remember t starts at 1000 and goes to 0)
            # Improves short trained models, but with better models it had diminishing returns. :/
            w = guidance_mapping(w, i)
            e = e + w * (e - e_unconditional)
        #################################################################################
        # 4_1 without "+ Var*noise"

        posterior_mean = self.constants.compute_new_mean(e, t, x, not self.opt.image_mode, self)

        # 3 pulling z
        noise = torch.randn_like(x)
        if not intermediate is None and i in intermediate:
            out_intermediate_list.append(x.cpu())
        # no noise when t == 0
        if i == 0:
            return posterior_mean
        # Optional: network based guidance (+scale*var*grad(log(p(y|x)))) where p(y|x) is a regression network
        # where we want a feature y to be maximized

        if cond_fn is not None:
            WarnOnlyOnce.warn("network_based_guidance is untested and in prototype phase, Debugging may be necessary.")
            posterior_mean += network_based_guidance(
                cond_fn, posterior_mean, t, variance_sqrt, x_conditional=x_conditional, label=label, embedding=embedding
            )
        # 4_2: Equation 7 + VAR[t]²*z if t != 0
        x_out = posterior_mean + variance_sqrt * noise
        if mask_for_inpainting is not None:
            x_out = x_out * (1 - mask_for_inpainting) + x * mask_for_inpainting
        x_out = x_out.to(self.constants.accuracy)  # type: ignore

        return x_out

    @torch.no_grad()
    def forward_ddim(
        self,
        batch_size: int,
        intermediate: list[int] | range,  # List of steps visited by DDIM
        eta: float = 0,  # (0 is DDIM, and 1 is one type of DDPM)
        x_conditional: Tensor | None = None,
        label: Tensor | None = None,
        embedding: Tensor | None = None,
        noise: Tensor | None = None,  # Fixed Noise, Chosen random if not provided
        return_noise: bool = False,
        w: float = 0,  # Classifier-Free Diffusion Guidance
        guidance_mapping=first,
        progressbar=True,
    ):
        # https://github.com/ermongroup/ddim/blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/functions/denoising.py#L10
        # ETA controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM).
        self.assert_input(x_conditional, label, embedding, w)
        seq = list(intermediate)  # List with Steps that will be used
        if not x_conditional is None and self.opt.conditional:
            shape = list(x_conditional.shape)
            shape[1] = self.channels
        else:
            if self.opt.size_w != -1:
                h, w2 = (self.opt.size, self.opt.size_w)
            else:
                h, w2 = (self.opt.size, self.opt.size)
            shape = (batch_size, self.channels, w2, h)

        device = self.device  # self.constants.betas.device
        # 1: x_T ~ Normal(0,I)
        if noise is None:
            img = torch.randn(shape, device=device).to(self.constants.accuracy)  # type: ignore
        else:
            assert shape == tuple(
                noise.shape
            ), f"Expected same shape. noise.shape {tuple(noise.shape)}; expected {shape}"
            img = noise.to(self.constants.accuracy)
        out_intermediate_list = [img.cpu()]
        assert len(seq) != 0
        x0_t: Tensor = img  # no unbound
        for i, j in tqdm(
            zip(reversed(seq), reversed([0] + seq[:-1])),
            total=len(seq),
            desc=f"sampling ddim eta={eta:.1f}",
            disable=not progressbar,
        ):
            # TODO Image_mode is untested
            # https://arxiv.org/pdf/2010.02502v3.pdf
            # Formula 12
            # but a_t-1 is replaced with a_t' where t' is any t' > 0 and t' < t.
            # This is done by replacing a_t-1 with a_t' and roh_t(eta) with formula 16
            # This leads to formula D.3 CLOSED FORM EQUATIONS FOR EACH SAMPLING STEP
            # alpha is alpha bar in all examples...
            # sqrt(a_t')*(x_t - epsilon_t*sqrt(1-a_t))/sqrt(a_t)+sqrt(1-a_t'- roh_t(eta)²)*epsilon_t + roh_t(eta)*epsilon
            # sqrt(a_t')*[                 xt0                 ]+[         c2           ] *epsilon_t + [   c1   ]*epsilon
            #                                                                 [   c1   ]               [   c1   ]*epsilon
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_next = torch.full((batch_size,), j, device=device, dtype=torch.long)
            # Note: this is alpha bar not alpha. The paper is lacy...
            # Note: we use not the precomputed values for higher accuracy
            a_bar_t, a_bar_t_next = self.constants.ddim_alphas(t, t_next)

            x = img
            if label is not None:
                label = label.to(torch.long)
            if embedding is not None:
                embedding = embedding.to(torch.float32)
            if not x_conditional is None and self.opt.conditional:
                x = torch.cat([x, x_conditional], dim=1).to(torch.float32)

                if w != 0:
                    x_in_unconditional = torch.cat([img, x_conditional * 0], dim=1).to(torch.float32)
            e_t = self.generator(
                x,
                t.to(torch.float32),
                label=label,
                embedding=embedding,
            ).to(self.constants.accuracy)

            if self.generator.learned_variance:
                # WarnOnlyOnce.warn("Variance is not used in the ddim sampling")
                e_t, var_pred = torch.split(e_t, self.channels, dim=1)
            #################################################################################
            # Classifier-Free Diffusion Guidance
            # https://openreview.net/forum?id=qw8AKxfYbI
            # x0_t is the final prediction if t = 0
            if self.opt.image_mode:
                x0_t = e_t
                # Instead of computing the image, we compute the noise.
                # Just inverted the formula in the "if not self.opt.image_mode"
                e_t = ((x0_t * a_bar_t.sqrt()) - img) / (1 - a_bar_t).sqrt()
            if w != 0:
                if label is not None:
                    WarnOnlyOnce.warn(
                        "There is currently no sensible default for 'no conditions', so if you ONLY train on label, w!=0 should no work"
                        + "\nTODO the dataset already produces a dummy label, but currently there is no communication between here and the ds."
                    )
                if embedding is not None:
                    embedding = embedding * 0

                e_unconditional = self.generator(
                    x_in_unconditional, t.to(torch.float32), label=label, embedding=embedding  # type: ignore
                ).to(self.constants.accuracy)
                if self.generator.learned_variance:
                    e_unconditional, _ = torch.split(e_unconditional, self.channels, dim=1)
                w = guidance_mapping(w, i)
                if self.opt.image_mode:
                    # Instead of noise we get the image, we compute the noise.
                    e_unconditional: Tensor = ((e_unconditional * a_bar_t.sqrt()) - img) / (1 - a_bar_t).sqrt()
                e_t = e_t + w * (e_t - e_unconditional)

            #################################################################################
            if not self.opt.image_mode:
                x0_t: Tensor = (img - e_t * (1 - a_bar_t).sqrt()) / a_bar_t.sqrt()
            out_intermediate_list.append(x0_t.cpu())

            # sqrt propagated
            c1 = eta * ((1 - a_bar_t_next) / (1 - a_bar_t) * (1 - (a_bar_t / a_bar_t_next))).sqrt()
            c2 = ((1 - a_bar_t_next) - c1**2).sqrt()
            xt_next = a_bar_t_next.sqrt() * x0_t + c2 * e_t + c1 * torch.randn_like(img)

            img = xt_next

        if return_noise:
            return self.denormalize(x0_t), out_intermediate_list, noise
        return self.denormalize(x0_t), out_intermediate_list

    def configure_optimizers(self):
        opt = self.opt
        assert opt != None
        optimizer_G = torch.optim.Adam(itertools.chain(self.generator.parameters()), lr=self.opt.lr)  # default betas
        # optimizer_G = torch.optim.AdamW(itertools.chain(self.generator.parameters()),
        #                               lr=self.opt.lr) #default betas

        return optimizer_G

    def training_step(
        self, train_batch: Batch_Dict, batch_idx, img_decoder: AutoencoderKL | None = None, return_loss=True
    ):
        # 2: x_0 ~ q(x_0)
        #'target' and optionally 'condition'(image), 'label', 'embedding'

        x_0: Tensor = train_batch.get("target")
        x_conditional: Tensor | None = train_batch.get("condition", None)  # condition if it is an Image
        label: Tensor | None = train_batch.get("label", None)  # condition if it is an Label
        embedding: Tensor | None = train_batch.get("embedding", None)
        # same shape as target with 1 and 0 as input, target input will be placed without the noise when input shape == 0 at that pixel
        mask_for_inpainting: Tensor | None = train_batch.get("mask", None)
        mask_for_inpainting_cond: Tensor | None = train_batch.get("mask_cond", None)
        # with autocast(enabled=False):  # self.amp
        if self.opt.volumes:  # 3d
            b = x_0.shape[0]
            c = x_0.shape[1]
        else:
            if self.opt.size_w != -1:
                h1, w1 = (self.opt.size, self.opt.size_w)
            else:
                h1, w1 = (self.opt.size, self.opt.size)
            b, c, h, w = x_0.shape
            assert h == h1 and w == w1, f"height and width of image must be {self.opt.size} and not {w}, {h}"
        # random timestep for each sample in bach | 3: t~Uniform(1...T)
        t = torch.randint(0, self.num_timesteps, (b,), device=x_0.device).long()
        # Gaussian | 4: epsilon ~ Normal(0,I)
        noise = torch.randn_like(x_0)
        if self.channels != 1 and self.easy_color_mode:
            for j in range(noise.shape[1]):
                noise[:, j] -= noise[:, j].mean()

        # Forward | 5_1: x_t ~ sqrt(a_bar_t) x_0 + sqrt(1-alpha_bar_t)*e
        x_t = self.constants.mix_img_and_noise(x_0, noise, t)
        if mask_for_inpainting is None:
            x = x_t
        else:
            noise = noise * (1 - mask_for_inpainting)
            x_t = x_t * (1 - mask_for_inpainting) + x_0 * (mask_for_inpainting)
            x = x_t
        if mask_for_inpainting_cond is not None:
            x_conditional = x_conditional * mask_for_inpainting_cond
        # Conditional p(x_0| y) -> p(x_0)*p(y|x_0) --> just added it to the input
        if not x_conditional is None and self.opt.conditional:
            x = torch.cat([x, x_conditional], dim=1)
        # --------------
        if hasattr(self.generator, "special_loss"):
            loss = self.generator.special_loss(self, x_0, noise, x, t, label, embedding, x_t=x_t)  # type: ignore
            return loss
        # --------------
        # Model predicts the noise; Also predicts the variance is close to beta_t or beta_tilde_t if var is not fixed.
        # Backward | 5_2: z_theta = Theta(x,t)
        model_out = self.generator(x, time=t, label=label, embedding=embedding)
        additional_loss = 0
        if isinstance(model_out, tuple):
            additional_loss = model_out[1]
            model_out = model_out[0]

        ########################################## Variance ############################################################################
        lamda_var_loss = 1 / 1000
        if self.generator.learned_variance:
            assert model_out.shape[1] == 2 * c
            # Split prediction in to two equal halves, with same dimensions as input image (excluding conditional).
            model_out, frac = torch.split(model_out, c, dim=1)
            # Compute the real mean and var
            target_mean = self.constants.compute_posterior_mean(x_0, x_t, t)
            target_log_var = self.constants.fixed_variance_log(t, x_t.shape)
            # Compute the predicted mean and var
            predicted_mean = self.constants.compute_new_mean(model_out, t, x_t, not self.opt.image_mode, self)
            predicted_log_variance = self.constants.predicted_variance_log(frac, t, x_t.shape)
            # Compute the Loss
            kl = losses.normal_kl(target_mean, target_log_var, predicted_mean, predicted_log_variance)
            kl = kl.view(b, -1).mean(1) / math.log(2.0)
            decoder_nll = -losses.discretized_gaussian_log_likelihood(
                x_0, means=predicted_mean, log_scales=0.5 * predicted_log_variance
            )
            decoder_nll = decoder_nll.view(b, -1).mean(1) / math.log(2.0)
            loss_var = torch.where((t == 0), decoder_nll, kl) * lamda_var_loss
            loss_var = loss_var.mean()
        else:
            loss_var = 0
        ###################################################################################################################################
        if return_loss:
            if self.opt.image_mode:
                # This is needed for stable diffusion, ignored for normal diffusion
                if img_decoder is not None:
                    model_out = img_decoder.decode(model_out)
                    x_0 = img_decoder.decode(x_0)
                loss: Tensor = self.criterion_generator(model_out, x_0)

                if opt.lambda_ssim > 0.0:
                    loss_ssim = opt.lambda_ssim * (
                        1
                        - structural_similarity_index_measure(
                            self.denormalize(model_out), self.denormalize(x_0), data_range=1.0
                        )
                    )  # type: ignore
                    self.log(f"train_loss_ssim", loss_ssim.detach())
                    loss += loss_ssim
            else:
                # We could predict the image instead of noise, but noise works better (claimed)
                # 5_3: ||epsilon - z_theta||
                loss = self.criterion_generator(model_out, noise)

            # The loss has a costume aggregation with weights dependent on t
            loss = self.constants.weight_and_mean_loss(loss, t) + loss_var + additional_loss

            # loss = self.scaler.scale(loss / self.gradient_accumulate_every)  # .backward()

            try:
                self.log(f"train_All", loss.detach())
            except Exception:
                pass
            return loss
        else:
            return (
                model_out,
                loss_var,
                x_0,
                x_conditional,
                x_t,
                label,
                embedding,
                mask_for_inpainting,
                mask_for_inpainting_cond,
                t,
                noise,
                additional_loss,
            )

    def validation_step_by_callback(
        self, batch: Batch_Dict, batch_idx, logger: TensorBoardLogger | None = None, ddpm=False
    ):
        opt = self.opt
        self.counter += 1

        if logger is None:
            tb_logger: TensorBoardLogger = self.logger  # type: ignore
        else:
            tb_logger = logger
        # if self.counter % 3 != 1:
        #    return

        # Possible returns of the dataset
        target: Tensor = batch.get("target")
        x_conditional: Tensor | None = batch.get("condition", None) if opt.conditional else None
        label: Tensor | None = batch.get("label", None)
        embedding: Tensor | None = batch.get("embedding", None)
        mask_for_inpainting: Tensor | None = batch.get("mask", None)
        mask_for_inpainting_cond: Tensor | None = batch.get("mask_cond", None)
        encode_target: Tensor | None = target if mask_for_inpainting is not None else None
        if mask_for_inpainting_cond is not None:
            x_conditional2 = x_conditional * mask_for_inpainting_cond
        else:
            x_conditional2 = x_conditional
        # Generate Image
        generated_other: List[Tensor] = []
        if ddpm:
            generated: Tensor = self.forward(
                opt.num_validation_images,
                x_conditional=x_conditional2,
                label=label,
                embedding=embedding,
                mask_for_inpainting=mask_for_inpainting,
                encode_target=encode_target,
            ).cpu()  # type: ignore
        else:
            generated: Tensor = self.forward_ddim(
                opt.num_validation_images,
                intermediate=list(range(0, self.num_timesteps, 10)),
                x_conditional=x_conditional2,
                label=label,
                embedding=embedding,
                w=0
                # mask_for_inpainting=mask_for_inpainting,
                # encode_target=encode_target,
            )[
                0
            ].cpu()  # type: ignore
        print(end="\r")
        # Log labels/embeddings
        try:
            # Print Label
            if label is not None:
                tb_logger.experiment.add_text(
                    "generated_images_labels",
                    str(label.tolist()),
                    self.counter,
                )
            if embedding is not None:
                tb_logger.experiment.add_text(
                    "generated_images_embedding",
                    str(embedding.tolist()),
                    self.counter,
                )
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
        if x_conditional is None and mask_for_inpainting is None:
            grid = torchvision.utils.make_grid(generated, nrow=4)
            tb_logger.experiment.add_image(
                "generated images",
                grid,
                self.counter,
            )
        # Print conditional images and generated images side by side
        else:
            # Map to [0,1]
            target = self.denormalize(target.cpu())
            # List of images that will be displayed side by side
            stacked_images = [target, generated]
            stacked_images += generated_other
            if x_conditional is not None:
                x_conditional = self.denormalize(x_conditional.cpu(), conditional=True)
                c = self.channels
                stacked_images += [x_conditional[:, i : i + c].cpu() for i in range(0, x_conditional.shape[1], c)]
            if mask_for_inpainting_cond is not None:
                assert x_conditional2 is not None
                c = self.channels
                x_conditional2 = self.denormalize(x_conditional2.cpu(), conditional=True)
                stacked_images += [x_conditional2[:, i : i + c].cpu() for i in range(0, x_conditional2.shape[1], c)]

            if mask_for_inpainting is not None:
                stacked_images.append((mask_for_inpainting.cpu() * target))
            if opt.transpose_preview:
                stacked_images = [i.transpose(-1, -2) for i in stacked_images]
            grid = torchvision.utils.make_grid(
                torch.cat(stacked_images, dim=-1).cpu(),
                nrow=2 if not self.opt.volumes else 5,
            )

            if self.opt.volumes:
                grid = torch.nn.functional.interpolate(grid, scale_factor=2, mode="nearest")  # type: ignore

            tb_logger.experiment.add_image("conditional image", grid, self.counter)

    def get_mean_std(self, conditional: bool):
        assert self.opt.normalize is not None
        if conditional and "conditional" in self.opt.normalize:
            info: dict[str, list[float]] = self.opt.normalize["conditional"]  # type: ignore
        elif not conditional and "target" in self.opt.normalize:
            info: dict[str, list[float]] = self.opt.normalize["target"]  # type: ignore
        else:
            info: dict[str, list[float]] = self.opt.normalize  # type: ignore
        mean = info["mean"]
        std = info["std"]
        if len(mean) == 1 and self.channels != 1:
            mean = [mean[0] for _ in range(self.channels)]
            std = [std[0] for _ in range(self.channels)]
            info["mean"] = mean
            info["std"] = std
        return mean, std

    def normalize(self, tensor, conditional=False) -> Tensor:
        if hasattr(self.opt, "normalize"):
            mean, std = self.get_mean_std(conditional)
            tensor = tensor.clone()
            for i in range(self.channels):
                tensor[:, i].sub_(mean[i]).div_(std[i])
            return tensor
        else:
            return (tensor * 2) - 1  # map [-1,1]->[0,1]

    def denormalize(self, tensor: Tensor, conditional=False) -> Tensor:
        if hasattr(self.opt, "normalize"):
            mean, std = self.get_mean_std(conditional)
            tensor = tensor.clone()
            for i in range(tensor.shape[1]):
                try:
                    tensor[:, i].mul_(std[i]).add_(mean[i])
                except:
                    tensor[:, i].mul_(std[0]).add_(mean[0])
            return tensor
        else:
            return (tensor + 1) / 2  # map [0,1]->[-1,1]

    def clamp(self, tensor):
        tensor = self.denormalize(tensor)
        tensor = torch.clamp(tensor, 0, 1)
        tensor = self.normalize(tensor)
        return tensor


from loader import arguments
from loader import load_dataset


def main(opt: arguments.Diffusion_Option, limit_train_batches=1.0):
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
    model = Diffusion(opt=opt, in_channel=in_channel)
    # Get last checkpoint. If there is non or --new was called this returns None and starts a new model.
    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name="logs_diffusion")

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
        def on_train_epoch_start(self, trainer, pl_module: Diffusion):
            pl_module.cuda()
            time.sleep(0.5)
            batch = {}
            for k, v in next(iter(dummy_val_loader)).items():
                batch[k] = v.to(pl_module.device)
            # next(iter(dummy_val_loader))
            pl_module.thread = Thread(target=pl_module.validation_step_by_callback, args=(batch, 0))
            pl_module.thread.start()
            print(end="\r")

        def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: Diffusion) -> None:
            if not pl_module.thread is None:
                pl_module.thread.join(timeout=600)
                pl_module.thread = None
                print(end="\r")

        def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: Diffusion, outputs, batch, batch_idx: int
        ) -> None:
            pl_module.ema.update()

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
        num_nodes=1,  # T
        limit_train_batches=limit_train_batches,  # Train only x % (if float) or train only on x batches (if int)
        max_epochs=opt.max_epochs,  # Stopping epoch
        logger=logger,
        callbacks=[mc_last, ValidationCallback()],
        resume_from_checkpoint=last_checkpoint,
        detect_anomaly=opt.prevent_nan,
        strategy=None if isinstance(gpus, int) or len(gpus) == 1 else "ddp",
        # amp_backend="apex",
        # amp_level="O3",
        # find_unused_parameters
    )
    trainer.tune(model)
    trainer.fit(
        model,
        train_loader,
        # dummy_val_loader,
    )  # ckpt_path=last_checkpoint
    # recommended lr = 2e-5

    # CUDA_VISIBLE_DEVICES=0 python3 diffusion.py -lr 2e-5 -bs 8 -epoch 1000 -en diffusion_ct -s 192 -ds Img --root /media/data/robert/datasets/TGD_2D/ct/train
    # CUDA_VISIBLE_DEVICES=1 python3 diffusion.py -lr 2e-5 -bs 8 -epoch 1000 -en diffusion_ct -s 104 -ds cat


def get_opt(parser=None, config=None) -> arguments.Diffusion_Option:
    torch.cuda.empty_cache()

    opt = arguments.parseTrainParam(parser=parser, config=config)
    opt = arguments.parseDiffusion(opt)

    load_dataset.parseParam_datasets(opt)
    # TODO add here model specific parameters
    # opt = arguments.parseParam_cyk(opt)
    opt = arguments.Diffusion_Option.from_kwargs(**opt.parse_args().__dict__)
    return opt


if __name__ == "__main__":
    opt = get_opt()

    main(opt)
