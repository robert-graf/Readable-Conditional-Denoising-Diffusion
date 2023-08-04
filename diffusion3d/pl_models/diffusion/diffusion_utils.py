from typing import Any, Tuple
from numpy import dtype
import torch
from torch import Size, Tensor, nn
from math import pi
from typing import List
import torch.nn.functional as nn_fun
from warnings import warn


def generate_gif(
    gen_samples: List[Tensor],  # List of tensors that should be used for the gif. (output when you use intermediate)
    time_in_s=4,  # Animation duration
    n_hold_final=10,  # The last image is sampled multiple times.
    gif_shape=(3, 3),  # Shape of the gif, additional image are cut away!
    path="pred.gif",
):
    gen_samples = [(i.clamp(-1, 1) + 1) / 2 for i in gen_samples]  # + [a]
    print(torch.min(gen_samples[-1]))
    sample_batch_size = gif_shape[0] * gif_shape[1]
    # Process samples and save as gif
    gen_samples = [(i[:sample_batch_size] * 255.0).type(torch.uint8) for i in gen_samples]
    img = gen_samples[-1]
    for _ in range(n_hold_final):
        gen_samples.append(img)

    gen_samples = [i.reshape(gif_shape + img.shape[-3:]) for i in gen_samples]
    gen_samples = [torch.cat([i[j] for j in range(i.shape[0])], dim=-1) for i in gen_samples]
    gen_samples = [torch.cat([i[j] for j in range(i.shape[0])], dim=-2) for i in gen_samples]
    gen_samples = [i.permute(1, 2, 0) for i in gen_samples]

    import imageio

    imageio.mimsave(
        path,
        list(gen_samples),
        fps=len(gen_samples) // time_in_s,
    )


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


def get_option(opt, attr, default, separated_list=False) -> Any:
    if opt != None and hasattr(opt, attr):
        if separated_list:
            a = getattr(opt, attr)
            if a is None:
                return default
            try:
                a = tuple([float(i) for i in a.split(",")])
            except:
                assert False, f"{getattr(opt, attr)} is not a comma separated list of numbers"
            return a
        return getattr(opt, attr)
    return default


def extract(a: Tensor, t: Tensor, x_shape: Size):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    a_bar = torch.cos(((x / timesteps) + s) / (1 + s) * torch.tensor(pi) * 0.5) ** 2
    a_bar = a_bar / a_bar[0]
    betas = 1 - (a_bar[1:] / a_bar[:-1])
    return torch.clip(betas, 0, 0.999)


schedule_dict = {
    "linear": linear_beta_schedule,
    "cosine": cosine_beta_schedule,
}


class Diffusions_Constance(nn.Module):
    def __init__(self, scheduler_name: str, timesteps: int = 1000, accuracy=torch.float32):
        super().__init__()
        self.accuracy = accuracy
        betas: Tensor = schedule_dict[scheduler_name](timesteps).to(torch.float64)
        self.schedule_name = scheduler_name
        self.p2_loss_weight_gamma = (
            0.0  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        )
        self.p2_loss_weight_k = 1
        self.compute_constants(betas)

    def s(self, name, value: Tensor):
        # save parameter and cast it to float32
        self.register_buffer(name, value.to(self.accuracy))
        # getattr(self, name).requires_grad_(False)

    def compute_constants(self, betas: Tensor):
        self.betas = betas.to(self.accuracy)
        alphas: Tensor = 1.0 - betas
        a_bar: Tensor = torch.cumprod(alphas, axis=0)  # type: ignore
        a_bar_prev = nn_fun.pad(a_bar[:-1], (1, 0), value=1.0)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_a_bar = torch.sqrt(a_bar).to(self.accuracy)
        self.sqrt_one_minus_a_bar = torch.sqrt(1.0 - a_bar).to(self.accuracy)
        self.sqrt_reciprocal_a_bar = torch.sqrt(1.0 / a_bar).to(self.accuracy)
        self.sqrt_reciprocal_m1_a_bar = torch.sqrt(1.0 / a_bar - 1).to(self.accuracy)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - a_bar_prev) / (1.0 - a_bar)).to(self.accuracy)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # this is min variance
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20)).to(self.accuracy)

        self.log_variance = torch.log(betas).to(self.accuracy)
        self.posterior_mean_coefficient1 = (betas * torch.sqrt(a_bar_prev) / (1.0 - a_bar)).to(self.accuracy)
        self.posterior_mean_coefficient2 = ((1.0 - a_bar_prev) * torch.sqrt(alphas) / (1.0 - a_bar)).to(self.accuracy)
        # scales the loss in the training step. Paper said it can be ignored.
        self.p2_loss_weight = ((self.p2_loss_weight_k + a_bar / (1 - a_bar)) ** -self.p2_loss_weight_gamma).to(self.accuracy)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.sqrt_a_bar = self.sqrt_a_bar.to(device)
        self.sqrt_one_minus_a_bar = self.sqrt_one_minus_a_bar.to(device)
        self.sqrt_reciprocal_a_bar = self.sqrt_reciprocal_a_bar.to(device)
        self.sqrt_reciprocal_m1_a_bar = self.sqrt_reciprocal_m1_a_bar.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.log_variance = self.log_variance.to(device)
        self.posterior_mean_coefficient1 = self.posterior_mean_coefficient1.to(device)
        self.posterior_mean_coefficient2 = self.posterior_mean_coefficient2.to(device)
        self.p2_loss_weight = self.p2_loss_weight.to(device)

    def backward_step_to_x0(self, t, shape):
        c1 = extract(self.sqrt_reciprocal_a_bar, t, shape)  # type: ignore
        c2 = extract(self.sqrt_reciprocal_m1_a_bar, t, shape)  # type: ignore
        return c1, c2

    def mean_step_to_x_t_new(self, t, shape) -> Tuple[Tensor, Tensor]:
        # coef1 = betas * torch.sqrt(a_bar_prev) / (1. - a_bar)
        c1 = extract(self.posterior_mean_coefficient1, t, shape)  # type: ignore
        # coef2 = (1. - a_bar_prev) * torch.sqrt(a) / (1. - a_bar)
        c2 = extract(self.posterior_mean_coefficient2, t, shape)  # type: ignore
        return c1, c2

    def fixed_variance_sqrt(self, t, shape):
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, shape)  # type: ignore
        return (0.5 * posterior_log_variance).exp()

    # The theoretical minimum Variance
    def fixed_variance_log(self, t, shape):
        return extract(self.posterior_log_variance_clipped, t, shape)  # type: ignore

    # The theoretical maximum Variance
    def fixed_variance_log_max(self, t, shape):
        return extract(self.log_variance, t, shape)  # type: ignore

    def predicted_variance_log(self, frac, t, shape):
        # Compute the predicted var. The model predicts how the variance was mixed
        # https://arxiv.org/pdf/2102.09672.pdf
        min_log = self.fixed_variance_log(t, shape)
        max_log = self.fixed_variance_log_max(t, shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (frac + 1) / 2
        return frac * max_log + (1 - frac) * min_log

    def predicted_variance_sqrt(self, frac, t, shape) -> Tensor:
        return (0.5 * self.predicted_variance_log(frac, t, shape)).exp()

    def compute_posterior_mean(self, x_0, x_t, t):
        # c1 = betas * torch.sqrt(a_bar_prev) / (1. - a_bar)
        # c2 = (1. - a_bar_prev) * torch.sqrt(a) / (1. - a_bar)
        c1, c2 = self.mean_step_to_x_t_new(t, x_t.shape)
        # Equation 7 from the paper
        return c1 * x_0 + c2 * x_t

    def compute_new_mean(self, e: Tensor, t, x_t: Tensor, noise_mode=True, clamping=None) -> Tensor:
        # Step 4 from Algorithm, but using equivalence from formula 9,7 https://arxiv.org/abs/2006.11239
        # c1 = 1/sqrt(a_bar_t)
        # c2 = sqrt(1/a_bar_t - 1)
        if noise_mode:
            c1, c2 = self.backward_step_to_x0(t, x_t.shape)
            # Equation 9 right side of mean_tilde_t
            x_0 = c1 * x_t - c2 * e
        else:
            x_0 = e
        if clamping is None:
            pass
            # x_0.clamp_(-1.0, 1.0)
        else:
            x_0 = clamping.clamp(x_0)

        # Equation 7 from the paper
        return self.compute_posterior_mean(x_0, x_t, t)

        # Simpler form: (This form has no clamp(-1,1))
        # https://github.com/azad-academy/denoising-diffusion-model/blob/main/utils.py
        # Factor to the model output
        # eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
        # Model output
        # eps_theta = model(x, t)
        # Final values
        # posterior_variance = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))

    def ddim_alphas(self, t: Tensor, t_next: Tensor) -> Tuple[Tensor, Tensor]:
        beta: Tensor = torch.cat([torch.zeros(1).to(t.device), self.betas], dim=0).to(self.accuracy)  # type: ignore
        a_bar: Tensor = (1 - beta).cumprod(dim=0).index_select(0, t.long() + 1).view(-1, 1, 1, 1)
        a_bar_next: Tensor = (1 - beta).cumprod(dim=0).index_select(0, t_next.long() + 1).view(-1, 1, 1, 1)
        return a_bar, a_bar_next

    def mix_img_and_noise(self, img, noise, t) -> Tensor:
        # x_t = sqrt(a_bar_t) x_0 + sqrt(1-alpha_bar_t)*e
        return extract(self.sqrt_a_bar, t, img.shape) * img + extract(self.sqrt_one_minus_a_bar, t, img.shape) * noise  # type: ignore

    def weight_and_mean_loss(self, loss: Tensor, t):
        loss = loss.view(loss.shape[0], -1)
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)  # type: ignore
        return loss.mean()
