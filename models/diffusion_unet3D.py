import torch
from torch._C import device
import torch.nn as nn
from einops import rearrange
from functools import partial
import math


def default(val, d):
    from inspect import isfunction

    if val is not None:
        return val
    return d() if isfunction(d) else d


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):

        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, input):
        b, c, x, y, z = input.shape
        qkv = self.to_qkv(input).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=x, y=y, z=z)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, input):
        b, c, x, y, z = input.shape
        qkv = self.to_qkv(input).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv)
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=x, y=y, z=z)
        return self.to_out(out)


# model


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        conditional_dimensions=0,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
        conditional_label_size=0,
        conditional_embedding_size=0,
        patch_size=1,  # Improving Diffusion Model Efficiency Through Patching https://arxiv.org/abs/2207.04316; 1 means deactivated
    ):
        super().__init__()
        self.patch_size = patch_size

        self.learned_variance = learned_variance

        self.conditional_label_size = conditional_label_size
        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d((channels + conditional_dimensions) * patch_size * patch_size * 1, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: int(dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        if conditional_label_size != 0:
            self.label_emb = nn.Embedding(conditional_label_size, time_dim)

        self.conditional_embedding_size = conditional_embedding_size
        if conditional_embedding_size:
            time_dim += conditional_embedding_size
        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        nn.Conv3d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        nn.ConvTranspose3d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim) * patch_size * patch_size * 1

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)
        self.first_forward = True

        # self.final_conv = nn.Sequential(
        #            # nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1)),
        #            # nn.SiLU(),
        #            nn.Conv3d(dim, self.out_dim, 1),
        #        )

    # Improving Diffusion Model Efficiency Through Patching https://arxiv.org/abs/2207.04316
    def to_patches(self, x):
        p = self.patch_size
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, D, H, W // p, C * p)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, D, W // p, H // p, C * p * p)
        return x.permute(0, 4, 1, 3, 2)

    def from_patches(self, x):
        p = self.patch_size
        B, C, D, H, W = x.shape

        x = x.permute(0, 2, 4, 3, 1).reshape(B, D, W, H * p, C // p)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, D, H * p, W * p, C // (p * p))
        return x.permute(0, 4, 1, 2, 3)

    def forward(self, x, time=None, label=None, embedding=None) -> torch.Tensor:  # time
        down_factor = 2 ** (len(self.downs) - 1)
        shape = x.shape
        assert shape[-1] % down_factor == 0, f"dimensions are not dividable by {down_factor}, {shape}, {shape[-1]}"
        assert shape[-2] % down_factor == 0, f"dimensions are not dividable by {down_factor}, {shape}, {shape[-2]}"
        assert shape[-3] % down_factor == 0, f"dimensions are not dividable by {down_factor}, {shape}, {shape[-3]}"
        if self.first_forward:
            print("|", x.shape)
        if self.patch_size != 1:
            x = self.to_patches(x)
        if self.first_forward:
            print("|", x.shape)

        # time = None
        if time is None:
            time = torch.ones((1,), device=x.device)
        x = self.init_conv(x)
        r = x.clone()
        if self.first_forward:
            print("-", x.shape, "\tAttention")

        t = self.time_mlp(time)

        if hasattr(self, "label_emb"):
            assert label is not None, "This UNet requires a class label"
            t = t + self.label_emb(label)

        if self.conditional_embedding_size != 0:
            assert embedding is not None
            t = torch.cat([embedding, t], dim=-1)
        h = []
        ö = "-"
        for block1, block2, attn, downsample in self.downs:  # type: ignore
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            if self.first_forward:
                ö += "-"
                print(ö, x.shape, "\t", isinstance(attn, Residual))

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        if self.first_forward:
            print(ö, x.shape)
            ö = ö[:-1]

        for block1, block2, attn, upsample in self.ups:  # type: ignore
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            if self.first_forward:
                print(ö, x.shape, "\t", isinstance(attn, Residual))
                ö = ö[:-1]

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        if self.first_forward:
            print("|", x.shape)

        x = self.final_conv(x)

        if self.first_forward:
            print("|", x.shape)

        if self.patch_size != 1:
            x = self.from_patches(x)
        if self.first_forward:
            print("|", x.shape)

        self.first_forward = False

        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
    )
    print(model)
    summary(model, (1, 32, 32, 32))
