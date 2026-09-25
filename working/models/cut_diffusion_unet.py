from models.diffusion_unet import *


class Unet_CUT(Unet):
    def __init__(self, dim, **kargs):
        super().__init__(dim, **kargs)
        self.final_activation = nn.Tanh()

    def forward(self, x: torch.Tensor, return_intermediate=False, layers=[], time=None, label=None, embedding=None) -> torch.Tensor:
        intermediate = []
        if self.patch_size != 1:
            x = self.to_patches(x)

        x = self.init_conv(x)
        intermediate.append(x.clone())  # 0
        r = x.clone()
        if time is None:
            time = torch.zeros((x.shape[0])).to(x.device)
        t = self.time_mlp(time)
        if hasattr(self, "label_emb"):
            assert label is not None, "This UNet requires a class label"
            t = t + self.label_emb(label)

        if self.conditional_embedding_size != 0:
            assert embedding is not None, "This UNet requires a embedding"
            # This is a general implementation, you my specialize this to your needs. The + operator instead of cat is possible.
            t = torch.cat([embedding, t], dim=-1)

        h = []

        for block1, block2, attn, downsample in self.downs:  # type: ignore
            x = block1(x, t)
            intermediate.append(x.clone()) if return_intermediate else None  # 1, 5, 9, 13
            x = block2(x, t)
            intermediate.append(x.clone()) if return_intermediate else None  # 2
            x = attn(x)
            intermediate.append(x.clone()) if return_intermediate else None  # 3
            h.append(x)
            x = downsample(x)
            intermediate.append(x.clone()) if return_intermediate else None  # 4

        x = self.mid_block1(x, t)
        intermediate.append(x.clone()) if return_intermediate else None  # 14
        x = self.mid_attn(x)
        intermediate.append(x.clone()) if return_intermediate else None  # 15
        x = self.mid_block2(x, t)
        intermediate.append(x.clone()) if return_intermediate else None  # 16

        for block1, block2, attn, upsample in self.ups:  # type: ignore
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            intermediate.append(x.clone()) if return_intermediate else None  # 17,21,25,29
            x = block2(x, t)
            intermediate.append(x.clone()) if return_intermediate else None
            x = attn(x)
            intermediate.append(x.clone()) if return_intermediate else None
            x = upsample(x)
            intermediate.append(x.clone()) if return_intermediate else None

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        intermediate.append(x.clone()) if return_intermediate else None  # 30
        x = self.final_conv(x)
        intermediate.append(x.clone()) if return_intermediate else None  # 31
        if self.patch_size != 1:
            x = self.from_patches(x)

        x = self.final_activation(x)
        if return_intermediate:
            try:
                return x, [intermediate[i] for i in layers]  # type: ignore
            except IndexError:
                assert False, f"There are only {len(intermediate)} layers to choose from. Your chosen layers are {layers}"
        del intermediate
        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=1, patch_size=4)
    print(model)
    from functools import partial

    model.cuda()
    model.forward = partial(model.forward, time=torch.zeros(2).cuda())  # type: ignore
    summary(model, (1, 224, 224))
