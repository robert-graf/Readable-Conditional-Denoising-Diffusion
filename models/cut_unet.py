import torch
import torch.nn as nn


class DownBLock(nn.Module):
    def __init__(self, in_features, out_features, index, reduce_size: bool):
        super(DownBLock, self).__init__()
        conv_block = [
            nn.Conv2d(in_features, out_features, kernel_size=3),  # Same padding
            nn.InstanceNorm2d(out_features),
            nn.ReflectionPad2d(1),
            nn.PReLU(out_features),
            nn.Conv2d(out_features, out_features, kernel_size=3),  # Same padding
            nn.InstanceNorm2d(out_features),
            nn.ReflectionPad2d(1),
            nn.PReLU(out_features),
        ]
        if reduce_size:
            self.pool = nn.Conv2d(out_features, out_features, kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        skip = self.conv_block(x)
        out = self.pool(skip)
        return out, skip


class BridgeBlock(nn.Module):
    def __init__(self, in_features):
        super(BridgeBlock, self).__init__()
        conv_block = [
            nn.Conv2d(in_features, in_features * 2, kernel_size=3),
            nn.InstanceNorm2d(in_features * 2),
            nn.ReflectionPad2d(1),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features * 2, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReflectionPad2d(1),
            nn.Dropout(),
            nn.ReLU(inplace=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class UpBLock(nn.Module):
    def __init__(self, in_features, out_features, index, increase_size=True, drop_out=0.5):
        super(UpBLock, self).__init__()
        conv_block = [
            nn.Conv2d(2 * in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReflectionPad2d(1),
            nn.PReLU(in_features),
            nn.Conv2d(in_features, out_features, kernel_size=3),
            nn.InstanceNorm2d(out_features),
            nn.ReflectionPad2d(1),
            nn.PReLU(out_features),
            nn.Dropout(drop_out if increase_size else 0),
        ]

        self.conv_block = nn.Sequential(*conv_block)
        if increase_size:
            self.up_conv = nn.ConvTranspose2d(in_channels=in_features, out_channels=in_features, kernel_size=2, stride=2)
        else:
            self.up_conv = nn.Identity()
        # h_in	32
        # stride	2
        # padding	0
        # dilation	1
        # kernel_size	2
        # output_padding	0
        # =(B1-1)*B2-2*B3+B4*(B5-1)+B6+1

    def forward(self, x, skip):
        up = self.up_conv(x)
        # print('x',x.shape,'skip', skip.shape,'up', up.shape)
        x_ = torch.cat((up, skip), 1)
        return self.conv_block(x_)


class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, net_G_channel=64, net_G_depth=9, reduce_till=100, net_G_drop_out=0.5, **kargs):
        super(UNet, self).__init__()
        self.num_downs = net_G_depth // 2
        self.num_blocks = self.num_downs * 2 + 1
        downs = []
        ups = []

        in_features = input_nc
        out_features = net_G_channel
        for i in range(self.num_downs):
            downs.append(DownBLock(in_features, out_features, i, i < reduce_till))
            in_features = out_features
            out_features *= 2

        self.bridgeBlock = BridgeBlock(in_features)

        in_features = in_features
        out_features = in_features // 2
        for i in range(self.num_downs):
            j = self.num_downs - i - 1
            ups.append(UpBLock(in_features, out_features, j, j < reduce_till, drop_out=net_G_drop_out if i != self.num_downs - 1 else 0))
            in_features = out_features
            out_features //= 2

        self.lastLayer = nn.Sequential(nn.Conv2d(in_features, output_nc, 1, padding=0), nn.Tanh())

        # required so pytorch can see lists of models
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)

    def forward(self, x, return_intermediate=False, layers=[]):
        if isinstance(return_intermediate, torch.Tensor):
            return_intermediate = False

        outputs = [None for i in range(self.num_blocks + 1)]
        # Down
        tmp = x
        for i in range(self.num_downs):
            tmp, skip = self.downs[i](tmp)
            outputs[i] = skip
        tmp = self.bridgeBlock(tmp)
        outputs[self.num_downs] = tmp
        # UP
        for i in range(self.num_downs):
            tmp = self.ups[i](tmp, outputs[self.num_downs - i - 1])
            outputs[self.num_downs + i + 1] = tmp
        # final
        tmp = self.lastLayer(tmp)
        assert outputs[-1] is None
        outputs[-1] = tmp
        self.frist = False
        if return_intermediate:
            try:
                return tmp, [outputs[i] for i in layers]
            except IndexError:
                assert False, f"there are only {len(outputs)} layers to choose from. Your chosen layers are {layers}"
        else:
            return tmp


if __name__ == "__main__":
    from torchsummary import summary

    model = UNet(1, 1, net_G_channel=64, net_G_depth=12)
    print(model)
    # model = UNet(1, 1, net_G_channel=2, net_G_depth=9)
    # model.cpu()
    # summary(model, (1, 2**7, 2**7))
    # model = UNet(1, 1, net_G_channel=2, net_G_depth=5)
    # summary(model, (1, 2**10, 2**10))
    # model = UNet(1, 1, net_G_channel=16, net_G_depth=9)
    # summary(model, (1, 2**10, 2**10))
    # model = UNet(1, 1, net_G_channel=2, net_G_depth=19)
    # summary(model, (1, 2**12, 2**12))
    # model = UNet(1, 1, net_G_channel=64, net_G_depth=9)
    # summary(model, (1, 2**7, 2**7))
