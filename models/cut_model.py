import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features, net_G_drop_out):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(net_G_drop_out),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, net_G_depth=9, net_G_downsampling=2, net_G_channel=64, net_G_drop_out=0.5, **args):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, net_G_channel, 7),
            nn.InstanceNorm2d(net_G_channel),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = net_G_channel
        out_features = in_features * 2
        for _ in range(net_G_downsampling):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(net_G_depth):
            model += [ResidualBlock(in_features, net_G_drop_out)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(net_G_downsampling):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(in_features, output_nc, 7), nn.Tanh()]
        # model += [nn.Conv2d(in_features, output_nc, 1), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc, depth=4, channels=64):
        super(Discriminator, self).__init__()
        # nn.Tanh(),
        model = [nn.Conv2d(input_nc, channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        # A bunch of convolutions one after another
        for _ in range(depth - 1):
            model += [
                nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels *= 2

        """
        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        
        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        """
        # FCN classification layer
        model += [nn.Conv2d(channels, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
