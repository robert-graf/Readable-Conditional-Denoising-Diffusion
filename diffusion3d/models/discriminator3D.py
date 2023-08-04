from torch import nn
import torch.nn.functional as F
import torch


class ClassPatchDiscriminator3D(nn.Module):
    def __init__(self, input_nc, num_classes=2, depth=4, channels=64, kernel_size=4):
        super(ClassPatchDiscriminator3D, self).__init__()
        # nn.Tanh(),
        model = [nn.Conv3d(input_nc, channels, kernel_size=kernel_size, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        # A bunch of convolutions one after another
        for _ in range(depth - 1):
            model += [
                nn.Conv3d(channels, channels * 2, kernel_size=kernel_size, stride=2, padding=1),
                nn.InstanceNorm3d(128),
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
        model += [nn.Conv3d(channels, num_classes, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)


if __name__ == "__main__":
    disc = ClassPatchDiscriminator3D(1, 2)
    inp = torch.ones((5, 1, 100, 100, 100))
    out = disc(inp)
    print(out.shape)
