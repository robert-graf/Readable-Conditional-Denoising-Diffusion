import random
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from math import floor, ceil


class SquarePad:
    def __init__(self, size) -> None:
        self.size = size
        pass

    def __call__(self, image):
        w, h = image.shape[-2], image.shape[-1]
        max_wh = self.size
        hp = max((max_wh - w) / 2, 0)
        vp = max((max_wh - h) / 2, 0)
        padding = (int(floor(vp)), int(ceil(vp)), int(floor(hp)), int(ceil(hp)))
        # print(padding,w,h)
        x = F.pad(image, padding, value=0, mode="constant")
        # print(x.shape)
        return x


class ReplayBuffer:
    def __init__(self, max_size=50, paired=False):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data_real = []
        self.data_fake = []
        self.peek_buffer = None
        self.paired = paired

    def push_and_pop(self, fake, real) -> torch.Tensor:
        to_return_fake = []
        to_return_real = []

        for ele_fake, ele_real in zip(fake.data, real.data):
            ele_fake = torch.unsqueeze(ele_fake, 0)
            ele_real = torch.unsqueeze(ele_real, 0)
            if len(self.data_real) < self.max_size:
                self.data_fake.append(ele_fake)
                self.data_real.append(ele_real)
                to_return_fake.append(ele_fake)
                to_return_real.append(ele_real)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return_fake.append(self.data_fake[i].clone())
                    to_return_real.append(self.data_real[i].clone())
                    self.data_fake[i] = ele_fake
                    self.data_real[i] = ele_real
                else:
                    to_return_fake.append(ele_fake)
                    to_return_real.append(ele_real)
        if self.paired:
            self.peek_buffer = torch.cat([torch.cat(to_return_fake), torch.cat(to_return_real)], dim=1)

        else:
            self.peek_buffer = torch.cat(to_return_fake)

        return self.peek_buffer

    def forced_push_info(self, data):
        idxs = set()
        l = min(data.shape[0], self.max_size / 2)
        while l == len(idxs):
            idxs.add(random.randint(0, self.max_size - 1))

        for i, j in zip(idxs, range(len(idxs))):
            self.data_real[i] = data[j]

    def peek(self):
        assert self.peek_buffer is not None
        return self.peek_buffer


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
