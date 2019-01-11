from collections import OrderedDict
import torch
from torch import nn
from spp import SPPLayer


class CNN(nn.Module):

    def __init__(self, spp_level=3, num_class=6):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2 ** (i * 2)

        features = []
        architecture = [
            [1, 8, 3, 2, 0],
            [8, 32, 3, 2, 0],
            # [64, 128],
            [32, 64, 3, 2, 0],
            [64, 128],
            [128, 256, 3, 2, 0],
            [256, 512],
            [512, 128, 3, 2, 0]
        ]
        for arch in architecture:
            features.extend(self.block(*arch))

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.num_grids * 128, self.num_class)),
        ]))

        self.feature = nn.Sequential(*features)
        self.spp = SPPLayer(spp_level)
        self.top = classifier

    def forward(self, x):
        x = self.feature(x)
        x = self.spp(x)
        x = self.top(x)
        return x

    def block(self, in_ch, out_ch, k=3, s=1, pad=1, use_bias=False):
        return [
            nn.Conv2d(in_ch, in_ch, groups=in_ch, kernel_size=k, stride=s, padding=pad, bias=use_bias),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch)
        ]


if __name__ == '__main__':
    model = CNN()