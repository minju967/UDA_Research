import torch
import torch.nn as nn
import functools
from batchinstancenorm import BatchInstanceNorm2d as Normlayer


## DRANet Encoder
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                bin(filters)
            )

    def forward(self, x):
        output = self.main(x)
        output += self.shortcut(x)
        return output
    
class DRANet_Encoder(nn.Module):
    # https://github.com/Seung-Hun-Lee/DRANet
    # https://arxiv.org/abs/2103.13447
    # MNIST-M(clf), Cityscapes(seg), GTA5(seg)

    def __init__(self, channels=3):
        super(DRANet_Encoder, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )
        self.Linear = nn.Linear(64*112*112, 64)
        

    def forward(self, x):
        # f_map: [batch, 64, 112, 112]
        # global vector: [batch, 64]
        f_map = self.model(x) 
        vector = self.Linear(f_map.view(x.shape[0], -1))
        return f_map, vector
    
## Deep InfoMax Encoder

class Vector_convert(nn.Module):
    # Contribution 1
    # This module should convert vector to only Content vector or Domain vector
    def __init__(self, in_size, args):
        super(Vector_convert, self).__init__()
        self.args  = args
        self.network = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, in_size),
            nn.ReLU()
        )
    
    def forward(self, vector1, vector2):
        # vec_1(dict)
        # vec_2(dict)
        content, domain = dict(), dict()
        for dset in self.args.datasets:
            content_vec = self.network(vector1[dset])
            domain_vec = self.network(vector2[dset])
            
            content[dset] = content_vec
            domain[dset]  = domain_vec

        return content, domain
