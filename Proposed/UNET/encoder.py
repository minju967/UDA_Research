import torch
import torch.nn as nn
from decoder import Decoder

class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.source, self.target = self.opts.domains[0], self.opts.domains[1]

        self.E_conv_1 = nn.Sequential(
                        nn.Conv2d(3, 32, 3),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        self.E_conv_2 = nn.Sequential(
                        nn.Conv2d(32, 64, 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        
        self.E_conv_3 = nn.Sequential(
                        nn.Conv2d(64, 128, 3),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        self.E_conv_4 = nn.Sequential(
                        nn.Conv2d(128, 256, 3),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        self.E_conv_5 = nn.Sequential(
                        nn.Conv2d(256, 512, 3),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        
        self.E_conv_6 = nn.Sequential(
                        nn.Conv2d(512, 512, 3),
                        nn.BatchNorm2d(512),
                        nn.ReLU())
        self.maxpool =  nn.MaxPool2d(kernel_size=6, return_indices=True)

        self.fc       = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.ReLU())

        self.decoder = dict()
        for dset in self.opts.domains:
            self.decoder[dset] = Decoder().cuda()

    def forward(self, x, dset):
        # print(x.size()) [b, 3, 256, 256]
        output1  = self.E_conv_1(x)
        output2  = self.E_conv_2(output1)
        # print(output2.size()) [b, 64, 64, 64]
        output3  = self.E_conv_3(output2)
        output4  = self.E_conv_4(output3)
        # print(output4.size()) [b, 256, 16, 16]
        output5  = self.E_conv_5(output4)
        output6  = self.E_conv_6(output5)   
        # print(output6.size()) [b, 512, 6, 6]

        pooling, indices  = self.maxpool(output6)    # [b, 512, 1, 1]
        flatten  = pooling.view(x.size(0), -1)  # [b, 512]
        output   = self.fc(flatten)         # [b, 256]

        latents  = [output5 ,output4, output3, output2, output1]

        image = self.decoder[dset](output, latents, indices)
        
        return image