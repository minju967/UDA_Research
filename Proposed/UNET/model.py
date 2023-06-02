import torch
import torch.nn as nn
from torchsummary import summary

from decoder import Decoder

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.E_conv_1 = nn.Sequential(
                        nn.Conv2d(3, 32, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        self.E_conv_2 = nn.Sequential(
                        nn.Conv2d(32, 64, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        
        self.E_conv_3 = nn.Sequential(
                        nn.Conv2d(64, 128, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        self.E_conv_4 = nn.Sequential(
                        nn.Conv2d(128, 256, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        self.E_conv_5 = nn.Sequential(
                        nn.Conv2d(256, 512, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2, 1))
        
        self.E_conv_6 = nn.Sequential(
                        nn.Conv2d(512, 512, 3),
                        nn.ReLU())
        
        self.decoder  = Decoder()

    def forward(self, x):
        output1  = self.E_conv_1(x)
        output2  = self.E_conv_2(output1)
        output3  = self.E_conv_3(output2)
        output4  = self.E_conv_4(output3)
        output5  = self.E_conv_5(output4)
        output   = self.E_conv_6(output5)
        latents  = [output5 ,output4, output3, output2, output1]

        image = self.decoder(output, latents)

        return image



def main():
    encoder = Encoder()
    train_dataset = None
    test_dataset = None

    train_dataloader = None
    test_dataloader  = None
