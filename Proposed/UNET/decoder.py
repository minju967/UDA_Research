import torch
import torch.nn as nn
from torchsummary import summary

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
                        nn.Linear(256, 512),
                        nn.ReLU())
        
        self.upsample = nn.MaxUnpool2d(6)

        self.D_convt_1 = nn.Sequential(
                        nn.ConvTranspose2d(512, 512, 3),
                        nn.BatchNorm2d(512),
                        nn.ReLU())
        self.D_conv_1 = nn.Sequential(
                        nn.Conv2d(1024, 512, 3, 1, 1),
                        nn.BatchNorm2d(512),
                        nn.ReLU())
        
        self.D_convt_2 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU())
        self.D_conv_2 = nn.Sequential(
                        nn.Conv2d(512, 256, 3, 1, 1),
                        nn.BatchNorm2d(256),
                        nn.ReLU())

        self.D_convt_3 = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU())
        self.D_conv_3 = nn.Sequential(
                        nn.Conv2d(256, 128, 3, 1, 1),
                        nn.BatchNorm2d(128),
                        nn.ReLU())

        self.D_convt_4 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.D_conv_4 = nn.Sequential(
                        nn.Conv2d(128, 64, 3, 1, 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())

        self.D_convt_5 = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU())
        self.D_conv_5 = nn.Sequential(
                        nn.Conv2d(64, 32, 3, 1, 1),
                        nn.BatchNorm2d(32),
                        nn.ReLU())

        self.convt_6 = nn.Sequential(
                        nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
                        nn.Sigmoid())
    
    def forward(self, code, latents, indices):
        input = self.fc(code)
        flatten = input.view(code.size(0), input.size(1), 1, 1)
        input_d = self.upsample(flatten, indices)

        output1 = self.D_convt_1(input_d)
        # concat1 = torch.cat((output1, latents[0]), dim=1)   # 1024, 8, 8
        # output1 = self.D_conv_1(concat1)

        output2 = self.D_convt_2(output1)
        # concat2 = torch.cat((output2, latents[1]), dim=1)   # 512, 16, 16
        # output2 = self.D_conv_2(concat2)

        output3 = self.D_convt_3(output2)   # 128, 32, 32
        concat3 = torch.cat((output3, latents[2]), dim=1)   # 
        output3 = self.D_conv_3(concat3)

        output4 = self.D_convt_4(output3)   # 64, 64, 64
        concat4 = torch.cat((output4, latents[3]), dim=1)
        output4 = self.D_conv_4(concat4)

        output5 = self.D_convt_5(output4)
        concat5 = torch.cat((output5, latents[4]), dim=1)
        output5 = self.D_conv_5(concat5)

        image = self.convt_6(output5)

        return image