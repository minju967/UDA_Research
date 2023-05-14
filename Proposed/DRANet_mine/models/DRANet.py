import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from .batchinstancenorm import BatchInstanceNorm2d as Normlayer
import functools
from functools import partial
import torchvision.transforms as ttransforms
from torchvision.models import resnet50, ResNet50_Weights

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

class Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Encoder, self).__init__()
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

    def forward(self, x):
        output = self.model(x)
        return output

class Separator(nn.Module):
    def __init__(self, imsize, converts, ch=64, down_scale=2):
        super(Separator, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            # nn.AvgPool2d(32)
        )
        self.pool = nn.AvgPool2d(32)
        self.w = nn.ParameterDict()
        w, h = imsize

    def forward(self, f_dict, converts=None):
        contents, styles = dict(), dict()

        for dset in f_dict.keys():
            contents_f = self.conv(f_dict[dset])  # equals to F - wS(F) see eq.(2)
            styles_f = f_dict[dset] - contents_f  # equals to wS(F)

            contents_f   = self.pool(contents_f)
            styles_f = self.pool(styles_f) 

            styles[dset]     = styles_f.view(contents_f.size(0), -1)
            contents[dset]   = contents_f.view(contents_f.size(0), -1)

        return contents, styles

class Generator(nn.Module):
    def __init__(self, channels=512):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            spectral_norm(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.Tanh()
        )

    def forward(self, content, style):
        return self.model(content+style)

class Classifier(nn.Module):
    def __init__(self, channels=3, num_classes=10):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(12288, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

class Classifier_OfficeHome(nn.Module):
    # 224, 224
    def __init__(self, num_cls):
        super(Classifier_OfficeHome, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)        
        in_dim = self.model.fc.in_features   # 2048

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, num_cls))

    def forward(self, input):
        feature = self.model(input)
        output  = self.fc(feature)

        return output
    
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2,7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7,12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12,21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21,25):
            self.to_relu_4_2.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_4_2(h)
        h_relu_4_2 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_4_2)
        return out

class Discriminator_USPS(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator_USPS, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

class Discriminator_MNIST(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator_MNIST, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0),-1)
        output = self.fc(output)
        return output

class Discriminator_OfficeHome(nn.Module):
    # 224, 224
    def __init__(self):
        super(Discriminator_OfficeHome, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)        
        in_dim = self.model.fc.in_features   # 2048

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, 1))

    def forward(self, input):
        feature = self.model(input)
        output  = self.fc(feature)
        return output