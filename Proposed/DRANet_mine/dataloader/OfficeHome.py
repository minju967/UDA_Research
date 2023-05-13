import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import random
import matplotlib.pyplot as plt


class OfficeHome(torch.utils.data.Dataset):
    def __init__(self, path, train, transform=None):
        self.transform = transform
        self.cls_list = os.listdir(path)
        all_data = []
        for cls in self.cls_list:
            data = glob.glob(os.path.join(path, cls)+'\\*.png')
            num_train = int(len(data)*0.8)
            random.shuffle(data)
            if train == 'train':
                all_data.extend(data[:num_train])
            else:
                all_data.extend(data[num_train:])
        
        self.all_data = all_data

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        path = self.all_data(index)
        image = self.transform(Image.open(path).convert('RGB'))
        label = self.cls_list.index(image.split('\\')[-2])

        return image, label
