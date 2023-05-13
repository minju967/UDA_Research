<<<<<<< HEAD
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
=======
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

class OfficeHome(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        self.domain_path = root
        self.train = train
        self.transform = transform
        data_dict = self.get_datalist(path=root)
        self.class_list = list(data_dict.keys())
        
        train_data = []
        test_data  = []
        for cls in self.class_list:
            num_train = int(len(data_dict[cls]) * 0.8)
            train_data.extend(list(map(lambda x:(cls, x), data_dict[cls][:num_train])))
            test_data.extend(list(map(lambda x:(cls, x), data_dict[cls][num_train:])))
        if train:
            self.image_dir = train_data
        else:
            self.image_dir = test_data

    def get_datalist(self, path):
        data = dict()
        class_list = os.listdir(path)
        for cls in class_list:
            data[cls] = os.listdir(os.path.join(path, cls))
        return data
    
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, idx):
        cls, image = self.image_dir[idx]
        image = os.path.join(self.domain_path, cls, image)
        image = self.transform(Image.open(image).convert('RGB'))
        label = self.class_list.index(cls)
        return image, label
>>>>>>> fc9af6d7a78531b2f9f5e2b70400e4680faf329d
