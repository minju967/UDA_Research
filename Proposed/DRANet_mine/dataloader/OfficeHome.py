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