import os
import random
import torch.utils.data

import torchvision.transforms as transforms
from PIL import Image

class make_dataset(torch.utils.data.Dataset):
    def __init__(self, phase, data, args, imsize) -> None:
        folder_path = os.path.join(args.project, 'Dataset', 'OfficeHomeDataset', data)
        self.class_list = os.listdir(folder_path)
        self.datalist = []
        self.imsize = imsize
        self.phase  = phase
        for cls in self.class_list:
            cls_path = os.path.join(folder_path, cls)
            imgs = os.listdir(cls_path)
            num_train = int(0.8*len(imgs))
            if phase == 'train':
                self.datalist.extend([(os.path.join(cls_path, name), cls) for name in imgs[:num_train]])
            else:
                self.datalist.extend([(os.path.join(cls_path, name), cls) for name in imgs[num_train:]])
        
        self.transform=transforms.Compose([transforms.Resize((imsize, imsize)),
                                           transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        path, label = self.datalist[idx]
        label = self.class_list.index(label)
        image = Image.open(path).convert('RGB')
        w, h = image.size
        
        _min = min([w, h])
        aug_transform = transforms.Compose([transforms.Resize((self.imsize,self.imsize)),
                                            transforms.ToTensor(),
                                            AddGaussianNoise(0.,1.)])
        if self.phase == 'train':
            image_1 = self.transform(image) 
            image_2 = aug_transform(image)
            return image_1, image_2, label
        else:
            image_1 = self.transform(image)
            return image_1, label

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)