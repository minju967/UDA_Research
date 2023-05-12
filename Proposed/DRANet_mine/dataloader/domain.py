import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset

from PIL import Image

# dsets: M MM / A CA

class make_dataset(torch.utils.data.Dataset):
    def __init__(self, datasets, root, train, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.domain = dict()

        for i, dset in enumerate(datasets):
            self.domain[dset] = i
        domain_dict = {'A':'Art', 'CA':'Clipart', 'P':'Product', 'R':'Real world'}
        data = []
        dset_path = ''
        for dset in datasets:
            if dset in ['A', 'CA', 'P', 'R']:
                dset_path = os.path.join(self.root, 'OfficeHomeDataset', domain_dict[dset])

            elif dset == 'MM':
                if train:
                    dset_path = os.path.join(self.root, 'mnist_m', 'mnist_m_train')
                else:
                    dset_path = os.path.join(self.root, 'mnist_m', 'mnist_m_test')
            
            for cls in os.listdir(dset_path):
                cls_file = glob.glob(os.path.join(dset_path, cls) + '\\*.jpg')
                if len(cls_file) > 0:
                    num_train = int(len(cls_file) * 0.8)
                    if train:
                        data.extend(cls_file[:num_train])
                    else:
                        data.extend(cls_file[num_train:])
                else:
                    cls_file = glob.glob(os.path.join(dset_path, cls) + '\\*.png')
                    num_train = int(len(cls_file) * 0.8)
                    if train:
                        data.extend(cls_file[:num_train])
                    else:
                        data.extend(cls_file[num_train:])
    
        self.all_data = data
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        img_path = self.all_data(idx)
        domain   = self.domain[img_path.split('\\')[-3]]
        image    = self.transform(Image.open(img_path).convert('RGB'))

        return image, domain