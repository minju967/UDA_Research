import torch
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary
from torch.utils.data import DataLoader
from torch.nn import functional as F


from encoder import Encoder
from utils.data_util import get_imagelist
from utils.common import show_image
from train_options import TrainOptions
from dataset.images_dataset import UNET_ImagesDataset

import json
import os
import pprint
import datetime


class Trainer():
    def __init__(self,save_folder, opt) -> None:
        super(Trainer, self).__init__()
        self.opts = opt
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(self.opts).to(self.device)
        
        self.train_dataset, self.test_dataset = dict(), dict()
        for dset in opt.domains:
            train_list, test_list = get_imagelist(domain=dset)    
            self.train_dataset[dset] = UNET_ImagesDataset(files=train_list, train=True)
            self.test_dataset[dset]  = UNET_ImagesDataset(files=test_list, train=False)

            print(f'[{dset}] Number of Train dataset: {len(self.train_dataset[dset])}')
            print(f'[{dset}] Number of Test  dataset: {len(self.test_dataset[dset])}\n')

        self.train_dataloader, self.test_dataloader = dict(), dict()
        for dset in opt.domains:
            self.train_dataloader[dset] = DataLoader(self.train_dataset[dset], batch_size=3, shuffle=True, num_workers=0, pin_memory=True)
            self.test_dataloader[dset]  = DataLoader(self.test_dataset[dset], batch_size=3, shuffle=False, num_workers=0)

        self.optimizer = optim.Adam(self.encoder.parameters(), lr=1e-3)

        # loss
        self.mse_loss = nn.MSELoss().eval()
        self.min_loss = 100000

        self.checkpoint_dir = save_folder

    def loss_function(self, recon, ori):
        loss_l2 = F.mse_loss(recon, ori)
        return loss_l2

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.opts.domains:
            try:
                batch_data[dset] = next(batch_data_iter[dset])
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_dataloader[dset])
                batch_data[dset] = next(batch_data_iter[dset])
        return batch_data

    def train(self):
        self.encoder.train()
        batch_data_iter = dict()
        for dset in self.opts.domains:
            batch_data_iter[dset] = iter(self.train_dataloader[dset])

        for i in range(self.opts.iter):
            loss = 0
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.opts.batch
            for dset in self.opts.domains:
                imgs[dset] = batch_data[dset]
                imgs[dset] = imgs[dset].cuda()
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            
            if min_batch < self.opts.batch:
                for dset in self.opts.domains:
                    imgs[dset] = imgs[dset][:min_batch].cuda()
            
            # train
            self.optimizer.zero_grad()
            recons_img = dict()
            for dset in self.opts.domains:
                recons_img[dset] = self.encoder(imgs[dset], dset=dset)
                recon_loss = self.loss_function(recons_img[dset], imgs[dset])
                if i % 100 == 0:
                    print(f'[{dset}] Reconst_loss: {recon_loss:.3f}')
                loss += recon_loss

            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f'[Train] {i} Iteration_Loss: {loss.item():.3f}')
                self.test(i)

    def test(self, iteration):
        target = self.opts.domains[1]
        self.encoder.eval()
        test_loss = 0
        save_dir=os.path.join(self.checkpoint_dir, 'image', 'test')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(self.checkpoint_dir, 'pt_file'))

        with torch.no_grad():
            for batch_idx, (imgs) in enumerate(self.test_dataloader[target]):
                imgs = imgs.to(self.device).float()
                recon_batch = self.encoder(imgs, dset=target)
                test_loss += self.loss_function(recon=recon_batch, ori=imgs).item()

                if batch_idx == 30 and iteration % 300 == 0:
                    self.make_images(save_dir, origin=imgs, reconst=recon_batch, epoch=iteration)

        print(f'\n====> Epoch: {iteration} Test dataset Loss: {test_loss/len(self.test_dataloader):.3f}')
        avg_loss = test_loss / len(self.test_dataloader)
        
        if avg_loss < self.min_loss:
            save_name = f'best_model.pt'
            save_dict = self.__get_save_dict()
            checkpoint_path = os.path.join(self.checkpoint_dir, 'pt_file', save_name)
            torch.save(save_dict, checkpoint_path)

    def make_images(self, save_dir, origin, reconst, epoch=0):
        path = os.path.join(save_dir, f'{epoch}.jpg')
        show_image(origin=origin, reconst=reconst, path=path, display=2)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.encoder.state_dict(),
            'opts': vars(self.optimizer)
        }
        return save_dict

def main():
    opts = TrainOptions().parse()

    now = datetime.datetime.now()
    exp_dir = now.strftime('%m%d_%H%M')

    save_dict = f'.\\save_Exp\\{exp_dir}'
    if not os.path.isdir(save_dict):
        os.makedirs(save_dict, exist_ok=True)
    else:
        Exception(f"{save_dict} Directory Already Exist. Bye Bye~~")

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    with open(os.path.join(save_dict, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    trainer = Trainer(save_folder=save_dict, opt=opts)
    trainer.train()

if __name__ == '__main__':
    main()
