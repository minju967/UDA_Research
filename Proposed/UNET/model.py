import torch
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary
from torch.utils.data import DataLoader
from torch.nn import functional as F

from decoder import Decoder

from utils.data_util import get_imagelist
from utils.common import show_image
from dataset.images_dataset import UNET_ImagesDataset

import os
import datetime

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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

class Trainer():
    def __init__(self,save_folder) -> None:
        super(Trainer, self).__init__()

        self.encoder = Encoder()
        summary(self.encoder, (3,256, 256))
        
        train_list, test_list = get_imagelist(domain='amazon')
        train_dataset = UNET_ImagesDataset(files=train_list, train=True)
        test_dataset  = UNET_ImagesDataset(files=test_list, train=False)

        print(f'Number of Train dataset: {len(train_dataset)}')
        print(f'Number of Test  dataset: {len(test_dataset)}\n')

        self.train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0, pin_memory=True)
        self.test_dataloader  = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

        self.optimizer = optim.Adam(self.encoder.parameters(), lr=1e-3)

        # loss
        self.mse_loss = nn.MSELoss().eval()
        self.min_loss = 100000

        self.checkpoint_dir = save_folder

    def loss_function(self, recon, ori):
        loss_l2 = F.mse_loss(recon, ori)
        return loss_l2

    def train(self, epoch):
        self.encoder.train()
        train_loss = 0
        for batch_idx, (imgs) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            recon_batch = self.encoder(imgs)
            loss = self.loss_function(recon_batch, imgs)
            loss.backward()
            train_loss += float(loss.item())
            self.optimizer.step()

            if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_dataloader)}] Loss: {loss.item():.3f}')
        
        print(f'\n====> Epoch: {epoch} Average Loss: {train_loss/len(self.train_dataloader):.3f}')

    def test(self, epoch):
        self.encoder.eval()
        test_loss = 0
        save_dir=os.path.join(self.checkpoint_dir, f'images\\test')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        with torch.no_grad():
            for batch_idx, (input_imgs) in enumerate(self.test_dataloader):
                recon_batch = self.encoder(input_imgs)
                test_loss += self.loss_function(recon=recon_batch, ori=input_imgs).item()

                if batch_idx in [10, 30, 50]:
                    self.make_images(save_dir, origin=input_imgs, reconst=recon_batch, epoch=epoch)

        print(f'\n====> Epoch: {epoch} Test dataset Loss: {test_loss/len(self.test_dataloader):.3f}')
        avg_loss = test_loss / len(self.test_dataloader)
        
        if avg_loss < self.min_loss:
            save_name = f'best_model_{epoch}.pt'
            save_dict = self.__get_save_dict()
            checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
            torch.save(save_dict, checkpoint_path)

    def make_images(self, save_dir, origin, reconst, display=2, epoch=0):
        path = os.path.join(save_dir, f'{epoch}.jpg')
        show_image(origin=origin, reconst=reconst, path=path)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.encoder.state_dict(),
            'opts': vars(self.optimizer)
        }
        return save_dict

def main():
    now = datetime.datetime.now()
    exp_dir = now.strftime('%m%d_%H%M')

    save_dict = f'save_Exp\{exp_dir}'
    if not os.path.isdir(save_dict):
        os.makedirs(save_dict, exist_ok=True)
    else:
        Exception(f"{save_dict} Directory Already Exist. Bye Bye~~")

    trainer = Trainer(save_folder=save_dict)
    epochs = 100

    for epoch in range(epochs):
       trainer.train(epoch)
       trainer.test(epoch)

if __name__ == '__main__':
	main()
