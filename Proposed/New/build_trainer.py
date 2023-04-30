from build_model import DRANet_Encoder, Vector_convert
from loss_function import Loss 
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torchvision.models import resnet50


class Trainer:
    def __init__(self, args, train_loader, test_loader) -> None:
        self.args = args
        self.train_dataset = train_loader
        self.test_dataset  = test_loader
        # self.writer        = SummaryWriter()

    def set_default(self, E, N, L, M):
        self.Enc_optim = Adam(E.parameters(), lr=1e-4)
        self.Net_optim = Adam(N.parameters(), lr=1e-4)
        self.Loss_optim = Adam(L.parameters(), lr=1e-4)
        self.Model_optim = Adam(M.parameters(), lr=1e-4)

    def set_networks(self):
        self.Encoder = DRANet_Encoder()
        self.Network = Vector_convert(256, self.args)
        self.Model         = resnet50(pretrained=True)
        fc_feature         = self.Model.fc.in_features
        self.Model.fc      = nn.Linear(fc_feature, 65)

    def set_lossfunction(self):
        self.loss = Loss(self.args)
        self.criterion = nn.CrossEntropyLoss()
    
    def record_output(self, tag, val, iter):
        self.writer.add_scalar(tag, val, iter)

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.args.datasets:
            try:
                batch_data[dset] = next(batch_data_iter[dset])
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_dataset[dset])
                batch_data[dset] = next(batch_data_iter[dset])
        
        return batch_data

    def extract_feature(self, imgs1, imgs2):
        features, global_vec, domain_vec = dict(), dict(), dict()
        for dset in self.args.datasets:
            fea_map1, g_vec  = self.Encoder(imgs1[dset])    # global vector (domain + content)
            fea_map2, d_vec  = self.Encoder(imgs2[dset])    # domain vector (content)

            # features[dset] = fea_map2
            features[dset] = fea_map1
            global_vec[dset] = g_vec
            domain_vec[dset] = d_vec

        content, domain = self.Network(global_vec, domain_vec)        

        return features, content, domain 

    def get_acc(self, output, label):
        _, pred = torch.max(output.data, 1)
        total   = label.size(0)
        correct = (pred == label).sum().item()
        acc = 100 * (correct//total)
        return acc
        
    def train(self):
        self.set_networks()
        self.set_lossfunction()
        self.set_default(self.Encoder, self.Network, self.loss, self.Model)
        
        for E in range(self.args.epoch):
            for dset in self.args.datasets[0]:
                for i, (img1, img2, label) in enumerate(self.train_dataset[dset]):
                    image     = img1
                    aug_image = img2
                    label     = label

                    self.Model_optim.zero_grad()
                    output1 = self.Model(image)
                    loss_1  = self.criterion(output1, label)
                    acc_1   = self.get_acc(output1, label)

                    loss_1.backward()
                    self.Model_optim.step()

                    output2 = self.Model(aug_image)
                    acc_2   = self.get_acc(output2, label)
                    if i % 10 == 0:
                        print(f'[{i}||{len(self.train_dataset[dset])}] Origin    Image  Iter_Acc: {acc_1:.3f} Image Loss: {loss_1:.3f}')
                        print(f'[{i}||{len(self.train_dataset[dset])}] transform Image  Iter_Acc: {acc_2:.3f}')

        '''
        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_dataset[dset])
    
        for i in range(self.args.iter):     # not epoch using iteration
            batch_data = self.get_batch(batch_data_iter)
            imgs_1, imgs_2, labels =  dict(), dict(), dict()
            min_batch = self.args.batch
            for dset in self.args.datasets:
                imgs_1[dset], imgs_2[dset], labels[dset] = batch_data[dset]
                # imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                if imgs_1[dset].size(0) < min_batch:
                    min_batch = imgs_1[dset].size(0)
            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs_1[dset], labels[dset] = imgs_1[dset][:min_batch], labels[dset][:min_batch]

            # F_map: feature map thatt output encoder
            # contents: only content vector
            # domains: only domain vector
            self.Enc_optim.zero_grad()
            self.Net_optim.zero_grad()
            self.Loss_optim.zero_grad()
            F_map, contents, domains = self.extract_feature(imgs_1, imgs_2)
            
            ## loss function define
            loss, acc = self.loss(F_map, contents, domains, labels)
            loss.backward()
            self.Enc_optim.step()
            self.Net_optim.step()
            self.Loss_optim.zero_grad()
            
            if i % 10 == 0:
                print(f'[Loss]: {loss:.4f}')
                self.record_output('Loss/train', loss, i)
                for dset in self.args.datasets:
                    self.record_output(f'{dset}_Acc/train', acc[dset], i)
                    print(f'[Acc:{dset}]: {acc[dset]:.4f}')
        '''           
            