from build_model import DRANet_Encoder, Vector_convert, EfficientNet
from loss_function import Loss 
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchsummary import summary

class Trainer:
    def __init__(self, args, train_loader, test_loader) -> None:
        self.args = args
        self.train_dataset = train_loader
        self.test_dataset  = test_loader
        # self.writer        = SummaryWriter()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

    def set_default(self):
        # self.Enc_optim = Adam(E.parameters(), lr=1e-4)
        # self.Net_optim = Adam(N.parameters(), lr=1e-4)
        # self.Loss_optim = Adam(L.parameters(), lr=1e-4)
        self.model_optim = Adam(self.model.parameters(), lr=0.001)

    def set_networks(self):
        # self.Encoder = DRANet_Encoder()
        # self.Network = Vector_convert(256, self.args)
        # self.Model   = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        # self.Model   = torch.nn.Sequential(*(list(self.Model.children())[:-1])).cuda()
        self.model     = EfficientNet().to(self.device)

        # summary(self.model , (3, 224, 224))

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
        self.set_default()

        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_dataset[dset])

        for i in range(self.args.iter):     # not epoch using iteration
            batch_data = self.get_batch(batch_data_iter)
            imgs_1, imgs_2, labels =  dict(), dict(), dict()
            min_batch = self.args.batch
            for dset in self.args.datasets:
                imgs_1[dset], imgs_2[dset], labels[dset] = batch_data[dset]
                imgs_1[dset], imgs_2[dset], labels[dset] = imgs_1[dset].cuda(), imgs_2[dset].cuda(), labels[dset].cuda()
                if imgs_1[dset].size(0) < min_batch:
                    min_batch = imgs_1[dset].size(0)
            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs_1[dset], imgs_2[dset], labels[dset] = imgs_1[dset][:min_batch], imgs_2[dset][:min_batch], labels[dset][:min_batch]

            self.model_optim.zero_grad()

            class_loss = 0
            domain_loss = 0
            
            for dset in self.args.datasets:
                img_1 = imgs_1[dset]
                img_2 = imgs_2[dset]
                label = labels[dset]

                if dset == self.args.datasets[0]:
                    do_label = torch.zeros(min_batch)
                    do_label = do_label.type(torch.LongTensor)
                else:
                    do_label = torch.ones(min_batch)
                    do_label = do_label.type(torch.LongTensor)

                do_label = do_label.cuda()

                cls_output, dom_output = self.model(img_1, img_2)

                loss_1 = self.criterion(cls_output, label)
                loss_2 = self.criterion(dom_output, do_label)

                class_loss += loss_1
                domain_loss += loss_2

                if i % 10 == 0:
                    c_acc = self.get_acc(cls_output, label)
                    d_acc = self.get_acc(dom_output, do_label)
                    # d_acc = 0.000
                    print(f'{dset} Class Accuracy: {c_acc:.3f} Domain Accuracy: {d_acc:.3f}')

            loss = class_loss + domain_loss
            loss.backward()
            self.model_optim.step()

            if i % 10 == 0:
                print(f'[Loss] Class_loss: {class_loss:.3f} Domain_loss: {domain_loss:.3f}\n')
            if min_batch < self.args.batch:
                print('Test')

        '''
        for E in range(self.args.epoch):
            self.Model.cuda()
            for dset in self.args.datasets[0]:
                for i, (img1, img2, label) in enumerate(self.train_dataset[dset]):
                    image     = img1.cuda()
                    aug_image = img2.cuda()
                    label     = label.cuda()

                    self.Model_optim.zero_grad()
                    output1 = self.Model(image)
                    loss_1  = self.criterion(output1, label)
                    acc_1   = self.get_acc(output1, label)

                    loss_1.backward()
                    self.Model_optim.step()

                    output2 = self.Model(aug_image)
                    loss_2   = self.criterion(output2, label)
                    acc_2   = self.get_acc(output2, label)
                    if i % 10 == 0:
                        print(f'[{i}||{len(self.train_dataset[dset])}] Origin    Image  Image Loss: {loss_1:.3f} Image Acc: {acc_1:.3f}')
                        print(f'[{i}||{len(self.train_dataset[dset])}] transform Image  Image Loss: {loss_2:.3f}\n')
            
            test_acc = self.test(E)
        '''

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
            self.Loss_optim.step()
            
            if i % 10 == 0:
                print(f'[Loss]: {loss:.4f}')
                self.record_output('Loss/train', loss, i)
                for dset in self.args.datasets:
                    self.record_output(f'{dset}_Acc/train', acc[dset], i)
                    print(f'[Acc:{dset}]: {acc[dset]:.4f}')
        '''           
    
    def test(self, e):
        self.Model.eval()
        self.cls_clf.eval()
        self.dom_clf.eval()

        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.test_dataset[dset])

        for i in range(self.args.iter):     # not epoch using iteration
            batch_data = self.get_batch(batch_data_iter)
            imgs_1, imgs_2, labels =  dict(), dict(), dict()
            min_batch = self.args.batch
            for dset in self.args.datasets:
                imgs_1[dset], imgs_2[dset], labels[dset] = batch_data[dset]
                imgs_1[dset], imgs_2[dset], labels[dset] = imgs_1[dset].cuda(), imgs_2[dset].cuda(), labels[dset].cuda()
                if imgs_1.size(0) < min_batch:
                    min_batch = imgs_1.size(0)
            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs_1[dset], imgs_2[dset], labels[dset] = imgs_1[dset][:min_batch], imgs_2[dset][:min_batch], labels[dset][:min_batch]

            for dset in self.args.datasets:
                img_1 = imgs_1[dset]
                img_2 = imgs_2[dset]
                label = labels[dset]

                if dset == self.args.datasets[0]:
                    do_label = torch.zeros(min_batch)
                else:
                    do_label = torch.ones(min_batch)
                do_label = do_label.cuda()

                f1 = self.Model(img_1)
                f2 = self.Model(img_2)
                cls_output = self.cls_clf(f1)
                dom_output = self.dom_clf(f2)