from __future__ import print_function
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.backends.cudnn
import torch.optim as optim
import numpy as np
import os
import sys
from utils import *


from models.mine import Mine
from models import *
from loss_functions import *
from dataset import get_dataset_NB, get_dataset_OH

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# converts:M2MM --> MM2M
# 모든 Network는 각각
'''
    1. MNIST <-> MNIST-M
        train_converts = ['M2MM', 'MM2M']
        test_converts = ['M2MM', 'MM2M']
        tensorboard_converts = ['M2MM', 'MM2M']
'''
def set_converts(datasets):
    training_converts, test_converts = [], []
    center_dset = datasets[0]
    for source in datasets:  # source
        if not center_dset == source:
            training_converts.append(center_dset + '2' + source)
            training_converts.append(source + '2' + center_dset)
    
        for target in datasets:  # target
            if not source == target:
                test_converts.append(source + '2' + target)
    
        tensorboard_converts = test_converts

    return training_converts, test_converts, tensorboard_converts

class Trainer:
    def __init__(self, args):
        self.args = args
        self.training_converts, self.test_converts, self.tensorboard_converts = set_converts(args.datasets)
        print(self.training_converts)
        self.imsize = (args.imsize, args.imsize)
        self.acc = dict()
        self.best_acc = dict()
        for cv in self.test_converts:
                self.best_acc[cv] = 0.

        # data loader
        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        if self.args.d_type == 'NB':
            for dset in self.args.datasets:
                train_loader, test_loader = get_dataset_NB(dataset=dset, batch=self.args.batch,
                                                        imsize=self.imsize, workers=self.args.workers)
                self.train_loader[dset] = train_loader
                self.test_loader[dset] = test_loader
                self.num_cls = 10
        else:
            for dset in self.args.datasets:
                train_loader, test_loader = get_dataset_OH(dataset=dset, batch=self.args.batch,
                                                        imsize=self.imsize, workers=self.args.workers)
                self.train_loader[dset] = train_loader
                self.test_loader[dset] = test_loader
                self.num_cls = 65

        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.loss_fns = Loss_Functions(args)

        self.writer = SummaryWriter('./tensorboard/%s/%s' % (args.task, args.ex))
        self.logger = getLogger()
        self.checkpoint = './checkpoint/%s/%s' % (args.task, args.ex)
        self.step = 0

    def set_default(self):
        torch.backends.cudnn.benchmark = True

        ## Random Seed ##
        print("Random Seed: ", self.args.manualSeed)
        seed(self.args.manualSeed)
        torch.manual_seed(self.args.manualSeed)
        torch.cuda.manual_seed_all(self.args.manualSeed)

        ## Logger ##
        file_log_handler = FileHandler(self.args.logfile)
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = StreamHandler(sys.stdout)
        self.logger.addHandler(stderr_log_handler)
        self.logger.setLevel('INFO')
        formatter = Formatter()
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

    def save_networks(self):
        if not os.path.exists(self.checkpoint+'/%d' % self.step):
            os.mkdir(self.checkpoint+'/%d' % self.step)

        for key in self.nets.keys():
            torch.save(self.nets[key].state_dict(), self.checkpoint + '/%d/net%s.pth' % (self.step, key))
        
        torch.save(self.MI_net.state_dict(), self.checkpoint + '/%d/MI.pth' % (self.step))

    def load_networks(self, step):
        self.step = step
        for key in self.nets.keys():
            if self.args.task != 'MI_net':
                self.nets[key].load_state_dict(torch.load(self.checkpoint + '/%d/net%s.pth' % (step, key)))
            else:
                if key == 'MI':
                    pass
                else:
                    self.nets[key].load_state_dict(torch.load(self.checkpoint + '/%d/net%s.pth' % (step, key)))

    def set_networks(self):
        if self.args.task == 'DRAnet' or self.args.task == 'DRA-MI':
            self.nets['E'] = Encoder()
            self.nets['G'] = Generator()
            self.nets['S'] = Separator(self.imsize, self.training_converts)
            self.nets['D'] = dict()

            # nets['D']: Discriminator
            # Discriminate that the generated image is real or fake
            for dset in self.args.datasets:
                if dset == 'U':
                    self.nets['D'][dset] = Discriminator_USPS()
                elif dset in ['M', 'MM']:
                    self.nets['D'][dset] = Discriminator_MNIST()
                else:
                    self.nets['D'][dset] = Discriminator_OfficeHome()

            self.nets['T'] = dict()

            for cv in self.test_converts:
                if self.args.d_type == 'NB':
                    self.nets['T'][cv] = Classifier()
                else:
                    self.nets['T'][cv] = Classifier_OfficeHome(num_cls=65)

            # initialization
            for net in self.nets.keys():
                if net == 'D':
                    for dset in self.args.datasets:
                        init_params(self.nets[net][dset])
                elif net == 'T':
                    for cv in self.test_converts:
                        init_params(self.nets[net][cv])
                else:
                    init_params(self.nets[net])

            self.nets['P'] = VGG19()

            for net in self.nets.keys():
                if net == 'D':
                    for dset in self.args.datasets:
                        self.nets[net][dset].cuda()
                elif net == 'T':
                    for cv in self.test_converts:
                        self.nets[net][cv].cuda()
                else:
                    self.nets[net].cuda()

            if self.args.task == 'DRA-MI':
                self.nets['MI'] = self.get_MINE()
        
        elif self.args.task == 'MI_net':
            self.nets['E'] = Encoder()
            self.nets['S'] = Separator_MI(self.imsize, self.training_converts)
            self.nets['MI'] = self.get_MINE()

            for net in self.nets.keys():
                init_params(self.nets[net])
            
            for net in self.nets.keys():
                self.nets[net].cuda()

    def get_MINE(self):
        T = nn.Sequential(
            nn.Linear(64 + 64, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
            )
        
        mine = Mine(
            args    = self.args,
            T       = T,
            loss    = 'mine_biased',        #mine_biased, fdiv
            method  = 'concat')

        return mine

    def set_optimizers(self):
        # when task = DRAnet or task = DRA_MI
        self.optims['E'] = optim.Adam(self.nets['E'].parameters(), lr=self.args.lr_dra,
                                    betas=(self.args.beta1, 0.999),
                                    weight_decay=self.args.weight_decay_dra)
        self.optims['D'] = dict()

        for dset in self.args.datasets:
            self.optims['D'][dset] = optim.Adam(self.nets['D'][dset].parameters(), lr=self.args.lr_dra,
                                                betas=(self.args.beta1, 0.999),
                                                weight_decay=self.args.weight_decay_dra)

        self.optims['G'] = optim.Adam(self.nets['G'].parameters(), lr=self.args.lr_dra,
                                    betas=(self.args.beta1, 0.999),
                                    weight_decay=self.args.weight_decay_dra)

        self.optims['S'] = optim.Adam(self.nets['S'].parameters(), lr=self.args.lr_dra,
                                    betas=(self.args.beta1, 0.999),
                                    weight_decay=self.args.weight_decay_dra)
        
        self.optims['T'] = dict()
        for convert in self.test_converts:
            self.optims['T'][convert] = optim.SGD(self.nets['T'][convert].parameters(), lr=self.args.lr_clf, momentum=0.9,
                                                weight_decay=self.args.weight_decay_task)
            
    def set_zero_grad(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].zero_grad()
            elif net == 'T':
                for convert in self.test_converts:
                    self.nets[net][convert].zero_grad()
            else:
                self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].train()
            elif net == 'T':
                for convert in self.test_converts:
                    self.nets[net][convert].train()
            else:
                self.nets[net].train()

    def set_eval(self):
        for convert in self.test_converts:
            self.nets['T'][convert].eval()

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.args.datasets:
            try:
                batch_data[dset] = next(batch_data_iter[dset])
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = batch_data_iter[dset].next()
        return batch_data

    def train_dis(self, imgs):  
        self.set_zero_grad()
        features, converted_imgs, D_outputs_fake, D_outputs_real = dict(), dict(), dict(), dict()

        # Real
        for dset in self.args.datasets:
            input = imgs[dset].to(device)
            features[dset] = self.nets['E'](input)
            D_outputs_real[dset] = self.nets['D'][dset](input)

        contents, styles = self.nets['S'](features, self.training_converts)
        
        if self.args.task == 'MI_net':
            return contents, styles
                
        # CADT
        if self.args.CADT:
            for convert in self.training_converts:
                source, target = convert.split('2')
                _, styles[target] = cadt(contents[source], contents[target], styles[target])

        # Fake
        for convert in self.training_converts:
            source, target = convert.split('2')
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])

            D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])

                
        errD = self.loss_fns.dis(D_outputs_real, D_outputs_fake)
        errD.backward()
        for optimizer in self.optims['D'].values():
            optimizer.step()
        self.losses['D'] = errD.data.item()

    def train_task(self, imgs, labels):  # Train Task Networks (T)
        self.set_zero_grad()
        features = dict()
        converted_imgs = dict()
        pred = dict()
        converts = self.training_converts
        with torch.no_grad():
            for dset in self.args.datasets:
                features[dset] = self.nets['E'](imgs[dset])
            contents, styles = self.nets['S'](features, converts)
            for convert in converts:
                source, target = convert.split('2')
                if self.args.CADT:
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])

        for convert in self.test_converts:
            pred[convert] = self.nets['T'][convert](converted_imgs[convert])
            source, target = convert.split('2')
            pred[source] = self.nets['T'][convert](imgs[source])

        errT = self.loss_fns.task(pred, labels)
        errT.backward()
        for optimizer in self.optims['T'].values():
            optimizer.step()
        self.losses['T'] = errT.data.item()

    def train_esg(self, imgs):  # Train Encoder(E), Separator(S), Generator(G)
        self.set_zero_grad()
        features, converted_imgs, recon_imgs, D_outputs_fake = dict(), dict(), dict(), dict()
        features_converted = dict()
        perceptual, style_gram = dict(), dict()
        perceptual_converted, style_gram_converted = dict(), dict()
        con_sim = dict()
        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            recon_imgs[dset] = self.nets['G'](features[dset], 0)
            perceptual[dset] = self.nets['P'](imgs[dset])
            style_gram[dset] = [gram(fmap) for fmap in perceptual[dset][:-1]]
        contents, styles = self.nets['S'](features, self.training_converts)

        for convert in self.training_converts:
            source, target = convert.split('2')
            if self.args.CADT:
                con_sim[convert], styles[target] = cadt(contents[source], contents[target], styles[target])
                style_gram[target] = cadt_gram(style_gram[target], con_sim[convert])
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
            
            D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
            
            features_converted[convert] = self.nets['E'](converted_imgs[convert])
            perceptual_converted[convert] = self.nets['P'](converted_imgs[convert])
            style_gram_converted[convert] = [gram(fmap) for fmap in perceptual_converted[convert][:-1]]
        
        contents_converted, styles_converted = self.nets['S'](features_converted)

        Content_loss = self.loss_fns.content_perceptual(perceptual, perceptual_converted)
        Style_loss = self.loss_fns.style_perceptual(style_gram, style_gram_converted)
        Consistency_loss = self.loss_fns.consistency(contents, styles, contents_converted, styles_converted, self.training_converts)
        G_loss = self.loss_fns.gen(D_outputs_fake)
        Recon_loss = self.loss_fns.recon(imgs, recon_imgs)

        errESG = G_loss + Content_loss + Style_loss + Consistency_loss + Recon_loss

        errESG.backward()
        for net in ['E', 'S', 'G']:
            self.optims[net].step()

        self.losses['G'] = G_loss.data.item()
        self.losses['Recon'] = Recon_loss.data.item()
        self.losses['Consis'] = Consistency_loss.data.item()
        self.losses['Content'] = Content_loss.data.item()
        self.losses['Style'] = Style_loss.data.item()

    def tensor_board_log(self, imgs, labels):
        nrow = 8 
        features, converted_imgs, recon_imgs = dict(), dict(), dict()
        converts = self.tensorboard_converts
        with torch.no_grad():
            for dset in self.args.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                recon_imgs[dset] = self.nets['G'](features[dset], 0)
            contents, styles = self.nets['S'](features, self.training_converts)
            for convert in self.training_converts:
                source, target = convert.split('2')
                if self.args.CADT:
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
        
        # Input Images & Reconstructed Images
        for dset in self.args.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('1_Input_Images/%s' % dset, x, self.step)
            x = vutils.make_grid(recon_imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('2_Recon_Images/%s' % dset, x, self.step)

        # Converted Images
        for convert in converts:
            x = vutils.make_grid(converted_imgs[convert].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('3_Converted_Images/%s' % convert, x, self.step)

        # Losses
        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)

    def eval(self, cv):
        source, target = cv.split('2')
        self.set_eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(self.test_loader[target]):
                imgs, labels = imgs.cuda(), labels.cuda()
                pred = self.nets['T'][cv](imgs)
                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                progress_bar(batch_idx, len(self.test_loader[target]), 'Acc: %.3f%% (%d/%d)'
                                % (100. * correct / total, correct, total))
            # Save checkpoint.
            acc = 100. * correct / total
            self.logger.info('======================================================')
            self.logger.info('Step: %d | Acc: %.3f%% (%d/%d)'
                        % (self.step / len(self.test_loader[target]), acc, correct, total))
            self.logger.info('======================================================')
            self.writer.add_scalar('Accuracy/%s' % cv, acc, self.step)
            if acc > self.best_acc[cv]:
                self.best_acc[cv] = acc
                self.writer.add_scalar('Best_Accuracy/%s' % cv, acc, self.step)
                self.save_networks()

        self.set_train()

    def print_loss(self):
        best = ''
        for cv in self.test_converts:
            best = best + cv + ': %.2f' % self.best_acc[cv] + '|'
        
        losses = ''
        for key in self.losses:
            losses += ('%s: %.2f|'% (key, self.losses[key])) 
        self.logger.info(
            '[%d/%d] %s| %s %s'
            % (self.step, self.args.iter, losses, best, self.args.ex))
        
    def DRANet(self):
        self.set_default()
        self.set_networks()
        self.set_optimizers()
        self.set_train()
        self.logger.info(self.loss_fns.alpha)
        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])

        for i in range(self.args.iter):
            self.step += 1
            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.args.batch
            for dset in self.args.datasets:
                imgs[dset], labels[dset] = batch_data[dset]
                imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]

            # training
            self.train_dis(imgs)
            for t in range(2):
                self.train_esg(imgs)
            self.train_task(imgs, labels)
            # tensorboard
            if self.step % self.args.tensor_freq == 0:
                self.tensor_board_log(imgs, labels)
            # evaluation
            if self.step % self.args.eval_freq == 0:
                for cv in self.test_converts:
                    self.eval(cv)
            self.print_loss()

    def DRANet_test(self):
        self.set_default()
        self.set_networks()
        self.load_networks(self.args.load_step)
        for cv in self.test_converts:
            self.eval(cv)        

    def set_MI_train(self):
        for net in self.nets.keys():
            if net == 'E' or net == 'S':
                self.nets[net].eval()
            else:
                self.nets[net].train()

    def MI(self):
        self.set_default()
        self.set_networks()
        self.load_networks()
        self.set_MI_train()

        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])
        
        for i in range(self.args.iter):
            self.step += 1
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.args.batch
            for dset in self.args.datasets:
                imgs[dset], labels[dset] = batch_data[dset]
                imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]
            
            contents, styles = self.train_dis(imgs) # dict[dset]

            for dset in self.args.datasets:
                _, _, pred_mi, loss = self.nets['MI'].optimize_MI(contents[dset], styles[dset])

                if i % self.args.tensor_freq == 0:
                    self.writer.add_scalar(f'Train/MI_{dset}', pred_mi, i)
                    self.writer.add_scalar(f'Train/Loss{dset}', loss, i)
                
                if i % self.args.eval_freq == 0:
                    for cv in self.test_converts:
                        self.MI_eval(cv, i)

    def MI_eval(self, ex, step):
        source, target = ex.split('2')

        print("================== TEST ==================")                                

        max_mi = 0
        with torch.no_grad():
            self.nets['MI'].eval()
            mi = 0
            for idx, (imgs, _) in enumerate(self.test_loader[target]):
                imgs = imgs.cuda()
                features = dict()
                features[target] = self.nets['E'](imgs)
                contents, styles = self.nets['S'](features, self.training_converts)
                mi += self.MI_net.eval_MI(contents[target], styles[target])

            MI = mi / (idx+1)
            print('\nStep: %d | MI: %.3f\n' %(idx, MI))
            self.writer.add_scalar(f'Test/MI[{ex}]', round(MI.item(), 3), step)
            
            if MI > max_mi:
                self.save_networks()
                max_mi = MI
        
        self.MI_net.train()

    def DRANet_MI(self):
        self.set_default()
        self.set_networks()    
        self.set_optimizers()
        self.set_train()


        # 기존의 DRANet에 MI Loss term 추가
