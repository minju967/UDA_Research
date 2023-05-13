from __future__ import print_function
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.backends.cudnn
import numpy as np
import os
import sys 

from models.mine import Mine
from models import *
from utils import *
from dataset import get_dataset

def set_converts(datasets, task):
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
    training_converts = None
    return training_converts, test_converts, tensorboard_converts

class Trainer:
    def __init__(self, args):
        self.args = args
        self.training_converts, self.test_converts, self.tensorboard_converts = set_converts(args.datasets, args.task)
        
        self.imsize = (args.imsize, args.imsize)
        self.acc = dict()
        self.best_acc = dict()
        for cv in self.test_converts:
                self.best_acc[cv] = 0.

        # data loader
        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.args.datasets:
            train_loader, test_loader = get_dataset(dataset=dset, batch=self.args.batch,
                                                    imsize=self.imsize, workers=self.args.workers)
            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader

        self.nets, self.optims, self.losses = dict(), dict(), dict()

        self.writer = SummaryWriter('./tensorboard/%s' % args.save)
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
        # only E, S
        self.step = step
        for key in self.nets.keys():
            self.nets[key].load_state_dict(torch.load(self.checkpoint + '/%d/net%s.pth' % (step, key)), strict=False)
        
        for key in self.nets.keys():
            self.nets[key].eval()
        
    def set_networks(self):
        with torch.no_grad():
            self.nets['E'] = Encoder()
            self.nets['S'] = Separator(self.imsize, self.training_converts)

        for net in self.nets.keys():
            self.nets[net].cuda()
        
        self.MI_net = self.get_MINE()

    def set_train(self):
        for net in self.nets.keys():
            self.nets[net].train()

    def set_eval(self):
        for convert in self.test_converts:
            self.nets['E'][convert].eval()
            self.nets['S'][convert].eval()
        
    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.args.datasets:
            try:
                batch_data[dset] = next(batch_data_iter[dset])
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = next(batch_data_iter[dset])
                
        return batch_data

    def train_dis(self, imgs):  
        features = dict()

        # Real
        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])

        contents, styles = self.nets['S'](features, self.training_converts)

        return contents, styles
    
    def tensor_board_log(self, imgs, labels):
        nrow = 8 if self.args.task == 'clf' else 2
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
            # 3 datasets
            for convert in list(set(self.test_converts) - set(self.training_converts)):
                features_mid = dict()
                source, target = convert.split('2')
                mid = list(set(self.args.datasets) - {source, target})[0]
                convert1 = source + '2' + mid
                convert2 = mid + '2' + target
                features_mid[convert1] = self.nets['E'](converted_imgs[convert1])
                contents_mid, _ = self.nets['S'](features_mid, [convert2])
                converted_imgs[convert] = self.nets['G'](contents_mid[convert2], styles[target])

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

        # Segmentation GT, Prediction
        if self.args.task == 'seg':
            vn = 2
            self.set_eval()
            preds = dict()
            for dset in self.args.datasets:
                x = decode_labels(labels[dset].detach(), num_images=vn)
                x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
                self.writer.add_image('4_GT/%s' % dset, x, self.step)
                preds[dset] = self.nets['T']['G2C'](imgs[dset])
            preds['G2C'] = self.nets['T']['G2C'](converted_imgs['G2C'])

            for key in preds.keys():
                pred = preds[key].data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                x = decode_labels(pred, num_images=vn)
                x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
                self.writer.add_image('5_Prediction/%s' % key, x, self.step)
            self.set_train()

    def train_mine(self, content, style):
        source, target = self.args.datasets
        
        mi_ss = self.mine.optimize(content[source], style[source], iters=100, batch_size=self.args.batch)
        # mi_st = self.mine.optimize(content[source], domain[target])
        # mi_ts = self.mine.optimize(content[target], domain[source])
        # mi_tt = self.mine.optimize(content[target], domain[target])

        print(mi_ss)

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

    def train(self):
        self.set_default()
        self.set_networks()
        self.load_networks(self.args.load_step)

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
            contents, styles      = self.train_dis(imgs)
            source, target        = self.args.datasets
            t1, t2, pred_mi, loss = self.MI_net.optimize_MI(contents[target], styles[target], i, batch_size=self.args.batch)

            if i % self.args.tensor_freq == 0:
                self.writer.add_scalar('Train/MI', pred_mi, i)
                self.writer.add_scalar('Train/Loss', loss, i)
            
            if i % self.args.eval_freq == 0:
                for cv in self.test_converts:
                    self.eval(cv, i)


    def eval(self, cv, step):
        source, target = cv.split('2')

        max_mi = 0
        print('=========== TEST ===========')

        with torch.no_grad():
            self.MI_net.eval()
            mi = 0
            for batch_idx, (imgs, _) in enumerate(self.test_loader[target]):
                imgs = imgs.cuda()
                features = dict()
                features[target] = self.nets['E'](imgs)
                contents, styles = self.nets['S'](features, self.training_converts)
                mi += self.MI_net.eval_MI(contents[target], styles[target])
            
            MI = mi / (batch_idx+1)
            self.logger.info('Step: %d | MI: %.3f%%' %(self.step, MI))
            print('\nStep: %d | MI: %.3f\n' %(batch_idx, MI))
            self.writer.add_scalar(f'Test/MI[{cv}]', round(MI.item(), 3), step)
            
            if MI > max_mi:
                self.save_networks()
                max_mi = MI
        
        self.MI_net.train()

    def test(self):
        self.set_default()
        self.set_networks()
        self.load_networks(self.args.load_step)
        
        step = self.args.MI_step
        self.MI_net.load_state_dict(torch.load(self.checkpoint + '/%d/MI.pth' % (step)))
        self.MI_net.eval()
