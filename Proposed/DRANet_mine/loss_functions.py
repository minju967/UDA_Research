import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_weights(task, dsets):
    '''
        You must set these hyperparameters to apply our method to other datasets.
        These hyperparameters may not be the optimal value for your machine.
    '''

    alpha = dict()
    alpha['style'], alpha['dis'], alpha['gen'] = dict(), dict(), dict()

    alpha['recon'], alpha['consis'], alpha['content'] = 5, 1, 1

    # MNIST <-> MNIST-M
    if 'M' in dsets and 'MM' in dsets and 'U' not in dsets:
        alpha['style']['M2MM'], alpha['style']['MM2M'] = 5e4, 1e4
        alpha['dis']['M'], alpha['dis']['MM'] = 0.5, 0.5
        alpha['gen']['M'], alpha['gen']['MM'] = 0.5, 1.0

    # MNIST <-> USPS
    elif 'M' in dsets and 'U' in dsets and 'MM' not in dsets:
        alpha['style']['M2U'], alpha['style']['U2M'] = 5e3, 5e3
        alpha['dis']['M'], alpha['dis']['U'] = 0.5, 0.5
        alpha['gen']['M'], alpha['gen']['U'] = 0.5, 0.5

    # MNIST <-> MNIST-M
    if 'A' in dsets and 'CA' in dsets:
        alpha['style']['A2CA'], alpha['style']['CA2A'] = 1e3, 1e3
        alpha['dis']['A'], alpha['dis']['CA'] = 0.5, 0.5
        alpha['gen']['A'], alpha['gen']['CA'] = 0.5, 0.5


    return alpha


class Loss_Functions:
    def __init__(self, args):
        self.args = args
        self.alpha = loss_weights(args.task, args.datasets)

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += F.l1_loss(imgs[dset], recon_imgs[dset])
        return self.alpha['recon'] * recon_loss
        
    def dis(self, real, fake):
        dis_loss = 0
        
        for dset in real.keys():
            dis_loss += self.alpha['dis'][dset] * F.binary_cross_entropy(real[dset], torch.ones_like(real[dset]))
        
        for cv in fake.keys():
            source, target = cv.split('2')
            dis_loss += self.alpha['dis'][target] * F.binary_cross_entropy(fake[cv], torch.zeros_like(fake[cv]))

        return dis_loss

    def gen(self, fake):
        gen_loss = 0
        for cv in fake.keys():
            source, target = cv.split('2')
            gen_loss += self.alpha['gen'][target] * F.binary_cross_entropy(fake[cv], torch.ones_like(fake[cv]))

        return gen_loss

    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            content_perceptual_loss += F.mse_loss(perceptual[source][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            source, target = cv.split('2')
            for gr in range(len(style_gram[target])):
                style_percptual_loss += self.alpha['style'][cv] * F.mse_loss(style_gram[target][gr], style_gram_converted[cv][gr])
        return style_percptual_loss

    def consistency(self, contents, styles, contents_converted, styles_converted, converts):
        consistency_loss = 0
        for cv in converts:
            source, target = cv.split('2')
            consistency_loss += F.l1_loss(contents[cv], contents_converted[cv])
            consistency_loss += F.l1_loss(styles[target], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss

    def task(self, pred, gt):
        task_loss = 0
        for key in pred.keys():
            if '2' in key:
                source, target = key.split('2')
            else:
                source = key
            task_loss += F.cross_entropy(pred[key], gt[source], ignore_index=-1)
        return task_loss