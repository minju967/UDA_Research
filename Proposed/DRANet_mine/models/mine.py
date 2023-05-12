import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.autograd import Variable

from torchvision import datasets
from torchvision.transforms import transforms


from models.layers import ConcatLayer, CustomSequential

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from utils.helpers import *

EPS = 1e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print("Device:", device)

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()    # second

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean

class Mine(nn.Module):
    def __init__(self, args, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.args   = args
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method

        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T).to(device=device)
            else:
                self.T = CustomSequential(ConcatLayer(), T).to(device=device)
        else:
            self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        # return -t + second_term
        return t - second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, iters, batch_size, opt=None):
        
        # opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        for iter in range(1, iters + 1):
            mu_mi = 0
            for i, (x, y) in enumerate(batch(X, Y, batch_size)):
                # opt.zero_grad()
                loss = self.forward(x, y)
                print(f'{iter}_{i}_loss: {loss:.3f}')
                # loss.backward()
                # opt.step()

                # mu_mi -= loss.item()
            if iter % (iters // 3) == 0:
                pass
                # print(f"It {iter} - MI: {mu_mi / batch_size}")

        final_mi = self.mi(X, Y)
        print(f"Final MI: {final_mi}")
        return final_mi

    def optimize_MI(self, X, Y, iter, batch_size):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        distance, sig_dis, loss = 0, 0, 0
        for i, (x, y) in enumerate(batch(X, Y, batch_size=batch_size)):
            opt.zero_grad()
            distance = self.forward(x, y)
            sig_dis  = torch.sigmoid(distance)
            loss     = -torch.log(sig_dis)
            loss.backward()
            opt.step()
        
        final_mi = self.mi(X, Y)
        print(f'[{iter}/{self.args.iter}] Distance(MI): {distance:.3f} || loss: {loss:.3f} || Final_MI:{final_mi:.3f}')
        return final_mi