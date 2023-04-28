import os
import argparse
import torch
from pathlib import Path
from datetime import datetime

from build_dataset import make_dataset
from build_trainer import Trainer

def get_args():
    project_path = Path(os.getcwd()).parent

    parser = argparse.ArgumentParser()
    parser.add_argument('-I','--iter', type=int, default=1)
    parser.add_argument('-D','--datasets', type=str, nargs='+', default='A', required=False, help='clf: A/CA/P/R (Art/Clipart/Product/Real World) ')

    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--imsize', type=int, default=224, help='the height of the input image')
    parser.add_argument('--project', type=str, default=project_path)

    parser.add_argument('--manualSeed', type=int, default=111)

    args = parser.parse_args()
    return args

def get_dataset(args, d):
    batch   = args.batch
    imsize  = args.imsize
    workers = args.workers
    
    domains = {'A':'Art', 'CA':'Clipart', 'P':'Product', 'R':'Real World'}
    domain  = domains[d]

    train_dataset = make_dataset('train', domain, args, imsize)
    test_dataset  = make_dataset('test', domain, args, imsize)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch,
                                                   shuffle=True, num_workers=int(workers), pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch*4,
                                                   shuffle=False, num_workers=int(workers))
    return train_dataloader, test_dataloader

def main(args):

    # build dataset
    train_loader, test_loader = dict(), dict()
    for dset in args.datasets:
        train_, test_ = get_dataset(args, dset)
        train_loader[dset] = train_
        test_loader[dset]  = test_
    
    # build trainer
    trainer = Trainer(args, train_loader, test_loader)
    trainer.train()
    pass

if __name__ == '__main__':
    opt = get_args()
    main(opt)