import os
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from build_dataset import make_dataset
from build_trainer import Trainer

def get_args():
    project_path = Path(os.getcwd()).parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument('-E','--epoch', type=int, default=10)
    parser.add_argument('-i','--iter', type=int, default=1000)
    parser.add_argument('-D','--datasets', type=str, nargs='+', default='A', required=False, help='clf: A/CA/P/R (Art/Clipart/Product/Real World) ')

    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--imsize', type=int, default=224, help='the height of the input image')
    parser.add_argument('--project', type=str, default=project_path)

    parser.add_argument('--manualSeed', type=int, default=111)

    args = parser.parse_args()
    return args

def show_img(dataloader, cls_list):
    figure1 = plt.figure(1, figsize=(8, 8))
    cols, rows = 6, 6
    for i in range(1, cols*rows+1, 2):
        idx = torch.randint(len(dataloader), size=(1,)).item()
        img1, img2, label = dataloader[idx]
        figure1.add_subplot(cols, rows, i)
        plt.title(cls_list[label])
        plt.axis('off')
        plt.imshow(img1.squeeze().permute(1,2,0))
        figure1.add_subplot(cols, rows, i+1)
        plt.title(cls_list[label])
        plt.axis('off')
        plt.imshow(img2.squeeze().permute(1,2,0))
    
    plt.show()

def get_dataset(args, d):
    batch   = args.batch
    imsize  = args.imsize
    workers = args.workers
    
    domains = {'A':'Art', 'CA':'Clipart', 'P':'Product', 'R':'Real World'}
    domain  = domains[d]

    train_dataset = make_dataset('train', domain, args, imsize)
    test_dataset  = make_dataset('test', domain, args, imsize)
    class_list    = train_dataset.class_list

    # show_img(train_dataset, class_list)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch,
                                                   shuffle=True, num_workers=int(workers), pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch,
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
    # trainer.set_networks()
    trainer.train()
    # pass

if __name__ == '__main__':
    opt = get_args()
    main(opt)