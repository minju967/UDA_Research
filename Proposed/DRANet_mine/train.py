from __future__ import print_function
from args import get_args
from trainer import Trainer

if __name__ == '__main__':
    opt = get_args()
    trainer = Trainer(opt)
    
    if opt.task == 'DRAnet':
        trainer.DRANet()
    elif opt.task == 'MI_net':
        trainer.MI()
    else:
        trainer.DRANet_MI()
    


