# -*- coding: utf-8 -*-
"""
"""
import argparse
import os
import sys
from pathlib import Path
try:
    from bpdb import set_trace
except ImportError:
    from pdb import set_trace

from fire.trainer import Trainer
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models.simpleCNN import simpleCNN

def main():
    model = simpleCNN()
    loss_func = F.nll_loss
    args = Trainer.get_args()
    train_data = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

    test_data = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())
    args_dict = vars(args)
    trainer = Trainer(**args_dict)
    trainer.fit(model, train_data, test_data, loss_func)

if __name__ == '__main__':
    main()

