#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

from torch import optim
from model.resnet_cifar import ResNetCIFAR10

def get_passive_model(input_shape, output_shape, option=10):
    if option == 10:
        return ResNetCIFAR10()
    else:
        raise 'Invalid model option!!!'

def get_optimizer(params, lr):
    return optim.Adam(params, lr=lr, weight_decay=1e-5)

