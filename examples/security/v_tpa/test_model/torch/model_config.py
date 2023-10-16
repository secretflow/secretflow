#!/usr/bin/env python
# coding=utf-8
import sys

sys.path.append("..")

from torch import nn, optim
from . import mnist_model
from . import nuswide_model
from . import cifar_model
from . import gtsrb_model

MODELS = {
    "mnist": {
        "lr": 0.01,
        "loss": nn.CrossEntropyLoss(),
        "epochs": 80,
        # 'epochs': 1,
        "batch_size": 128,
        "model": mnist_model,
        "party_num": 2,
    },
    "nus-wide": {
        "lr": 0.0001,
        "loss": nn.CrossEntropyLoss(),
        "epochs": 100,
        # 'epochs': 1,
        "batch_size": 128,
        "model": nuswide_model,
        "party_num": 2,
    },
    "cifar10": {
        "lr": 0.001,
        "loss": nn.CrossEntropyLoss(),
        "epochs": 20,
        # 'epochs': 1,
        "batch_size": 128,
        "model": cifar_model,
        "party_num": 2,
    },
    "gtsrb": {
        "lr": 0.001,
        "loss": nn.CrossEntropyLoss(),
        "epochs": 20,
        # 'epochs': 1,
        "batch_size": 128,
        "model": gtsrb_model,
        "party_num": 2,
    },
}
