#!/usr/bin/env python
# coding=utf-8
import sys

sys.path.append("..")

from .tf.mnist_model import get_passive_model as mnist_passive_model
from .tf.nuswide_model import get_passive_model as nuswide_passive_model
from .tf.cifar10_model import get_passive_model as cifar_passive_model

MODELS = {
    "cifar10": {
        "optimizer": "adam",
        "lr": 0.001,
        "loss": "categorical_crossentropy",
        "epochs": 20,
        "batch_size": 128,
        "model": cifar_passive_model,
        "metrics": ["accuracy"],
    },
    "mnist": {
        "optimizer": "sgd",
        "lr": 0.01,
        "loss": "categorical_crossentropy",
        "epochs": 100,
        "batch_size": 256,
        "model": mnist_passive_model,
        "metrics": ["accuracy"],
    },
    "nus-wide": {
        "optimizer": "sgd",
        "lr": 0.0001,
        "loss": "categorical_crossentropy",
        "epochs": 100,
        "batch_size": 128,
        "model": nuswide_passive_model,
        "metrics": ["accuracy"],
    },
}
