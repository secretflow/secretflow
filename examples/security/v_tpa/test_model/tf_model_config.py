#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

sys.path.append("..")

from .tf.cifar10_model import get_passive_model as cifar_passive_model
from .tf.mnist_model import get_passive_model as mnist_passive_model
from .tf.nuswide_model import get_passive_model as nuswide_passive_model

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
