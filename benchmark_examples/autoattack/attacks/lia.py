# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow import tune
from secretflow.ml.nn.sl.attacks.lia_torch import LabelInferenceAttack
from secretflow.tune.tune_config import RunConfig


def weights_init_ones(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)


# for attacker
class BottomModelPlus(nn.Module):
    def __init__(
        self,
        bottom_model,
        size_bottom_out=10,
        num_classes=10,
        num_layer=1,
        activation_func_type='ReLU',
        use_bn=True,
    ):
        super(BottomModelPlus, self).__init__()
        self.bottom_model = bottom_model

        dict_activation_func_type = {'ReLU': F.relu, 'Sigmoid': F.sigmoid, 'None': None}
        self.activation_func = dict_activation_func_type[activation_func_type]
        self.num_layer = num_layer
        self.use_bn = use_bn

        self.fc_1 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_1 = nn.BatchNorm1d(size_bottom_out)
        self.fc_1.apply(weights_init_ones)

        self.fc_2 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_2 = nn.BatchNorm1d(size_bottom_out)
        self.fc_2.apply(weights_init_ones)

        self.fc_3 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_3 = nn.BatchNorm1d(size_bottom_out)
        self.fc_3.apply(weights_init_ones)

        self.fc_4 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_4 = nn.BatchNorm1d(size_bottom_out)
        self.fc_4.apply(weights_init_ones)

        self.fc_final = nn.Linear(size_bottom_out, num_classes, bias=True)
        self.bn_final = nn.BatchNorm1d(size_bottom_out)
        self.fc_final.apply(weights_init_ones)

    def forward(self, x):
        x = self.bottom_model(x)

        if self.num_layer >= 2:
            if self.use_bn:
                x = self.bn_1(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_1(x)

        if self.num_layer >= 3:
            if self.use_bn:
                x = self.bn_2(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_2(x)

        if self.num_layer >= 4:
            if self.use_bn:
                x = self.bn_3(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_3(x)

        if self.num_layer >= 5:
            if self.use_bn:
                x = self.bn_4(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_4(x)
        if self.use_bn:
            x = self.bn_final(x)
        if self.activation_func:
            x = self.activation_func(x)
        x = self.fc_final(x)

        return x


def lia(config: Optional[Dict], *, alice, bob, app: TrainBase):
    if config is None:
        config = {}

    model = app.lia_auxiliary_model(ema=False)
    ema_model = app.lia_auxiliary_model(ema=True)

    data_buil = app.lia_auxiliary_data_builder()
    # for precision unittest
    # data_buil = data_builder(batch_size=16, file_path=data_file_path)
    model_save_path = './lia_model'

    lia_cb = LabelInferenceAttack(
        alice if app.device_y == bob else bob,
        model,
        ema_model,
        app.num_classes,
        data_buil,
        attack_epochs=1,
        save_model_path=model_save_path,
        T=config.get('T', 0.8),
        alpha=config.get('alpha', 0.75),
        val_iteration=config.get('val_iteration', 1024),
        k=4 if app.num_classes == 10 else 2,
        lr=config.get('lr', 2e-3),
        ema_decay=config.get('ema_decay', 0.999),
        lambda_u=config.get('lambda_u', 50),
    )
    app.train(lia_cb)
    logging.warning(f"lia cb attack metrics = {lia_cb.get_attack_metrics()}")
    return lia_cb.get_attack_metrics()


def auto_lia(alice, bob, app: TrainBase):
    search_space = {
        'T': tune.search.uniform(0.7, 1.0),  # 别是负值 0.8附近可能好 0.7 - 1
        'alpha': tune.search.uniform(0.8, 0.9999),  # 0 - 1 之间，0.9附近可能好
        'val_iteration': 1024,  # 整数 1 - 无穷
        'lr': tune.search.loguniform(2e-5, 2e-1),
        'ema_decay': tune.search.uniform(0.9, 1.0),
        'lambda_u': tune.search.grid_search([40, 45, 50, 60]),
    }
    trainable = tune.with_parameters(lia, alice=alice, bob=bob, app=app)
    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(
            storage_path=global_config.get_autoattack_path(),
            name=f"{type(app).__name__}_lia",
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    result_config = results.get_best_result(metric="val_acc_0", mode="max").config
    logging.warning(f"the best acc = {result_config}")
