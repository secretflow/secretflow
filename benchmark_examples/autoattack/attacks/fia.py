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
import torch.optim as optim

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import TrainBase
from benchmark_examples.autoattack.applications.image.drive.dnn.drive_dnn import (
    SLBaseNet,
)
from secretflow import tune
from secretflow.ml.nn.fl.utils import optim_wrapper
from secretflow.ml.nn.sl.attacks.fia_torch import FeatureInferenceAttack
from secretflow.ml.nn.utils import TorchModel
from secretflow.tune.tune_config import RunConfig


class Generator(nn.Module):
    def __init__(self, latent_dim=48, target_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(600, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, target_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def fia(config=Optional[Dict], *, alice, bob, app: TrainBase):
    if config is None:
        config = {}
    bob_mean = app.mean_attr[28:]
    victim_model_save_path = './sl_model_victim'
    victim_model_dict = {
        bob: [SLBaseNet, victim_model_save_path],
    }
    optim_fn = optim_wrapper(optim.Adam, lr=config.get('optim_lr', 0.0001))
    generator_model = TorchModel(
        model_fn=Generator,
        loss_fn=None,
        optim_fn=optim_fn,
        metrics=None,
    )
    data_buil = app.fia_auxiliary_data_builder()
    generator_save_path = './generator'
    fia_callback = FeatureInferenceAttack(
        victim_model_path=victim_model_save_path,
        attack_party=alice,
        victim_party=bob,
        victim_model_dict=victim_model_dict,
        base_model_list=[alice, bob],
        generator_model_wrapper=generator_model,
        data_builder=data_buil,
        victim_fea_dim=20,
        attacker_fea_dim=28,
        enable_mean=config.get('enale_mean', False),
        enable_var=True,
        mean_lambda=1.2,
        var_lambda=0.25,
        attack_epochs=config.get('attack_epochs', 60),
        victim_mean_feature=bob_mean,
        save_attacker_path=generator_save_path,
    )
    app.train(fia_callback)
    return fia_callback.get_attack_metrics()


def auto_fia(alice, bob, app):
    search_space = {
        # 'enable_mean': tune.search.grid_search([True,False]),
        'attack_epochs': tune.search.choice([20, 60, 120]),
        'optim_lr': tune.search.loguniform(1e-5, 1e-1),
    }
    trainable = tune.with_parameters(fia, alice=alice, bob=bob, app=app)
    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(
            storage_path=global_config.get_autoattack_path(),
            name=f"{type(app).__name__}_fia",
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    result_config = results.get_best_result(metric="mean_model_loss", mode="min").config
    logging.warning(f"the best result config = {result_config}")
