# Copyright 2024 Ant Group Co., Ltd.
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

from typing import Dict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    DatasetType,
    InputMode,
)
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.core.torch import TorchModel
from secretflow.ml.nn.sl.attacks.fsha_torch import FeatureSpaceHijackingAttack
from secretflow.ml.nn.utils import optim_wrapper


class Pilot(nn.Module):
    def __init__(self, input_dim=20, target_dim=64):
        super().__init__()
        self.net = nn.Linear(input_dim, target_dim)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, target_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, target_dim),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, target_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, target_dim),
        )

    def forward(self, x):
        return self.net(x)


def data_builder(device_f_dataset, batch_size, train_size):
    def prepare_data():
        target_loader = DataLoader(
            dataset=device_f_dataset, shuffle=False, batch_size=batch_size
        )
        aux_dataloader = DataLoader(
            dataset=device_f_dataset, shuffle=False, batch_size=batch_size
        )
        return target_loader, aux_dataloader

    return prepare_data


class FshaAttackCase(AttackBase):
    """
    Fsha attack needs:
    - A databuilder which returns victim dataloader, need impl in app.
    """

    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def __str__(self):
        return 'fsha'

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        pilot_optim_fn = optim_wrapper(optim.Adam, lr=0.0001)
        pilot_model = TorchModel(
            model_fn=Pilot,
            loss_fn=None,
            optim_fn=pilot_optim_fn,
            metrics=None,
            input_dim=app.get_device_f_fea_nums(),
            target_dim=app.hidden_size,
        )

        decoder_optim_fn = optim_wrapper(optim.Adam, lr=0.0001)
        decoder_model = TorchModel(
            model_fn=Decoder,
            loss_fn=None,
            optim_fn=decoder_optim_fn,
            metrics=None,
            latent_dim=app.hidden_size,
            target_dim=app.get_device_f_fea_nums(),
        )

        discriminator_optim_fn = optim_wrapper(optim.Adam, lr=0.0001)
        discriminator_model = TorchModel(
            model_fn=Discriminator,
            loss_fn=None,
            optim_fn=discriminator_optim_fn,
            metrics=None,
            input_dim=app.hidden_size,
            target_dim=1,
        )
        data_buil = data_builder(
            app.get_device_f_train_dataset(),
            app.train_batch_size,
            app.get_device_f_input_shape()[0],
        )

        return FeatureSpaceHijackingAttack(
            attack_party=app.device_y,
            victim_party=app.device_f,
            victim_model_dict=2,
            base_model_list=[self.alice, self.bob],
            pilot_model_wrapper=pilot_model,
            decoder_model_wrapper=decoder_model,
            discriminator_model_wrapper=discriminator_model,
            reconstruct_loss_builder=torch.nn.MSELoss,
            data_builder=data_buil,
            victim_fea_dim=(
                app.alice_fea_nums if app.device_f == app.alice else app.bob_fea_nums
            ),
            attacker_fea_dim=(
                app.alice_fea_nums if app.device_y == app.alice else app.bob_fea_nums
            ),
            gradient_penalty_weight=500,
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )

    def attack_type(self) -> AttackType:
        return AttackType.FEATURE_INFERENCE

    def check_app_valid(self, app: ApplicationBase) -> bool:
        # image not support.
        return app.base_input_mode() in [InputMode.SINGLE] and app.dataset_type() in [
            DatasetType.TABLE,
            DatasetType.RECOMMENDATION,
        ]

    def tune_metrics(self) -> Dict[str, str]:
        return {'val_acc_0': 'max'}
