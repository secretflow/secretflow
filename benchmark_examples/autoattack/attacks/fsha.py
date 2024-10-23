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
    ModelType,
)
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.core.torch import TorchModel
from secretflow.ml.nn.sl.attacks.fsha_torch import FeatureSpaceHijackingAttack
from secretflow.ml.nn.utils import optim_wrapper
from secretflow.utils.errors import NotSupportedError


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


class DecoderTable(nn.Module):
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


# Resnet18  input_channels=512
# Resnet20  input_channels=10
class DecoderResnet(nn.Module):
    def __init__(self, input_channels=512, output_channels=3, is_cifar10=True):
        super(DecoderResnet, self).__init__()

        # Cifar10 + Resnet18: [BN, 512, 1, 1] => [BN, 3, 32, 16]
        # Cifar10 + Resnet20: [BN, 10, 1, 1] => [BN, 3, 32, 16]
        # Mnist + Resnet18: [BN, 512, 1, 1] => [BN, 3, 28, 14]
        # Mnist + Resnet20: [BN, 10, 1, 1] => [BN, 3, 28, 14]
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                256,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ConvTranspose2d(
                128,
                128,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=(1 if is_cifar10 else 0),
                bias=False,
            ),
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ConvTranspose2d(
                64,
                output_channels,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0),
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1], 1, 1)
        out = self.model(x)
        return out


class DecoderVGG16(nn.Module):
    def __init__(self, input_channels=512, output_channels=3, is_cifar10=True):
        super(DecoderVGG16, self).__init__()

        # cifar10: [BN, 512, 3, 3] => [BN, 3, 32, 16]
        # mnist: [BN, 512, 1, 1] => [BN, 3, 112, 56]
        layer_in = nn.ConvTranspose2d(
            input_channels,
            256,
            kernel_size=(1, 1),
            stride=2 if is_cifar10 else 3,
            padding=1,
            output_padding=1 if is_cifar10 else 2,
            bias=False,
        )
        layer_mid = [
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            )
        ] + [
            nn.ConvTranspose2d(
                128,
                128,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            )
        ] * (
            1 if is_cifar10 else 2
        )
        layer_out = nn.ConvTranspose2d(
            128,
            output_channels,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding=1,
            output_padding=(1, 0),
            bias=False,
        )
        layers = [layer_in] + layer_mid + [layer_out, nn.Tanh()]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size()[0], 512, 3, 3)
        out = self.model(x)
        return out


def get_decoder(app: ApplicationBase):
    decoder_optim_fn = optim_wrapper(optim.Adam, lr=0.0001)
    if app.dataset_type() != DatasetType.IMAGE:
        return TorchModel(
            model_fn=DecoderTable,
            loss_fn=None,
            optim_fn=decoder_optim_fn,
            metrics=None,
            latent_dim=app.hidden_size,
            target_dim=app.get_device_f_fea_nums(),
        )
    elif (
        app.model_type() == ModelType.RESNET18 or app.model_type() == ModelType.RESNET20
    ):
        assert app.dataset_name() == 'cifar10' or app.dataset_name() == 'mnist'
        return TorchModel(
            model_fn=DecoderResnet,
            loss_fn=None,
            optim_fn=decoder_optim_fn,
            metrics=None,
            input_channels=512 if app.model_type() == ModelType.RESNET18 else 10,
            output_channels=3 if app.dataset_name() == 'cifar10' else 1,
            is_cifar10=True if app.dataset_name() == 'cifar10' else False,
        )
    elif app.model_type() == ModelType.VGG16:
        assert app.dataset_name() == 'cifar10' or app.dataset_name() == 'mnist'
        return TorchModel(
            model_fn=DecoderVGG16,
            loss_fn=None,
            optim_fn=decoder_optim_fn,
            metrics=None,
            input_channels=512,
            output_channels=3,
            is_cifar10=True if app.dataset_name() == 'cifar10' else False,
        )
    else:
        raise NotSupportedError(
            f"Fsha attack not supported in dataset {app.dataset_name()} app {app.model_type()}! "
        )


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
        pilot_model = (
            app.create_base_model_alice()
            if app.device_f == self.alice
            else app.create_base_model_bob()
        )

        decoder_model = get_decoder(app)
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
            app.get_device_f_train_dataset(list_return=True),
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
        return app.base_input_mode() in [InputMode.SINGLE]

    def tune_metrics(self) -> Dict[str, str]:
        return {'mean_model_loss': 'min', 'mean_guess_loss': 'min'}

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.18
        update_mem = lambda x: x * 1.05
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_mem)
            .apply_sim_resources(app.device_y.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_y.party, 'memory', update_mem)
        )
