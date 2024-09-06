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

from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import ApplicationBase, InputMode
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.global_config import is_simple_test
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.attacks.lia_torch import LabelInferenceAttack


def weights_init_ones(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)


# for attacker
class BottomModelPlus(nn.Module):
    def __init__(
        self,
        size_bottom_out=10,
        num_classes=10,
        num_layer=1,
        activation_func_type='ReLU',
        use_bn=True,
    ):
        super(BottomModelPlus, self).__init__()
        self.bottom_model = None

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
        # print(f"IN ATTACK model ,after bottom model ,the x shape = {x.shape}")
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


def data_builder(
    train_complete_dataset: Dataset,
    train_sample_labeled_dataset: Dataset,
    train_sample_unlabeled_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 16,
):
    def prepare_data():
        train_complete_loader = DataLoader(
            train_complete_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        train_labeled_loader = DataLoader(
            train_sample_labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        train_unlabeled_loader = DataLoader(
            train_sample_unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        return (
            train_labeled_loader,
            train_unlabeled_loader,
            test_loader,
            train_complete_loader,
        )

    return prepare_data


class LiaAttackCase(AttackBase):
    """
    Lia attack needs:
    - an aux model: with same BottomModelPlus
    - an aux ema model
    - an aux databuilder which return some samples.
    """

    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def __str__(self):
        return 'lia'

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:

        att_model = BottomModelPlus(
            size_bottom_out=app.hidden_size, num_classes=app.num_classes
        )
        ema_att_model = BottomModelPlus(
            size_bottom_out=app.hidden_size, num_classes=app.num_classes
        )
        for param in ema_att_model.parameters():
            param.detach_()
        # lia need device_f data with devicc_y label
        train_complete_dataset = app.get_device_f_train_dataset(enable_label=0)
        train_sample_labeled_dataset = app.get_device_f_train_dataset(
            sample_size=50, enable_label=0
        )
        train_sample_unlabeled_dataset = app.get_device_f_train_dataset(
            sample_size=50, enable_label=-1
        )
        test_dataset = app.get_device_f_test_dataset(enable_label=0)
        data_buil = data_builder(
            train_complete_dataset,
            train_sample_labeled_dataset,
            train_sample_unlabeled_dataset,
            test_dataset,
        )
        # for precision unittest
        return LabelInferenceAttack(
            app.device_f,
            att_model,
            ema_att_model,
            app.num_classes,
            data_buil,
            attack_epochs=1 if is_simple_test() else 5,
            save_model_path=None,
            T=self.config.get('T', 0.8),
            alpha=self.config.get('alpha', 0.75),
            val_iteration=self.config.get('val_iteration', 1024),
            k=4 if app.num_classes == 10 else 2,
            lr=self.config.get('lr', 2e-3),
            ema_decay=self.config.get('ema_decay', 0.999),
            lambda_u=self.config.get('lambda_u', 50),
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )

    def attack_type(self) -> AttackType:
        return AttackType.LABLE_INFERENSE

    def tune_metrics(self) -> Dict[str, str]:
        return {'val_acc_0': 'max'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return app.base_input_mode() in [InputMode.SINGLE]

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.3
        update_memory = lambda x: x * 1.02
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_memory)
            .apply_sim_resources(app.device_f.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_f.party, 'memory', update_memory)
        )
