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

from benchmark_examples.autoattack.applications.base import ApplicationBase, ModelType
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.attacks.grad_lia_attack_torch import (
    GradientClusterLabelInferenceAttack,
)


class GradLiaAttackCase(AttackBase):
    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def __str__(self):
        return 'grad_lia'

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        num_classes = app.num_classes
        return GradientClusterLabelInferenceAttack(
            attack_party=app.device_f, label_party=app.device_y, num_classes=num_classes
        )

    def attack_type(self) -> AttackType:
        return AttackType.LABLE_INFERENSE

    def tune_metrics(self) -> Dict[str, str]:
        return {'val_acc_0': 'max'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return app.model_type() != ModelType.DEEPFM

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        func = lambda x: x * 1.2
        return cluster_resources_pack.apply_debug_resources(
            'gpu_mem', func
        ).apply_sim_resources(app.device_y.party, 'gpu_mem', func)
