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

from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
)
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow import reveal
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.attacks.norm_torch import NormAttack


class NormAttackCase(AttackBase):

    def __str__(self):
        return 'norm'

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        label = reveal(app.get_plain_train_label())
        return NormAttack(app.device_f, label)

    def attack_type(self) -> AttackType:
        return AttackType.LABLE_INFERENSE

    def tune_metrics(self) -> Dict[str, str]:
        return {'auc': 'max'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        # TODO: support multiclass
        return app.classfication_type() in [ClassficationType.BINARY]

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        return cluster_resources_pack
