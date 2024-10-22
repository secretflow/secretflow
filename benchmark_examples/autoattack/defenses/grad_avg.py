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

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
)
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.defenses.base import DefenseBase
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.sl.defenses.gradient_average import GradientAverage


class GradientAverageCase(DefenseBase):

    def __str__(self):
        return "grad_avg"

    def build_defense_callback(
        self, app: ApplicationBase, attack: AttackBase | None = None
    ) -> Callback | None:
        return GradientAverage(
            backend='torch', exec_device='cuda' if global_config.is_use_gpu() else 'cpu'
        )

    def check_attack_valid(self, attack: AttackBase) -> bool:
        return attack.attack_type() == AttackType.LABLE_INFERENSE

    def tune_metrics(self, app_metrics: Dict[str, str]) -> Dict[str, str]:
        return {}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return app.classfication_type() == ClassficationType.BINARY

    def update_resources_consumptions(
        self,
        cluster_resources_pack: ResourcesPack,
        app: ApplicationBase,
        attack: AttackBase | None,
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.1
        update_mem = lambda x: x * 1.1
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_sim_resources(app.device_y.party, 'gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_mem)
            .apply_sim_resources(app.device_y.party, 'memory', update_mem)
        )
