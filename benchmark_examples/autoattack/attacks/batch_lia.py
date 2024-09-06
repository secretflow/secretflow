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
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.attacks.batch_level_lia_torch import (
    BatchLevelLabelInferenceAttack,
)


class BatchLevelLiaAttackCase(AttackBase):
    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def __str__(self):
        return 'batch_lia'

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        return BatchLevelLabelInferenceAttack(
            attack_party=app.device_f,
            victim_party=app.device_y,
            victim_hidden_size=[app.hidden_size],
            dummy_fuse_model=app.create_fuse_model(),
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
            label=app.get_plain_train_label(),
            lr=self.config.get('lr', 0.001),
            label_size=[app.num_classes],
            epochs=10,
        )

    def attack_type(self) -> AttackType:
        return AttackType.LABLE_INFERENSE

    def tune_metrics(self) -> Dict[str, str]:
        return {'recovery_rate': 'max'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return app.classfication_type() == ClassficationType.MULTICLASS

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.3
        update_mem = lambda x: x * 1.2
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_mem)
            .apply_sim_resources(app.device_f.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_f.party, 'memory', update_mem)
        )
