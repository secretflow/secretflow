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

from typing import Dict, Tuple

import numpy as np

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.attacks.replay_attack_torch import ReplayAttack


class ReplayAttackCase(AttackBase):
    """
    Replay attack needs:
    - target class such as 8
    - choose some (like 15) target set with target indexes.
    - same as train, chosse some eval poison set.
    """

    def __str__(self):
        return 'replay'

    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)
        self.target_class = None
        self.eval_set = None

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback | None:
        target_nums = self.config.get('target_nums', 15)
        if app.num_classes == 2:
            self.target_class = 1
            self.poison_class = 0
        else:
            self.target_class = 5
            self.poison_class = 1
        target_indexes = np.where(
            np.array(app.get_plain_train_label()) == self.target_class
        )[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)
        eval_indexes = np.where(
            np.array(app.get_plain_test_label()) == self.poison_class
        )[0]
        self.eval_set = np.random.choice(
            eval_indexes, min(100, len(eval_indexes) - 1), replace=False
        )
        return ReplayAttack(
            app.device_f,
            target_set,
            self.eval_set,
            batch_size=app.train_batch_size,
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )

    def attack_type(self) -> AttackType:
        return AttackType.BACKDOOR

    def attack_metrics_params(self) -> Tuple | None:
        assert self.target_class is not None and self.eval_set is not None
        return self.target_class, self.eval_set

    def tune_metrics(self) -> Dict[str, str]:
        return {'acc': 'max'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return True

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.16
        return cluster_resources_pack.apply_debug_resources(
            'gpu_mem', update_gpu
        ).apply_sim_resources(app.device_f.party, 'gpu_mem', update_gpu)
