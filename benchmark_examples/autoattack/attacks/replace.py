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

from typing import Dict, List, Tuple

import numpy as np

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.utils.data_utils import get_np_data_from_dataset
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.attacks.grad_replace_attack_torch import GradReplaceAttack


class ReplaceAttackCase(AttackBase):
    """
    Replace attack needs:
    - get target_indexs whoes label is target_class (such as 8)
    - choose some (like 15) target set with target indexes.
    - choose some train poison set (like 100)
    - choose the train poison set with device_f party (return numpy.)
    - same as train poison set, get eval poison set.
    """

    def __str__(self):
        return 'replace'

    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)
        self.target_class = None
        self.eval_poison_set = None

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        target_nums = self.config.get("target_nums", 15)
        poision_nums = 10 if global_config.is_simple_test() else 100
        if app.num_classes == 2:
            self.target_class = 1
        else:
            self.target_class = 5
        target_indexes = np.where(
            np.array(app.get_plain_train_label()) == self.target_class
        )[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)
        train_poison_set = np.random.choice(
            range(len(app.get_plain_train_label())), poision_nums, replace=False
        )
        sample_device_f_poison_dataset = app.get_device_f_train_dataset(
            indexes=target_indexes
        )
        train_poison_np = get_np_data_from_dataset(sample_device_f_poison_dataset)
        if isinstance(train_poison_np, List):
            # multi input scene
            train_poison_np = [np.stack(x) for x in train_poison_np]
        self.eval_poison_set = np.random.choice(
            range(len(app.get_plain_test_label())), poision_nums, replace=False
        )
        grad_callback = GradReplaceAttack(
            attack_party=app.device_f,
            target_idx=target_set,
            poison_idx=train_poison_set,
            poison_input=train_poison_np,
            gamma=self.config.get('gamma', 1),
            batch_size=app.train_batch_size,
            blurred=self.config.get("blurred", False),
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )
        return grad_callback

    def attack_type(self) -> AttackType:
        return AttackType.BACKDOOR

    def attack_metrics_params(self) -> Tuple | None:
        assert self.target_class is not None and self.eval_poison_set is not None
        return self.target_class, self.eval_poison_set

    def tune_metrics(self) -> Dict[str, str]:
        return {'acc': 'max'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return True

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.14
        return cluster_resources_pack.apply_debug_resources(
            'gpu_mem', update_gpu
        ).apply_sim_resources(app.device_f.party, 'gpu_mem', update_gpu)
