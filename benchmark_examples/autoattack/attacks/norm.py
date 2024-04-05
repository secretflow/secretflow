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

from benchmark_examples.autoattack.attacks.base import AttackCase
from secretflow import reveal
from secretflow.ml.nn.sl.attacks.norm_torch import NormAttack


class NormAttackCase(AttackCase):
    def _attack(self):
        self.app.prepare_data()
        label = reveal(self.app.get_train_label().partitions[self.app.device_y].data)
        norm_callback = NormAttack(self.app.device_f, label)
        history = self.app.train(norm_callback)
        logging.warning(
            f"RESULT: {type(self.app).__name__} norm attack metrics = {norm_callback.get_attack_metrics()}"
        )
        return history, norm_callback.get_attack_metrics()

    def attack_search_space(self):
        # norm attack does not have search space.
        return {}

    def metric_name(self):
        return 'auc'

    def metric_mode(self):
        return 'max'
