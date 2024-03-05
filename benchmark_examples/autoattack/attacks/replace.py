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

import logging

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.attacks.base import AttackCase
from secretflow import reveal, tune
from secretflow.ml.nn.sl.attacks.grad_replace_attack_torch import GradReplaceAttack


class ReplaceAttackCase(AttackCase):
    def _attack(self):
        self.app.prepare_data()
        (
            target_class,
            target_set,
            train_poinson_set,
            train_poison_np,
            eval_poinson_set,
        ) = self.app.replace_auxiliary_attack_configs()
        replace_callback = GradReplaceAttack(
            attack_party=self.app.device_f,
            target_idx=target_set,
            poison_idx=train_poinson_set,
            poison_input=train_poison_np,
            gamma=self.config.get('gamma', 1),
            batch_size=self.app.train_batch_size,
            blurred=self.config.get("blurred", False),
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )
        history = self.app.train(replace_callback)
        preds = self.app.predict(callbacks=replace_callback)
        attack_metrics = replace_callback.get_attack_metrics(
            reveal(preds), target_class, eval_poinson_set
        )
        logging.warning(
            f"RESULT: {type(self.app).__name__} replace attack metrics = {attack_metrics}"
        )
        return history, attack_metrics

    def attack_search_space(self):
        search_space = {
            # blurred does not support embedding layer, so shutdown,
            # 'blurred': tune.search.grid_search([True, False]),
            'gamma': tune.search.grid_search([10, 30]),  # 1 - 20
        }
        return search_space

    def metric_name(self):
        return 'acc'

    def metric_mode(self):
        return 'max'
