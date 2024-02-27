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
from secretflow.ml.nn.sl.attacks.replay_attack_torch import ReplayAttack


class ReplayAttackCase(AttackCase):
    def _attack(self):
        self.app.prepare_data()
        target_nums = self.config.get('target_nums', 15)
        target_class, target_set, eval_set = self.app.replay_auxiliary_attack_configs(
            target_nums
        )
        replay_cb = ReplayAttack(
            self.alice if self.app.device_y == self.bob else self.bob,
            target_set,
            eval_set,
            batch_size=self.app.train_batch_size,
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )
        history = self.app.train(replay_cb)
        preds = self.app.predict(callbacks=replay_cb)
        attack_metrics = replay_cb.get_attack_metrics(
            reveal(preds), target_class, eval_set
        )
        logging.warning(
            f"RESULT: {type(self.app).__name__} replay attack metrics = {attack_metrics}"
        )
        return history, attack_metrics

    def attack_search_space(self):
        return {
            'target_nums': tune.search.grid_search([10, 50, 100]),
        }

    def metric_name(self):
        return 'acc'

    def metric_mode(self):
        return 'max'
