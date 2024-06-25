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
from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.attacks.exploit import ExploitAttackCase
from benchmark_examples.autoattack.defenses.base import DefenseBase
from secretflow.ml.nn.callbacks import Callback
from secretflow.ml.nn.sl.defenses.confusional_autoencoder import CAEDefense


class CAE(DefenseBase):
    def __str__(self):
        return 'cae'

    def build_defense_callback(
        self, app: ApplicationBase, attack: AttackBase | None = None
    ) -> Callback | None:
        return CAEDefense(
            defense_party=app.device_y,
            num_classes=app.num_classes,
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
            autoencoder_epochs=20,
            train_sample_size=30000,
            test_sample_siz=10000,
            T=self.config.get('T', 0.025),
            hyper_lambda=self.config.get('hyper_lambda', 2.0),
            learning_rate=self.config.get('learning_rate', 5e-4),
        )

    def check_attack_valid(self, attack: AttackBase) -> bool:
        return attack.attack_type() in [AttackType.LABLE_INFERENSE] and not isinstance(
            attack, ExploitAttackCase
        )

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return True

    def tune_metrics(self) -> Dict[str, str]:
        return {}
