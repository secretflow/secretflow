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
"""
This file references code of paper : Similarity-based Label Inference Attack against Training and Inference of Split Learning(IEEE2024)

https://ieeexplore.ieee.org/document/10411061
"""

from typing import Dict

from benchmark_examples.autoattack.applications.base import ApplicationBase, InputMode
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from secretflow_fl.ml.nn.sl.attacks.sim_lia_torch import SimilarityLabelInferenceAttack


class SimilarityLiaAttackCase(AttackBase):
    """
    Similartity-based LIA method.
    """

    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def __str__(self):
        return 'sim_lia'

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        # although the original paper use different known_data but
        availabel_data_type = ["feature", "grad"]
        availabel_attack_method = ["k-means", "distance"]
        availabel_distance_metric = ["euclidean", "cosine"]
        all_availabel_options = {
            "k-means": ["feature", "grad"],
            "distance": {
                "cosine": ["feature", "grad"],
                "euclidean": ["feature", "grad"],
            },
        }

        return SimilarityLabelInferenceAttack(
            attack_party=app.device_f,
            label_party=app.device_y,
            data_type=self.config.get("data_type", "grad"),
            attack_method=self.config.get("attack_method", "distance"),
            known_num=1,
            distance_metric="cosine",
            exec_device="cpu",
        )

    def attack_type(self) -> AttackType:
        return AttackType.LABLE_INFERENSE

    def tune_metrics(self) -> Dict[str, str]:
        return {'attack_acc': 'max'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return app.base_input_mode() in [InputMode.SINGLE]

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.3
        update_memory = lambda x: x * 1.02
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_memory)
            .apply_sim_resources(app.device_f.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_f.party, 'memory', update_memory)
        )
