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

from benchmark_examples.autoattack.applications.base import ApplicationBase, ModelType
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.sl.attacks.grad_lia_attack_torch import (
    GradientClusterLabelInferenceAttack,
)


class GradLiaAttackCase(AttackBase):
    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def __str__(self):
        return 'grad_lia'

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        num_classes = app.num_classes
        return GradientClusterLabelInferenceAttack(
            attack_party=app.device_f, label_party=app.device_y, num_classes=num_classes
        )

    def attack_type(self) -> AttackType:
        return AttackType.LABLE_INFERENSE

    def tune_metrics(self) -> Dict[str, str]:
        return {'attack_auc': 'max', 'attack_acc': 'max'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return app.model_type() != ModelType.DEEPFM

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.2

        # k-means use a lot of memory
        def update_mem(__):
            G = 1024 * 1024 * 1024
            match app.dataset_name():
                case "bank":
                    return 12 * G
                case "creditcard":
                    return 27 * G
                case "drive":
                    return 10 * G
                case "movielens":
                    return 30 * G
                case "criteo":
                    return 30 * G
                case "cifar10":
                    return 25 * G
                case "mnist":
                    return 32 * G
                case _:
                    return 25 * G

        cluster_resources_pack = (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_mem)
            .apply_sim_resources(app.device_y.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_y.party, 'memory', update_mem)
        )
        print(
            f"grad lia resource m = {cluster_resources_pack.resources['debug']['memory']}, updatemem = {update_mem(1)}"
        )
        return cluster_resources_pack
