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
from benchmark_examples.autoattack.applications.base import ApplicationBase, InputMode
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.defenses.base import DefenseBase
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks import Callback
from secretflow.ml.nn.sl.defenses.mid import MIDefense


class Mid(DefenseBase):
    def __str__(self):
        return 'mid'

    def build_defense_callback(
        self, app: ApplicationBase, attack: AttackBase | None = None
    ) -> Callback | None:
        base_params = {}
        fuse_params = {}
        if attack.attack_type() == AttackType.LABLE_INFERENSE:
            base_params[app.device_y] = {
                "input_dim": app.hidden_size,
                "output_dim": app.hidden_size,
                "mid_lambda": self.config.get('mid_lambda', 0.5),
            }
        elif attack.attack_type() == AttackType.FEATURE_INFERENCE:
            fuse_params[app.device_f] = {
                "input_dim": app.hidden_size,
                "output_dim": app.hidden_size,
                "mid_lambda": self.config.get('mid_lambda', 0.5),
            }
        return MIDefense(
            base_params=base_params,
            fuse_params=fuse_params,
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )

    def check_attack_valid(self, attack: AttackBase) -> bool:
        return True

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return app.base_input_mode() == InputMode.SINGLE

    def tune_metrics(self, app_metrics: Dict[str, str]) -> Dict[str, str]:
        return {}

    def update_resources_consumptions(
        self,
        cluster_resources_pack: ResourcesPack,
        app: ApplicationBase,
        attack: AttackBase | None,
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.1
        update_mem = lambda x: x * 1.08
        cluster_resources_pack = cluster_resources_pack.apply_debug_resources(
            'gpu_mem', update_gpu
        ).apply_debug_resources('memory', update_mem)
        if attack is not None:
            if attack.attack_type() == AttackType.LABLE_INFERENSE:
                cluster_resources_pack = cluster_resources_pack.apply_sim_resources(
                    app.device_y.party, 'gpu_mem', update_gpu
                ).apply_sim_resources(app.device_y.party, 'memory', update_mem)
            elif attack.attack_type() == AttackType.FEATURE_INFERENCE:
                cluster_resources_pack = cluster_resources_pack.apply_sim_resources(
                    app.device_f.party, 'gpu_mem', update_gpu
                ).apply_sim_resources(app.device_f.party, 'memory', update_mem)
        return cluster_resources_pack
