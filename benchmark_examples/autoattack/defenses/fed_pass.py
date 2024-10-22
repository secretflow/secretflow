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
from benchmark_examples.autoattack.defenses.base import DefenseBase
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.core.torch import module
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel


class FedPassDefense(Callback):
    """A special callback implementation, temporaily put it here."""

    def __init__(self, use_passport: Dict[str, bool], **kwargs):
        super().__init__(**kwargs)
        self.use_passport = use_passport

    @staticmethod
    def inject_model(
        worker: SLBaseTorchModel,
        use_passport,
    ):
        if worker.builder_base is not None:
            worker.builder_base.kwargs['use_passport'] = use_passport
            worker.model_base = module.build(worker.builder_base, worker.exec_device)

        if worker.builder_fuse is not None:
            worker.builder_fuse.kwargs['use_passport'] = use_passport
            worker.model_fuse = module.build(worker.builder_fuse, worker.exec_device)

    def on_train_begin(self, logs=None):
        for device, worker in self._workers.items():
            worker.apply(self.inject_model, self.use_passport[device.party])


class FedPass(DefenseBase):
    def __str__(self):
        return 'fed_pass'

    def build_defense_callback(
        self, app: ApplicationBase, attack: AttackBase | None = None
    ) -> Callback | None:
        return FedPassDefense(
            use_passport=self.config.get('use_passport', {'alice': True, 'bob': True}),
        )

    def check_attack_valid(self, attack: AttackBase) -> bool:
        return (
            attack.attack_type() == AttackType.LABLE_INFERENSE
            or attack.attack_type() == AttackType.FEATURE_INFERENCE
        )

    def check_app_valid(self, app: ApplicationBase) -> bool:
        """only support dnn"""
        return app.model_type() in [
            ModelType.DNN,
            ModelType.RESNET18,
            ModelType.VGG16,
            ModelType.DEEPFM,
        ]

    def tune_metrics(self, app_metrics: Dict[str, str]) -> Dict[str, str]:
        return {}

    def update_resources_consumptions(
        self,
        cluster_resources_pack: ResourcesPack,
        app: ApplicationBase,
        attack: AttackBase | None,
    ) -> ResourcesPack:

        update_gpu = lambda x: x * 1.2
        update_mem = lambda x: x * 1.17
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_mem)
            .apply_sim_resources(app.device_y.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_f.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_y.party, 'memory', update_mem)
            .apply_sim_resources(app.device_f.party, 'memory', update_mem)
        )
