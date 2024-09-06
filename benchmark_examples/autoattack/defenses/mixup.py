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

from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    InputMode,
    ModelType,
)
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.defenses.base import DefenseBase
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.core.torch import loss_wrapper, module
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.sl.defenses.mixup import Mixuplayer, Mixuploss


class Mixup(DefenseBase):

    def __str__(self):
        return "mixup"

    def build_defense_callback(
        self, app: ApplicationBase, attack: AttackBase | None = None
    ) -> Callback | None:
        return MixupDefense(
            lam=self.config.get('lam', 0.6),
            perm_seed=self.config.get('perm_seed', 1234),
        )

    def check_attack_valid(self, attack: AttackBase) -> bool:
        return attack.attack_type() == AttackType.LABLE_INFERENSE

    def tune_metrics(self, app_metrics: Dict[str, str]) -> Dict[str, str]:
        return {}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        """only support dnn"""
        return (
            app.model_type()
            in [ModelType.DNN, ModelType.RESNET18, ModelType.VGG16, ModelType.CNN]
            and app.base_input_mode() == InputMode.SINGLE
        )

    def update_resources_consumptions(
        self,
        cluster_resources_pack: ResourcesPack,
        app: ApplicationBase,
        attack: AttackBase | None,
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.3
        update_mem = lambda x: x * 1.15
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_mem)
            .apply_sim_resources(app.device_y.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_y.party, 'memory', update_mem)
        )


class MixupDefense(Callback):
    def __init__(self, lam=0.5, perm_seed=42, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.perm_seed = perm_seed

    @staticmethod
    def inject_mixup(worker: SLBaseTorchModel, lam, perm_seed, process_loss):
        mixup_layer = Mixuplayer(lam, perm_seed)
        worker._callback_store['mixup'] = {'preprocess_layer': mixup_layer}
        if process_loss:
            worker.builder_fuse.loss_fn = loss_wrapper(
                Mixuploss, worker.builder_fuse.loss_fn, lam, perm_seed
            )
            worker.model_fuse = module.build(worker.builder_fuse, worker.exec_device)

    @staticmethod
    def do_preprocess(worker: SLBaseTorchModel):
        preprocess_layer = worker._callback_store['mixup']['preprocess_layer']
        worker._data_x = preprocess_layer(worker._data_x)

    def on_base_forward_begin(self):
        worker = self._workers[self.device_y]
        worker.apply(self.do_preprocess)

    def on_train_begin(self, logs=None):
        worker = self._workers[self.device_y]
        worker.apply(
            self.inject_mixup, self.lam, self.perm_seed, worker == self.device_y
        )
