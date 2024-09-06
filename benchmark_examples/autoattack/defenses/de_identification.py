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

from typing import Dict, List

import torch

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    DatasetType,
    InputMode,
    ModelType,
)
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.defenses.base import DefenseBase
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.sl.defenses.de_identification import Maskinglayer


class DeIdentificationDefense(Callback):
    """A special callback implementation, temporaily put it here."""

    def __init__(
        self,
        subset_num=3,
        input_dim_dict: Dict | None = None,
        exec_device: torch.device | str = 'cpu',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.subset_num = subset_num
        self.input_dim_dict = input_dim_dict
        self.exec_device = exec_device

    @staticmethod
    def inject_model(
        worker: SLBaseTorchModel,
        subset_num,
        input_dims: List | None = None,
        device: torch.device | str = 'cpu',
    ):
        if input_dims is None:
            input_dims = worker.builder_base.kwargs['input_dims']
        assert (
            len(input_dims) == 1
        ), f"De identification only support input dims len = 1, got {len(input_dims)}"
        masking_layer = Maskinglayer(
            input_dim=input_dims[0],
            subset_num=subset_num,
        ).to(device)
        worker._callback_store['de_identification'] = {
            'preprocess_layer': masking_layer
        }

    def on_train_begin(self, logs=None):
        for device, worker in self._workers.items():
            if device != self.device_y:
                input_dims = (
                    self.input_dim_dict[device]
                    if self.input_dim_dict is not None
                    else None
                )
                worker.apply(
                    self.inject_model, self.subset_num, input_dims, self.exec_device
                )

    @staticmethod
    def do_preprocess(worker: SLBaseTorchModel):
        preprocess_layer = worker._callback_store['de_identification'][
            'preprocess_layer'
        ]
        worker._data_x = preprocess_layer(worker._data_x)

    def on_base_forward_begin(self):
        for device, worker in self._workers.items():
            if device != self.device_y:
                worker.apply(self.do_preprocess)


class DeIdentification(DefenseBase):
    def __str__(self):
        return 'de_identification'

    def build_defense_callback(
        self, app: ApplicationBase, attack: AttackBase | None = None
    ) -> Callback | None:
        input_dim_dict = None
        if app.model_type() in [ModelType.RESNET18, ModelType.VGG16, ModelType.OTHER]:
            input_dim_dict = {
                self.alice: [app.alice_fea_nums],
                self.bob: [app.bob_fea_nums],
            }
        return DeIdentificationDefense(
            subset_num=self.config.get('subset_num', 3),
            input_dim_dict=input_dim_dict,
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )

    def check_attack_valid(self, attack: AttackBase) -> bool:
        return attack.attack_type() == AttackType.FEATURE_INFERENCE

    def check_app_valid(self, app: ApplicationBase) -> bool:
        """only support dnn, only support 2D
        TODO: image dataset can also apply de_identification by using DCT.
        """
        return (
            app.model_type()
            in [
                ModelType.DNN,
                ModelType.RESNET18,
                ModelType.VGG16,
                ModelType.OTHER,
            ]
            and app.dataset_type() != DatasetType.IMAGE
            and app.base_input_mode() == InputMode.SINGLE
        )

    def tune_metrics(self, app_metrics: Dict[str, str]) -> Dict[str, str]:
        return {}

    def update_resources_consumptions(
        self,
        cluster_resources_pack: ResourcesPack,
        app: ApplicationBase,
        attack: AttackBase | None,
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 1.3
        update_mem = lambda x: x * 1.13
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_sim_resources(app.device_f.party, 'gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_mem)
            .apply_sim_resources(app.device_f.party, 'memory', update_mem)
        )
