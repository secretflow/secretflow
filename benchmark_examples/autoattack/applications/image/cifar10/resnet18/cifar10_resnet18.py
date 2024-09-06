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

from torch import nn, optim
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack.applications.base import ModelType
from benchmark_examples.autoattack.applications.image.cifar10.cifar10_base import (
    Cifar10ApplicationBase,
)
from benchmark_examples.autoattack.utils.resources import ResourceDict, ResourcesPack
from secretflow.ml.nn.applications.sl_resnet_torch import (
    BasicBlock,
    ResNetBase,
    ResNetFuse,
)
from secretflow.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper


class Cifar10Resnet18(Cifar10ApplicationBase):
    def __init__(self, alice, bob):
        super().__init__(
            alice,
            bob,
            train_batch_size=128,
            hidden_size=512,
            dnn_fuse_units_size=[512 * 2],
        )
        self.metrics = [
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(AUROC, task="multiclass", num_classes=10),
        ]

    def model_type(self) -> ModelType:
        return ModelType.RESNET18

    @staticmethod
    def _create_base_model():
        return TorchModel(
            model_fn=ResNetBase,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            block=BasicBlock,
            layers=[2, 2, 2, 2],
        )

    def create_base_model_alice(self):
        return self._create_base_model()

    def create_base_model_bob(self):
        return self._create_base_model()

    def create_fuse_model(self):
        return TorchModel(
            model_fn=ResNetFuse,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=self.metrics,
            dnn_units_size=self.dnn_fuse_units_size,
        )

    def resources_consumption(self) -> ResourcesPack:
        # 980MiB
        return (
            ResourcesPack()
            .with_debug_resources(
                ResourceDict(
                    gpu_mem=1 * 1024 * 1024 * 1024, CPU=1, memory=4 * 1024 * 1024 * 1024
                )
            )
            .with_sim_resources(
                self.device_y.party,
                ResourceDict(
                    gpu_mem=1 * 1024 * 1024 * 1024, CPU=1, memory=4 * 1024 * 1024 * 1024
                ),
            )
            .with_sim_resources(
                self.device_f.party,
                ResourceDict(
                    gpu_mem=0.9 * 1024 * 1024 * 1024,
                    CPU=1,
                    memory=4 * 1024 * 1024 * 1024,
                ),
            )
        )
