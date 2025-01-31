# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.optim as optim
from secretflow.ml.nn.fl.backend.torch.strategy.fed_dyn import FedDYN
from secretflow_fl.ml.nn.utils import BaseModule
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision, Recall


class MNIST_Model(BaseModule):
    def __init__(self):
        super(MNIST_Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
        )
        self.head = nn.Linear(100, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


class TestFedDYN:
    def test_fed_dyn_local_step(self, sf_simulation_setup_devices):
        class ConvNetBuilder:
            def __init__(self):
                self.metrics = [
                    lambda: Accuracy(task="multiclass", num_classes=10, average="macro")
                ]

            def model_fn(self):
                return MNIST_Model()

            def loss_fn(self):
                return CrossEntropyLoss()

            def optim_fn(self, parameters):
                return optim.Adam(parameters)

        # Initialize ConvNetBuilder
        conv_net_builder = ConvNetBuilder()

        # Manually initialize FedDYN strategy
        fed_dyn_worker = FedDYN()
        fed_dyn_worker.metrics = conv_net_builder.metrics
        fed_dyn_worker.model = conv_net_builder.model_fn()
        fed_dyn_worker.loss_fn = conv_net_builder.loss_fn()
        fed_dyn_worker.optimizer = conv_net_builder.optim_fn(fed_dyn_worker.model.parameters())

        # Prepare dataset
        x_test = torch.rand(128, 1, 28, 28)
        y_test = torch.randint(0, 10, (128,))
        test_loader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=32, shuffle=True
        )
        fed_dyn_worker.train_set = iter(test_loader)
        fed_dyn_worker.train_iter = iter(fed_dyn_worker.train_set)

        # Perform a training step
        gradients = None
        gradients, num_sample = fed_dyn_worker.train_step(
            gradients, cur_steps=0, train_steps=1
        )

        # Apply weights update
        fed_dyn_worker.apply_weights(gradients)

        # Assert the sample number and length of gradients
        assert num_sample == 32  # Batch size
        assert len(gradients) == len(list(fed_dyn_worker.model.parameters()))  # Number of model parameters

        # Perform another training step to test cumulative behavior
        _, num_sample = fed_dyn_worker.train_step(gradients, cur_steps=1, train_steps=2)
        assert num_sample == 64  # Cumulative batch size over two steps
