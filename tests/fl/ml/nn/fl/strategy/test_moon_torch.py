# Copyright xuxiaoyang, ywenrou123@163.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, Precision

from secretflow import reveal
from secretflow.security import SecureAggregator
from secretflow_fl.ml.nn import FLModel
from secretflow_fl.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from secretflow_fl.utils.simulation.datasets_fl import load_mnist


class ConvNet(BaseModule):
    """Small ConvNet for MNIST."""

    def __init__(self, cosine_similarity_fn, out_dim=256, temperature=0.5, mu=1):
        super(ConvNet, self).__init__()

        self.cosine_similarity_fn = cosine_similarity_fn
        self.out_dim = out_dim
        self.temperature = temperature
        self.mu = mu

        # Define the head using nn.Sequential
        self.head = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Projection MLP
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, self.out_dim)
        # Last layer
        self.l3 = nn.Linear(self.out_dim, 10)

    def forward(self, x, return_all=False):
        x = self.head(x)  # Pass through the head
        # Projection MLP
        h = x.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        y = self.l3(x)
        if return_all:
            return h, x, y
        else:
            return y


class TestMOON:

    def test_moon_local_step(self, sf_simulation_setup_devices):
        devices = sf_simulation_setup_devices
        # Initialize the MOON strategy with a ConvNet model
        loss_fn = nn.CrossEntropyLoss  # Loss function for classification
        optim_fn = optim_wrapper(optim.Adam, lr=1e-2)  # Optimizer with learning rate
        cosine_similarity_fn = nn.CosineSimilarity(dim=-1)

        # Define the main model to be used
        model_def = TorchModel(
            model_fn=ConvNet,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
            ],
            cosine_similarity_fn=cosine_similarity_fn,
        )

        # Set up the MOON server actor
        device_list = [devices.alice, devices.bob]

        # Initialize the aggregator for the federated learning process
        aggregator = SecureAggregator(devices.carol, device_list)

        # Create the federated learning model
        fl_model = FLModel(
            server=devices.carol,
            device_list=device_list,
            model=model_def,
            strategy="moon",
            backend="torch",
            aggregator=aggregator,
            model_buffer_size=1,
        )

        # Prepare the dataset
        (_, _), (data, label) = load_mnist(
            parts={devices.alice: 0.4, devices.bob: 0.6},
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )

        # Train the model with the prepared data
        history = fl_model.fit(
            data,
            label,
            validation_data=(data, label),
            epochs=1,
            batch_size=32,
            aggregate_freq=1,
        )

        # Make predictions using the trained model
        result = fl_model.predict(data, batch_size=32)
        assert (
            len(reveal(result[device_list[0]])) == 4000
        )  # Check the number of predictions

        global_metric, _ = fl_model.evaluate(
            data, label, batch_size=32, random_seed=1234
        )
        print(history, global_metric)

        # Assert that the final accuracy matches the recorded history
        assert (
            global_metric[0].result().numpy()
            == history["global_history"]['val_multiclassaccuracy'][-1]
        )
        assert global_metric[0].result().numpy() > 0.1
