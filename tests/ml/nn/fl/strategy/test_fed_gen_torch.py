# !/usr/bin/env python3
# *_* coding: utf-8 *_*
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

import torch
import torch.optim as optim
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from secretflow.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from secretflow.ml.nn.fl.backend.torch.strategy.fed_gen import (
    FedGen,
    FedGenGeneratorModel,
)


class ConvNet(BaseModule):
    """Small ConvNet for MNIST."""

    def __init__(self, kl_div_loss, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)
        self.kl_div_loss = kl_div_loss
        self.num_classes = num_classes

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx == -1:
            x = self.fc(x)
            return x
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.fc_in_dim)
        x = self.fc(x)
        return x


class DiversityLoss(nn.Module):
    """
    Custom diversity loss function, designed to encourage diversity in model predictions.
    """

    def __init__(self, metric):
        """
        Initializes the DiversityLoss class.

        Parameters:
        metric (str): The metric for computing distances, can be 'l1', 'l2', or 'cosine'.
        """
        super(
            DiversityLoss, self
        ).__init__()  # Call the parent class's initialization method
        self.metric = metric  # Save the metric
        self.cosine = nn.CosineSimilarity(
            dim=2
        )  # Initialize the cosine similarity computation object, dim=2 indicates similarity is computed along the 2nd dimension

    def compute_distance(self, tensor1, tensor2):
        """
        Computes the distance between two tensors.

        Parameters:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.

        Returns:
        torch.Tensor: The distance between the two tensors.
        """
        if self.metric == 'l1':
            # If the metric is L1 norm, compute the mean of the absolute differences between tensor elements
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif self.metric == 'l2':
            # If the metric is L2 norm, compute the mean of the squared differences between tensor elements
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif self.metric == 'cosine':
            # If the metric is cosine similarity, use cosine similarity to compute the distance between the two tensors
            # Cosine similarity values range from -1 to 1, here we convert it to a distance by 1 - cosine_similarity
            return 1 - self.cosine(tensor1, tensor2)
        else:
            # If the metric is not one of the above three, raise a ValueError exception
            raise ValueError("Unsupported metric: {}".format(self.metric))


class TestFedGen:

    def test_fed_gen_local_step(self, sf_simulation_setup_devices):
        # Initialize Scaffold strategy with ConvNet model
        num_classes = 10
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
        kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        diversity_loss = DiversityLoss(metric='l1')

        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=CrossEntropyLoss,
            optim_fn=optim.Adam,
            kl_div_loss=kl_div_loss,
            num_classes=num_classes,
        )

        generator = FedGenGeneratorModel(
            hidden_dimension=256,
            latent_dimension=192,
            noise_dim=64,
            num_classes=num_classes,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            diversity_loss=diversity_loss,
        )

        # Initialize Scaffold strategy with ConvNet model
        fed_gen_worker = FedGen(builder_base=builder)

        # Prepare dataset
        x_test = torch.rand(128, 1, 28, 28)  # Randomly generated data
        y_test = torch.randint(0, 10, (128,))

        test_loader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=32, shuffle=True
        )
        fed_gen_worker.train_set = iter(test_loader)
        fed_gen_worker.train_iter = iter(fed_gen_worker.train_set)

        # Perform a training step
        gradients = None
        gradients, num_sample = fed_gen_worker.train_step(
            gradients, cur_steps=0, train_steps=1, generator=generator
        )

        # Apply weights update
        fed_gen_worker.apply_weights(gradients)

        # Assert the sample number and length of gradients
        assert num_sample == 32  # Batch size
        assert len(gradients) == len(
            list(fed_gen_worker.model.parameters())
        )  # Number of model parameters

        # Perform another training step to test cumulative behavior
        _, num_sample = fed_gen_worker.train_step(
            gradients, cur_steps=1, train_steps=2, generator=generator
        )
        assert num_sample == 64  # Cumulative batch size over two steps
