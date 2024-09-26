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

from secretflow.ml.nn.core.torch import BaseModule, TorchModel
from secretflow.ml.nn.fl.backend.torch.strategy.fed_gen import FedGen


class ConvNet(BaseModule):
    """Small ConvNet for MNIST."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx == -1:
            x = self.fc(x)
            return x
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.fc_in_dim)
        x = self.fc(x)
        # should be logit
        return x


class GeneratorModel(nn.Module):
    def __init__(
        self, hidden_dimension, latent_dimension, n_class, noise_dim, embedding=False
    ):
        super(GeneratorModel, self).__init__()
        self.hidden_dim = hidden_dimension
        self.latent_dim = latent_dimension
        self.n_class = n_class
        self.noise_dim = noise_dim
        self.embedding = embedding
        input_dim = (
            self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        )
        self.build_network(input_dim)

    def build_network(self, input_dim):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        # Representation layer
        self.representation_layer = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        result = {}
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim))  # sampling from Gaussian
        if verbose:
            result['eps'] = eps
        if self.embedding:  # embedded dense vector
            y_input = self.embedding_layer(labels)
        else:  # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class)
            y_input.zero_()
            # labels = labels.view
            y_input.scatter_(1, labels.view(-1, 1), 1)
        z = torch.cat((eps, y_input), dim=1)
        # FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result


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

        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=CrossEntropyLoss,
            optim_fn=optim.Adam,
        )

        generator = GeneratorModel(
            hidden_dimension=256,
            latent_dimension=192,
            n_class=10,
            noise_dim=64,
            embedding=False,
        )
        # Creating a KL Divergence Loss Function
        kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        diversity_loss = DiversityLoss(metric='l1')
        generative_optimizer = torch.optim.Adam(
            params=generator.parameters(),
            lr=0.0003,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
            amsgrad=False,
        )
        generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=generative_optimizer, gamma=0.98
        )
        cross_entropy_loss = nn.CrossEntropyLoss()

        generator_config = {
            'generator_model': generator,
            'optimizer': generative_optimizer,
            'scheduler': generative_lr_scheduler,
            'loss_fn': cross_entropy_loss,
            'kl_div_loss': kl_div_loss,
            'diversity_loss': diversity_loss,
            'num_classes': 10,
        }

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
            gradients, cur_steps=0, train_steps=1, generator_config=generator_config
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
            gradients, cur_steps=1, train_steps=2, generator_config=generator_config
        )
        assert num_sample == 64  # Cumulative batch size over two steps
