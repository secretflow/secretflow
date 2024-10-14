#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# Copyright xuxiaoyang, ywenrou123@163.com
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

import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from secretflow import PYUObject, proxy
from secretflow.ml.nn.core.torch import BaseModule
from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


class FedGen(BaseTorchModel):
    """
    FedGen: This methodology is designed to address data heterogeneity and enhance model generalization
    within the Federated Learning (FL) framework. It leverages Data-Free Knowledge Distillation (DFKD)
    technology to learn a lightweight generator model on the server side, which captures and aggregates
    model knowledge from diverse clients without access to actual training data.
    """

    def predict_with_generator_output(self, generated_result):
        """Predicts the output using the -1 layer of the model

        Args:
            x: Input data to the model.
        Returns:
            The output of the model's -1 layer.
        """
        self.model.eval()
        return self.model(generated_result['output'], start_layer_idx=-1)

    def get_cur_step_label_counts(self):
        """Returns the counts of each label in the current training set.

        The return value is a dictionary where keys are the labels and values are the counts.

        Returns:
            A dictionary mapping labels to their counts.
        """
        return self.step_label_counts

    def train_step(
        self,
        weights: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params, then do local train

        Args:
            weights: global weight from params server
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """

        assert self.model is not None, "Model cannot be none, please give model define"
        assert (
            kwargs.get('generator', None) is not None
        ), "Generator cannot be none, please give Generator define"
        generator = kwargs.get('generator')
        self.model.train()
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        if weights is not None:
            self.set_weights(weights)
        num_sample = 0
        dp_strategy = kwargs.get('dp_strategy', None)
        logs = {}
        loss: torch.Tensor = None
        # Reset label_counts at the beginning of each train step
        self.step_label_counts = {}
        generator.eval()
        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += len(y)
            # Determine if y is one-hot encoded or class indices
            if y.ndim > 1 and y.shape[1] > 1:
                # y is one-hot encoded, convert to class indices
                y_labels = torch.argmax(y, dim=1)
            else:
                # y is already class indices
                y_labels = y

            # Accumulate label counts for monitoring class distribution
            for label in y_labels:
                self.step_label_counts[label.item()] = (
                    self.step_label_counts.get(label.item(), 0) + 1
                )

            # Forward pass through the model
            client_predict_logit = self.model(x)

            # Compute the loss using the training step method
            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)

            # Annealing factors for generative losses, calculated for the current step
            generative_alpha = max(1e-4, 0.1 * (0.98**cur_steps))
            generative_beta = max(1e-4, 0.1 * (0.98**cur_steps))

            # Convert model outputs to predicted labels, using y_labels
            y_input = y_labels.clone().detach()

            # Generate data and compute model output based on the input labels
            generated_result = generator(y_input)
            generated_predict_logit = self.model(
                generated_result['output'], start_layer_idx=-1
            )
            generated_predict_softmax = (
                F.softmax(generated_predict_logit, dim=1).clone().detach()
            )

            # KL loss encourages the model to produce similar predictions for generated data as for real data
            generative_kl_loss = generative_beta * self.model.kl_div_loss(
                F.log_softmax(client_predict_logit, dim=1), generated_predict_softmax
            )

            # Sample labels and generate data
            sampled_labels = np.random.choice(
                self.model.num_classes, generator.batch_size
            )
            input_labels_tensor = torch.tensor(sampled_labels)
            generated_result = generator(input_labels_tensor)
            generated_predict_logit = self.model(
                generated_result['output'], start_layer_idx=-1
            )

            # Teacher loss guides the model to predict the original labels from generated representations
            teacher_loss = generative_alpha * torch.mean(
                self.model.loss(generated_predict_logit, input_labels_tensor)
            )
            # Combine losses
            loss += teacher_loss + generative_kl_loss

            if self.model.automatic_optimization:
                self.model.backward_step(loss)

        loss_value = loss.item()
        logs['train-loss'] = loss_value

        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        model_weights = self.get_weights(return_numpy=True)

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                model_weights = dp_strategy.model_gdp(model_weights)

        return model_weights, num_sample

    def apply_weights(self, weights, **kwargs):
        """Accept ps model params, then update local model

        Args:
            weights: global weight from params server
        """
        if weights is not None:
            self.set_weights(weights)


@proxy(PYUObject)
class FedGenActor(object):
    """
    FedGenActor: This class is responsible for handling the generation of samples and training
    the generator model in a federated learning setting. It interacts with the generator to
    generate synthetic data and trains the generator based on user results and worker label counts.
    """

    def generate_samples(self, generator):
        """
        Generates synthetic samples using the generator model.

        Args:
            generator (FedGenGeneratorModel): The generator model used to produce synthetic data.

        Returns:
            tuple:
                - y_input (torch.Tensor): The labels for the generated data.
                - generated_result (dict): The output from the generator, including noise and generated data.
        """
        # Randomly sample labels from the available classes
        sampled_labels = np.random.choice(generator.num_classes, generator.batch_size)
        y_input = torch.LongTensor(sampled_labels)

        # Generate synthetic data using the generator
        generated_result = generator(y_input)
        return y_input, generated_result

    def train_generator(
        self, user_results, worker_label_counts, y_input, generated_result, generator
    ):
        """
        Trains the generator model using teacher-student learning and diversity loss.

        Args:
            user_results (list): The results from different users/workers used for teacher loss calculation.
            worker_label_counts (list): A list of dictionaries, each representing the label distribution from a worker.
            y_input (torch.Tensor): The input labels used for data generation.
            generated_result (dict): The generated data and noise produced by the generator.
            generator (FedGenGeneratorModel): The generator model to be trained.

        Returns:
            None: The generator's parameters are updated in place.
        """
        label_weights = []
        num_workers = len(worker_label_counts)

        # Iterate over each label to compute its weight across all workers
        for label in range(generator.num_classes):
            # Get the count of this label from each worker
            weights = [
                worker_counts.get(label, 0) for worker_counts in worker_label_counts
            ]

            # Sum the counts, add a small epsilon to avoid division by zero
            label_sum = (
                np.sum(weights) + 1e-6
            )  # Tolerance adjusted based on dataset size

            # Compute the weight of this label across all workers
            label_weights.append(np.array(weights) / label_sum)

        # Convert label weights to a numpy array
        label_weights = np.array(label_weights).reshape(
            (generator.num_classes, num_workers)
        )

        # Begin training the generator
        generator.train()
        generator.optimizer.zero_grad()

        # Calculate the diversity loss using the noise and generated output
        diversity_loss = generator.diversity_loss(
            generated_result['eps'], generated_result['output']
        )

        # Initialize teacher loss
        teacher_loss = 0
        for idx in range(len(user_results)):
            # Compute the weight of each label for each worker
            weight = torch.tensor(
                label_weights[y_input][:, idx].reshape(-1, 1), dtype=torch.float32
            )
            user_result_given_gen = user_results[idx]

            # Calculate the weighted teacher loss for each worker
            teacher_loss_ = torch.mean(
                generator.loss(user_result_given_gen, y_input) * weight
            )
            teacher_loss += teacher_loss_

        # Combine teacher loss and diversity loss
        loss = teacher_loss + diversity_loss
        loss.backward()  # Perform backpropagation
        generator.optimizer.step()  # Update generator parameters


class FedGenGeneratorModel(BaseModule):
    def __init__(
        self,
        hidden_dimension,
        latent_dimension,
        noise_dim,
        num_classes,
        loss_fn,
        optim_fn,
        diversity_loss,
        epochs=50,
        batch_size=32,
        embedding=False,
    ):
        """
        Initializes the FedGen generator model.

        :param hidden_dimension: Dimension size of the hidden layer.
        :param latent_dimension: Dimension size of the latent representation layer.
        :param noise_dim: Dimension size of the noise vector input to the generator.
        :param num_classes: Number of classes (condition labels for the generator).
        :param loss_fn: Loss function for generator training.
        :param optim_fn: Optimizer function for the generator.
        :param diversity_loss: Loss function to encourage sample diversity.
        :param epoch: Number of training epochs (default: 50).
        :param batch_size: Batch size for training (default: 32).
        :param embedding: Whether to use embedding layer to convert class labels to vectors (default: False).
        """
        super(FedGenGeneratorModel, self).__init__()
        self.hidden_dim = hidden_dimension
        self.latent_dim = latent_dimension
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.embedding = embedding
        # Determine input dimension. If embedding is used, input is a concatenation of noise and embedded labels;
        # otherwise, input is a concatenation of noise and one-hot encoded labels.
        input_dim = (
            self.noise_dim * 2 if self.embedding else self.noise_dim + self.num_classes
        )
        self.build_network(input_dim)

        self.loss = loss_fn()
        self.optimizer = optim_fn(self.parameters())
        self.diversity_loss = diversity_loss
        self.epochs = epochs
        self.batch_size = batch_size

    def build_network(self, input_dim):
        """
        Builds the generator network, including embedding layer, fully connected layers, and latent representation layer.

        :param input_dim: Dimension size of the input layer.
        """
        if self.embedding:
            # Embedding layer: converts labels to dense embedding vectors
            self.embedding_layer = nn.Embedding(self.num_classes, self.noise_dim)

        # Fully connected layers: maps input to hidden dimension
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),  # Fully connected layer
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization layer
            nn.ReLU(),  # Activation function layer
        )

        # Representation layer: maps hidden layer output to latent representation
        self.representation_layer = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        """
        Forward pass: generates latent representations or original samples conditioned on class labels.

        :param labels: Class labels used for conditional generation.
        :param latent_layer_idx: Specifies which layer's representation to output:
            -1: Output latent representation of the final layer,
            -2: Output latent representation of the second-to-last layer,
            0: Output original samples.
        :param verbose: If True, returns the sampled Gaussian noise vector.
        :return: A dictionary containing output information.
        """
        result = {}
        batch_size = labels.shape[0]

        # Sample noise from a standard Gaussian distribution
        eps = torch.rand((batch_size, self.noise_dim))  # Gaussian noise
        if verbose:
            result['eps'] = eps  # Store noise in the result dictionary

        # If embedding is used, embed the labels, otherwise use one-hot encoding
        if self.embedding:
            y_input = self.embedding_layer(labels)  # Embedding mapping
        else:
            y_input = torch.FloatTensor(
                batch_size, self.num_classes
            )  # Create one-hot encoded tensor
            y_input.zero_()  # Initialize with zeros
            y_input.scatter_(1, labels.view(-1, 1), 1)  # One-hot encoding

        # Concatenate noise and label inputs
        z = torch.cat((eps, y_input), dim=1)

        # Forward pass through fully connected layers
        for layer in self.fc_layers:
            z = layer(z)

        # Get latent representation
        z = self.representation_layer(z)
        result['output'] = z  # Store result in the dictionary

        return result


@register_strategy(strategy_name='fed_gen', backend='torch')
class PYUFedGen(FedGen):
    pass
