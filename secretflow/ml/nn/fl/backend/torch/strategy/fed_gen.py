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
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from secretflow import PYU, DeviceObject, proxy
from secretflow.device import PYUObject
from secretflow.ml.nn.core.torch import BuilderType
from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy
from secretflow.security import SecureAggregator


class FedGen(BaseTorchModel):
    """
    FedGen: This methodology is designed to address data heterogeneity and enhance model generalization
    within the Federated Learning (FL) framework. It leverages Data-Free Knowledge Distillation (DFKD)
    technology to learn a lightweight generator model on the server side, which captures and aggregates
    model knowledge from diverse clients without access to actual training data.
    """

    def __init__(
        self,
        builder_base: BuilderType,
        random_seed: int = None,
        skip_bn: bool = False,
    ):
        super().__init__(builder_base, random_seed=random_seed, skip_bn=skip_bn)
        self.label_counts_dict = {}
        self.cur_epochs = 0

    def _exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def train_step(
        self,
        weights: dict,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Performs a local training step on the client's data, updating the model's parameters.

        Args:
            weights (dict): Dictionary containing the global model weights and generator weights from the server.
            cur_steps (int): The current step in the training process.
            train_steps (int): The total number of local training steps to perform.
            kwargs: Additional strategy-specific parameters, including the generator and data refresh flag.

        Returns:
            Tuple: A tuple containing the updated model parameters and a dictionary with the number of samples
            and label counts.
        """
        assert self.model is not None, "Model cannot be none, please define the model."
        assert (
            kwargs.get('generator', None) is not None
        ), "Generator cannot be none, please define the Generator."
        generator = kwargs.get('generator')
        self.model.train()

        # Optionally refresh the data iterator for a new batch
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
            self.label_counts_dict = {}
            self.cur_epochs += 1
        if (
            weights is not None
            and "generator_params" in weights
            and "model_params" in weights
        ):
            generator_params = weights["generator_params"]
            model_params = weights["model_params"]
            if generator_params is not None:
                generator.load_state_dict(generator_params)
            if model_params is not None:
                self.set_weights(model_params)

        num_sample = 0
        dp_strategy = kwargs.get('dp_strategy', None)
        logs = {}
        loss: torch.Tensor = None
        generator.eval()
        for step in range(train_steps):
            # Fetch the next batch of data
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
                self.label_counts_dict[label.item()] = (
                    self.label_counts_dict.get(label.item(), 0) + 1
                )

            # Forward pass through the model
            client_predict_logit = self.model(x)

            # Compute the loss using the model's training step method
            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)

            # Convert model outputs to predicted labels
            y_input = y_labels.clone().detach()
            generative_alpha = self._exp_lr_scheduler(
                self.cur_epochs, decay=0.98, init_lr=generator.generative_alpha
            )
            generative_beta = self._exp_lr_scheduler(
                self.cur_epochs, decay=0.98, init_lr=generator.generative_beta
            )
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

            # Sample labels and generate data for further training
            sampled_labels = np.random.choice(
                self.model.num_classes, generator.batch_size
            )
            input_labels_tensor = torch.tensor(sampled_labels)
            generated_result = generator(input_labels_tensor)
            generated_predict_logit = self.model(
                generated_result['output'], start_layer_idx=-1
            )

            # Compute the teacher loss, encouraging the model to predict the original labels from generated data
            teacher_loss = generative_alpha * torch.mean(
                self.model.loss(generated_predict_logit, input_labels_tensor)
            )
            # this is to further balance oversampled down-sampled synthetic data
            gen_ratio = generator.batch_size / num_sample
            # Combine losses: base loss, KL loss, and teacher loss
            loss += gen_ratio * teacher_loss + generative_kl_loss

            # Perform backward pass if using automatic optimization
            if self.model.automatic_optimization:
                self.model.backward_step(loss)

        loss_value = loss.item()
        logs['train-loss'] = loss_value

        # Log the results and update local metrics
        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        model_weights = self.get_weights(return_numpy=True)

        # Apply differential privacy (DP) strategy if defined
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                model_weights = dp_strategy.model_gdp(model_weights)

        return model_weights, {
            "num_sample": num_sample,
            "label_counts_dict": self.label_counts_dict,
        }

    def apply_weights(self, weights, **kwargs):
        """Updates the local model with the given weights from the server.

        Args:
            weights (dict): Dictionary containing the model weights.
            kwargs: Additional parameters if needed for specific update logic.

        """
        if weights is not None:
            if isinstance(weights, dict):
                weights = weights["model_params"]
            self.set_weights(weights)


class FedGenGeneratorModel(nn.Module):
    def __init__(
        self,
        hidden_dimension,
        latent_dimension,
        noise_dim,
        num_classes,
        loss_fn,
        optim_fn,
        diversity_loss,
        epochs=10,
        batch_size=32,
        generative_alpha=10,
        generative_beta=10,
        ensemble_alpha=1,
        ensemble_beta=0,
        ensemble_eta=1,
        embedding=False,
    ):
        """
        Initializes the FedGen generator model.

        Args:
            hidden_dimension (int): Size of the hidden layer.
            latent_dimension (int): Size of the latent representation layer.
            noise_dim (int): Dimension size of the noise vector input to the generator.
            num_classes (int): Number of classes (condition labels for the generator).
            loss_fn (callable): Loss function used for training the generator.
            optim_fn (callable): Optimizer function used for updating the generator's parameters.
            diversity_loss (callable): Loss function to encourage diversity among generated samples.
            epochs (int, optional): Number of training epochs (default: 10).
            batch_size (int, optional): Batch size for training (default: 32).
            generative_alpha (float, optional): Weight for the generative teacher loss term (default: 10).
            generative_beta (float, optional): Weight for the generative student loss term (default: 10).
            ensemble_alpha (float, optional): Weight for the teacher loss in the ensemble (default: 1).
            ensemble_beta (float, optional): Weight for the student loss in the ensemble (default: 0).
            ensemble_eta (float, optional): Weight for the diversity loss in the ensemble (default: 1).
            embedding (bool, optional): Whether to use an embedding layer to convert class labels to vectors (default: False).
        """

        super(FedGenGeneratorModel, self).__init__()
        self.hidden_dim = hidden_dimension
        self.latent_dim = latent_dimension
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.generative_alpha = generative_alpha
        self.generative_beta = generative_beta
        self.ensemble_alpha = ensemble_alpha
        self.ensemble_beta = ensemble_beta
        self.ensemble_eta = ensemble_eta
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

    def forward(self, labels, verbose=True):
        """
        Forward pass: generates latent representations or original samples conditioned on class labels.

        :param labels: Class labels used for conditional generation.
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


class StatefulFedGenAggregator(SecureAggregator):
    """
    StatefulFedGenAggregator: Extends SecureAggregator to handle aggregation of model parameters
    and training of the generator model in a federated learning context.
    """

    def __init__(
        self, device, participants: List[PYU], server_actor, fxp_bits: int = 18
    ):
        super().__init__(device, participants, fxp_bits)
        self.server_actor = server_actor

    def average(self, data: List[PYUObject], axis=None, weights=None):
        """
        Overrides the average method to perform parameter aggregation and generator model training.

        Args:
            data (List[PYUObject]): A list of participant model parameters.
            axis: The axis along which the average operation is performed.
            weights (optional): Weights to use during aggregation, which can influence the importance of each participant.

        Returns:
            avg_model_params: The aggregated model parameters if no generator training is needed.
            If generator training is involved, returns a dictionary with updated generator and model parameters.
        """

        def _get_label_counts(client_result):
            """Extracts the label count dictionary from the client's result."""
            return client_result["label_counts_dict"]

        def _get_num_samples(client_result):
            """Extracts the number of samples from the client's result."""
            return client_result["num_sample"]

        # Default aggregation without using weights
        _num_simple = None
        avg_model_params = super().average(data, axis, _num_simple)

        # If weights are provided, perform weighted aggregation and generator training
        if weights is not None and isinstance(weights, (list, tuple, np.ndarray)):
            synthetic_data_result = self.server_actor.generate_synthetic_data()
            _worker_label_counts = []
            _user_results = []
            _num_simple = []

            # Ensure the weights list length matches the number of participants
            assert len(weights) == len(
                data
            ), f'Length of the weights does not match the data: {len(weights)} vs {len(data)}.'

            for i, w in enumerate(weights):
                if isinstance(w, DeviceObject):
                    # Ensure each weight is associated with the correct device
                    assert (
                        w.device == data[i].device
                    ), 'Device of weight does not match the corresponding data device.'

                    # Extract label counts and user results
                    _worker_label_counts.append(
                        self._device(_get_label_counts)(w.to(self._device))
                    )
                    _user_results.append(
                        self.server_actor.get_penultimate_layer_output(
                            data[i].to(self._device), synthetic_data_result
                        )
                    )
                    _num_simple.append(w.device(_get_num_samples)(w))

            # Train the generator model
            self.server_actor.train_generator(
                _user_results,
                _worker_label_counts,
                synthetic_data_result,
                avg_model_params,
            )

            # Return updated generator and model parameters
            return self._device(lambda x: x)(
                {
                    "generator_params": self.server_actor.get_generator_weights(),
                    "model_params": avg_model_params,
                }
            )

        # If no weights are provided, return the average model parameters
        return avg_model_params


@proxy(PYUObject)
class FedGenActor(object):
    """
    FedGenActor: This class handles the generation of synthetic data and training of
    the generator model in a federated learning context. It interacts with the generator
    to create synthetic data and trains the generator based on user results and worker label counts.
    """

    def __init__(self, generator):
        self.generator = generator

    def generate_synthetic_data(self):
        """
        Generates synthetic data using the generator.

        Returns:
            dict: A dictionary containing the synthetic labels and the generated result.
        """
        # Generate random labels
        sampled_labels = np.random.choice(
            self.generator.num_classes, self.generator.batch_size
        )
        synthetic_data_label = torch.LongTensor(sampled_labels)

        # Use the generator to generate data
        generated_result = self.generator(synthetic_data_label)

        # Return a dictionary instead of a tuple
        return {
            "synthetic_data_label": synthetic_data_label,
            "generated_result": generated_result,
        }

    def train_generator(
        self, user_results, worker_label_counts, synthetic_data_result, avg_model_params
    ):
        """
        Trains the generator model using teacher-student learning and diversity loss.

        Args:
            user_results (list): The outputs from different users/workers used to compute teacher loss.
            worker_label_counts (list): A list of dictionaries, each representing the label distribution from a worker.
            synthetic_data_result (dict): A dictionary returned by the generate_synthetic_data function,
                                          containing synthetic labels and generated results.
            avg_model_params (list): A list containing the parameters of the average model.

        Returns:
            None: The generator's parameters are updated in place.
        """

        label_weights = []
        num_workers = len(worker_label_counts)

        # Compute the weight of each label across all workers
        for label in range(self.generator.num_classes):
            # Get the count of this label from each worker
            weights = [
                worker_counts.get(label, 0) for worker_counts in worker_label_counts
            ]

            # Sum the counts, add a small epsilon to avoid division by zero
            label_sum = np.sum(weights) + 1e-6  # Small tolerance to avoid zero division

            # Compute the weight for this label across all workers
            label_weights.append(np.array(weights) / label_sum)

        # Convert label weights to a numpy array
        label_weights = np.array(label_weights).reshape(
            (self.generator.num_classes, num_workers)
        )

        # Begin training the generator
        self.generator.train()
        for i in range(self.generator.epochs):
            self.generator.optimizer.zero_grad()
            generated_result = synthetic_data_result["generated_result"]
            synthetic_data_label = synthetic_data_result["synthetic_data_label"]

            # Calculate the diversity loss using noise and generated output
            diversity_loss = self.generator.diversity_loss(
                generated_result['eps'], generated_result['output']
            )

            # Initialize teacher loss
            teacher_loss = 0
            teacher_logit = 0
            for idx in range(len(user_results)):
                # Compute the weight of each label for each worker
                weight = torch.tensor(
                    label_weights[synthetic_data_label][:, idx].reshape(-1, 1),
                    dtype=torch.float32,
                )
                expand_weight = np.tile(weight, (1, self.generator.num_classes))

                user_result_given_gen = user_results[idx]

                # Calculate the weighted teacher loss for each worker
                teacher_loss_ = torch.mean(
                    self.generator.loss(user_result_given_gen, synthetic_data_label)
                    * weight
                )
                teacher_loss += teacher_loss_
                teacher_logit += user_result_given_gen * torch.tensor(
                    expand_weight, dtype=torch.float32
                )

            # Calculate student loss using KL divergence
            student_output = self.get_penultimate_layer_output(
                avg_model_params, synthetic_data_result
            )
            student_loss = F.kl_div(
                F.log_softmax(student_output, dim=1), F.softmax(teacher_logit, dim=1)
            )

            # Combine teacher and student losses with diversity loss
            if self.generator.ensemble_beta > 0:
                loss = (
                    self.generator.ensemble_alpha * teacher_loss
                    - self.generator.ensemble_beta * student_loss
                    + self.generator.ensemble_eta * diversity_loss
                )
            else:
                loss = (
                    self.generator.ensemble_alpha * teacher_loss
                    + self.generator.ensemble_eta * diversity_loss
                )

            loss.backward()
            self.generator.optimizer.step()

    def get_generator_weights(self):
        """
        Retrieves the generator's parameters.

        Returns:
            dict: A dictionary containing the current state of the generator's parameters.
        """
        return self.generator.state_dict()

    def get_penultimate_layer_output(self, model_params, synthetic_data_result):
        """
        Computes the output of the final layer using given model parameters and synthetic data.

        Args:
            model_params (list): A list containing the parameters of the final layer (weights and biases).
            synthetic_data_result (dict): The synthetic data generated using the generator.

        Returns:
            torch.Tensor: The output after applying the final layer's weights and bias.
        """
        generated_result = synthetic_data_result["generated_result"]

        # Assume the last two items in model_params are weights and biases
        weights = model_params[-2]  # Second to last element is the weights
        bias = model_params[-1]  # Last element is the bias
        generated_output = generated_result["output"]

        # Ensure generated_output, weights, and bias are NumPy arrays
        generated_output_np = (
            generated_output.detach().numpy()
            if isinstance(generated_output, torch.Tensor)
            else generated_output
        )
        weights_np = (
            weights.detach().numpy() if isinstance(weights, torch.Tensor) else weights
        )
        bias_np = bias.detach().numpy() if isinstance(bias, torch.Tensor) else bias

        # Perform matrix multiplication
        output = np.dot(generated_output_np, weights_np.T) + bias_np

        # Convert the result back to a PyTorch Tensor
        output_tensor = torch.from_numpy(output)

        return output_tensor


@register_strategy(strategy_name='fed_gen', backend='torch')
class PYUFedGen(FedGen):
    pass
