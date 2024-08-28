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

from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


class FedGen(BaseTorchModel):
    """
    FedGen: This methodology is designed to address data heterogeneity and enhance model generalization
    within the Federated Learning (FL) framework. It leverages Data-Free Knowledge Distillation (DFKD)
    technology to learn a lightweight generator model on the server side, which captures and aggregates
    model knowledge from diverse clients without access to actual training data.
    """

    def predict_with_generator_output(self, x):
        """Predicts the output using the -1 layer of the model

        Args:
            x: Input data to the model.
        Returns:
            The output of the model's -1 layer.
        """
        self.model.eval()
        return self.model(x, -1)

    def get_cur_step_label_counts(self):
        """Returns the counts of each label in the current training set.

        The return value is a dictionary where keys are the labels and values are the counts.

        Returns:
            A dictionary mapping labels to their counts.
        """
        return self.label_counts

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
            kwargs.get('generator_config', None) is not None
        ), "Generator Config cannot be none, please give Generator define"
        generator_config = kwargs.get('generator_config')
        # Check if all required keys are in the generator_config dictionary
        required_keys = ['generator_model', 'loss_fn', 'kl_div_loss', 'num_classes']
        for key in required_keys:
            if key not in generator_config:
                raise ValueError(
                    f"The '{key}' key is missing in the generator_config dictionary."
                )

        generative_num_classes = generator_config['num_classes']
        generative_batch_size = generator_config.get('batch_size', 32)
        generator = generator_config['generator_model']
        loss_fn = generator_config['loss_fn']
        kl_div_loss = generator_config['kl_div_loss']

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
        self.label_counts = {}
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
                self.label_counts[label.item()] = (
                    self.label_counts.get(label.item(), 0) + 1
                )

            # Forward pass through the model
            user_output_logp = self.model(x)

            # Compute the loss using the training step method
            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)

            # Annealing factors for generative losses, calculated for the current step
            generative_alpha = max(1e-4, 0.1 * (0.98**cur_steps))
            generative_beta = max(1e-4, 0.1 * (0.98**cur_steps))

            # Convert model outputs to predicted labels, using y_labels
            y_input = y_labels.clone().detach()

            # Generate data and compute model output based on the input labels
            gen_result = generator(y_input)
            logit_given_gen = self.model(gen_result['output'], start_layer_idx=-1)
            target_p = F.softmax(logit_given_gen, dim=1).clone().detach()

            # Latent loss encourages the model to produce similar predictions for generated data as for real data
            user_latent_loss = generative_beta * kl_div_loss(
                F.log_softmax(user_output_logp, dim=1), target_p
            )

            # Sample labels and generate data
            sampled_y = np.random.choice(generative_num_classes, generative_batch_size)
            y_input = torch.tensor(sampled_y)
            gen_result = generator(y_input)
            user_output_logp = self.model(gen_result['output'], start_layer_idx=-1)

            # Teacher loss guides the model to predict the original labels from generated representations
            teacher_loss = generative_alpha * torch.mean(
                loss_fn(user_output_logp, y_input)
            )

            # Combine losses
            loss += teacher_loss + user_latent_loss

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


@register_strategy(strategy_name='fed_gen', backend='torch')
class PYUFedGen(FedGen):
    pass
