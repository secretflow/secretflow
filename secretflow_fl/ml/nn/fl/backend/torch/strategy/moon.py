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

from secretflow.ml.nn.core.torch import BuilderType
from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


class MOON(BaseTorchModel):
    """
    MOON: MOON is a simple yet effective federated learning framework that
    leverages the similarity between model representations to correct the local
    training of participating clients, effectively employing contrastive learning
    at the model level.

    Given that local training is often subject to drift and the representations
    learned by the global model are superior to those of local models, the goal
    of MOON is to minimize the distance between the representations learned by
    the local models and those learned by the global model, while simultaneously
    increasing the distance between the current local model's representation
    and that of its previous version.
    """

    def __init__(
        self,
        builder_base: BuilderType,
        random_seed: int = None,
        skip_bn: bool = False,
        **kwargs,
    ):
        super().__init__(builder_base, random_seed=random_seed, skip_bn=skip_bn)

        self.model_buffer_size = kwargs.get("model_buffer_size", 1)
        self.prev_model_list = []

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
        self.model.train()
        refresh_data = kwargs.get("refresh_data", False)
        global_model = None
        if refresh_data:
            self._reset_data_iter()
        if weights is not None:
            # Create a copy of the global model
            global_model = copy.deepcopy(self.model)
            global_model.update_weights(weights)
            global_model.eval()

            # Disable gradient computation for the global model
            for param in global_model.parameters():
                param.requires_grad = False
            self.set_weights(weights)
        num_sample = 0
        dp_strategy = kwargs.get('dp_strategy', None)
        logs = {}
        loss: torch.Tensor = None
        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]
            _, pro1, y_pred = self.model(x, return_all=True)
            self.model.update_metrics(y_pred, y)
            loss = self.model.loss(y_pred, y)

            logits = None
            if global_model is not None:
                _, pro2, _ = global_model(x, return_all=True)
                posi = self.model.cosine_similarity_fn(pro1, pro2)
                logits = posi.reshape(-1, 1)
                for pre_model in self.prev_model_list:
                    _, pro3, _ = pre_model(x, return_all=True)
                    nega = self.model.cosine_similarity_fn(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= self.model.temperature

            if logits is not None:
                labels = torch.zeros(x.size(0)).long()
                loss_con = self.model.mu * self.model.loss(logits, labels)
                loss += loss_con

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

        # Save the local historical models
        if len(self.prev_model_list) >= self.model_buffer_size:
            # If prev_model_list reaches the buffer size, remove the oldest model weights
            self.prev_model_list.pop(0)

        # Create a copy of the model for evaluation only
        history_model = copy.deepcopy(self.model)
        history_model.eval()

        # Disable gradient computation for the evaluation model
        for param in history_model.parameters():
            param.requires_grad = False

        # Add the evaluation model to the previous model list
        self.prev_model_list.append(history_model)

        return model_weights, num_sample

    def apply_weights(self, weights, **kwargs):
        """Accept ps model params, then update local model

        Args:
            weights: global weight from params server
        """
        if weights is not None:
            self.set_weights(weights)


@register_strategy(strategy_name='moon', backend='torch')
class PYUFedMOON(MOON):
    pass
