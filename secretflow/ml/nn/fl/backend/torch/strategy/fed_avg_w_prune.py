#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# Copyright 2023 Ant Group Co., Ltd. imogenxingren
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

import secretflow as sf
from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


def mask_model(model, mask, initial_state_dict):
    """
    :param model: current model
    :param mask: current mask
    :param initial_state_dict: model state dict before mask
    :return:
    """
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            # renew model's param data
            param.data = torch.from_numpy(
                mask[step] * initial_state_dict[name].cpu().numpy()
            ).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]


# Prune model by Percentile and update mask
def prune_by_percentile(model, mask, percent, resample=False, reinit=False, **kwargs):
    """
    :param model:  current model
    :param mask:  current mask
    :param percent:  prune rate speed in each spoch
    :param resample:
    :param reinit:
    :param kwargs:
    :return:
    """
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            percentile_value = np.percentile(abs(alive), percent)
            weight_dev = param.device
            # renew new mask
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
            # Apply mask and new weight
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    return mask


first1 = True
first2 = True
global is_update_mask

class FedAvgWPrune(BaseTorchModel):
    """
    FedAvgWPrune: A implementation of FedAvg with pruning, where the clients upload their trained model
    weights after weights pruning to the server for averaging and update their local models via the aggregated weights
    from the server in each federated round.
    """

    def train_step_with_prune(
        self,
        weights: np.ndarray,
        cur_steps: int,
        train_steps: int,
        prune_current_mask: np.ndarray,
        prune_current_rate: float,
        is_update_mask,
        **kwargs,
    ) -> Tuple[np.ndarray, int, np.ndarray, int]:
        """train step with pruning
        Args:
            weights: transmit weights of epoch t
            cur_steps: transmit step
            train_steps:
            prune_current_mask:current mask
            prune_current_rate:current prune rate
            kwargs: strategy-specific parameters
                prune_end_rate：
                prune_percent：
            :return: update weight, num_sample, update mask, update rate
        """
        prune_end_rate = kwargs.get("prune_end_rate")  # prune end rate
        prune_percent = kwargs.get("prune_percent", True)  # prune dp increase rate
        weights, num_sample = self.train_step(
            weights,
            cur_steps,
            train_steps,
            prune_current_mask,
            **kwargs,
        )
        # if need update mask and not end prune, prune local model, update local mask and renew parameter
        if is_update_mask:
            if prune_current_rate > prune_end_rate:
                prune_current_mask = prune_by_percentile(
                    self.model, prune_current_mask, prune_percent
                )
                prune_current_rate = prune_current_rate * (1 - prune_percent / 100)
        return weights, num_sample, prune_current_mask, prune_current_rate

    def train_step(
        self,
        weights: np.ndarray,
        cur_steps: int,
        train_steps: int,
        prune_current_mask: np.ndarray,  # mask
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params, then do local train

        Args:
            weights: global weight from params server
            cur_steps: current train step
            train_steps: local training steps
            prune_current_mask: np.ndarray,
        Returns:
            Parameters after local training
        """
        assert self.model is not None, "Model cannot be none, please give model define"
        self.model.train()
        # step1: global model download
        if weights is not None:
            self.model.update_weights(weights)
        num_sample = 0
        dp_strategy = kwargs.get("dp_strategy", None)
        local_gradients_sum = None
        logs = {}

        # step2: mask the model and get local parameter before current pruning   initial s ,ask
        mask_model(self.model, prune_current_mask, self.model.state_dict())

        # step3: normal train of current local model(t)
        for _ in range(train_steps):
            self.optimizer.zero_grad()
            iter_data = next(self.train_iter)
            global first1
            global first2
            if len(iter_data) == 2:
                x, y = iter_data
                # save image
                if first1 is True:
                    torch.save(x, "./dgl/sf_output/x.pt")
                    torch.save(y, "./dgl/sf_output/y.pt")
                    first1 = False
                s_w = None
            elif len(iter_data) == 3:
                x, y, s_w = iter_data
            x = x.float()
            num_sample += x.shape[0]
            if len(y.shape) == 1:
                y_t = y
            else:
                if y.shape[-1] == 1:
                    y_t = torch.squeeze(y, -1).long()
                else:
                    y_t = y.argmax(dim=-1)
            if self.use_gpu:
                x = x.to(self.exe_device)
                y_t = y_t.to(self.exe_device)
                if s_w is not None:
                    s_w = s_w.to(self.exe_device)
            y_pred = self.model(x)
            loss = self.loss(
                y_pred, y_t
            )  # do back propagation predict current model(t) and get loss
            loss.backward()
            self.optimizer.step()
            local_gradients = self.model.get_gradients()
            # save_gradient
            if local_gradients_sum is None:
                if first2 is True:
                    for i in range(len(local_gradients)):
                        np.save(
                            "./dgl/sf_output/gradients" + str(i) + ".npy",
                            local_gradients[i],
                        )
                    first2 = False
                local_gradients_sum = local_gradients
            else:
                local_gradients_sum += local_gradients

            for m in self.metrics:
                m.update(y_pred.cpu(), y_t.cpu())
        loss_value = loss.item()
        logs["train-loss"] = loss_value

        self.logs = self.transform_metrics(logs)
        self.epoch_logs = copy.deepcopy(self.logs)
        model_weights = self.model.get_weights(return_numpy=True)

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                model_weights = dp_strategy.model_gdp(model_weights)

        return model_weights, num_sample


@register_strategy(strategy_name="fed_avg_w_prune", backend="torch")
class PYUFedAvgWPrune(FedAvgWPrune):
    pass
