#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# Copyright 2022 Ant Group Co., Ltd.
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

from secretflow.ml.nn.fl.backend.torch.fl_base import FedPACTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy

from copy import deepcopy
import torch
from typing import Dict, Tuple
import copy
import numpy as np
import logging


class FedPAC(FedPACTorchModel):
    def train_step(
        self,
        step,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[
        torch.device,
        Dict[str, torch.Tensor],
    ]:
        """Accept ps model params, then do local train

        Args:
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """
        # Set mode to train model
        assert self.model is not None, "Model cannot be none, please give model define"
        model = self.local_model
        model.train()
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        logs = {}
        iter_loss = []
        model.zero_grad()
        global_protos = self.global_protos
        self.last_model = deepcopy(model)
        # get local prototypes before training, dict:={label: list of sample features}
        local_protos1 = self.get_local_protos(step, train_steps)
        # Set optimizer for the local updates, default sgd
        lr = kwargs.get("lr", 0.01)

        epoch_classifier = 1
        optimizer = torch.optim.SGD(
            model.parameters(), lr, momentum=0.5, weight_decay=0.0005
        )
        for name, param in model.named_parameters():
            if name in self.w_local_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        lr_g = 0.1
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_g,
            momentum=0.5,
            weight_decay=0.0005,
        )
        for ep in range(epoch_classifier):
            # local training for 1 epoch
            for step in range(train_steps):
                images, labels, _ = self.next_batch()
                images, labels = images.to(self.exe_device), labels.to(self.exe_device)
                model.zero_grad()
                protos, output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()

        for name, param in model.named_parameters():
            if name in self.w_local_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr,
            momentum=0.5,
            weight_decay=0.0005,
        )
        for step in range(train_steps):
            model.zero_grad()
            protos, output = model(images)
            model.update_metrics(output, labels)
            loss0 = self.criterion(output, labels)
            loss1 = 0
            protos_new = protos.clone().detach()
            for i in range(len(labels)):
                yi = labels[i].item()
                if yi in global_protos:
                    protos_new[i] = global_protos[yi].detach()
                elif yi in local_protos1:
                    protos_new[i] = local_protos1[yi].detach()
                else:
                    logging.info(
                        f"Key {yi} not found in both global_protos and local_protos1"
                    )
            loss1 = self.mse_loss(protos_new, protos)
            loss = loss0 + self.lam * loss1
            loss.backward()
            optimizer.step()
            iter_loss.append(loss.item())

        train_loss = sum(iter_loss) / len(iter_loss)
        logging.info(f'local train loss: {train_loss}')
        logs['train-loss'] = train_loss
        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        return (
            self.exe_device,
            model.state_dict(),
        )

    def apply_weights(
        self,
        global_weight,
        global_protos,
        new_weight,
        **kwargs,
    ):
        """Accept ps model params, then update local model

        Args:
            weights: global weight from params server
        """

        def update_base_model(self, global_weight):
            local_weight = self.local_model.state_dict()
            w_local_keys = self.w_local_keys
            for k in local_weight.keys():
                if k not in w_local_keys:
                    local_weight[k] = global_weight[k]
            self.local_model.load_state_dict(local_weight)

        def update_local_classifier(self, new_weight):
            local_weight = self.local_model.state_dict()
            w_local_keys = self.w_local_keys
            for k in local_weight.keys():
                if k in w_local_keys:
                    local_weight[k] = new_weight[k]
            self.local_model.load_state_dict(local_weight)

        def update_global_protos(self, global_protos):
            self.global_protos = global_protos
            global_protos = self.global_protos
            g_classes, g_protos = [], []
            for i in range(self.num_classes):
                if i in global_protos:
                    g_classes.append(torch.tensor(i))
                    g_protos.append(global_protos[i])
                else:
                    logging.info(f"Key {i} not found in global_protos")
                    logging.info(f'global protos type: {type(global_protos)}')
            self.g_classes = torch.stack(g_classes).to(self.exe_device)
            self.g_protos = torch.stack(g_protos)

        update_base_model(self, global_weight)
        update_global_protos(self, global_protos)
        agg_g = kwargs.get("agg_g", 1)
        if agg_g:
            update_local_classifier(self, new_weight)


@register_strategy(strategy_name="fed_pac", backend="torch")
class PYUFedPAC(FedPAC):
    pass
