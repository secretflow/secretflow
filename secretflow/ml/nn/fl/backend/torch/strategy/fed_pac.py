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
        self, cur_steps: int, train_steps: int, **kwargs,
    ) -> Tuple[
        float,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        float,
        float,
        float,
        float,
        np.ndarray,
        list,
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
        v, h_ref = self.statistics_extraction()
        # logging.info(f"data type of h after use statistics_extraction:{type(h_ref)}")
        size_label = self.size_label(self.train_set).to(self.exe_device)
        agg_weight = self.aggregate_weight()
        model = self.local_model
        model.train()
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        logs = {}
        round_loss = []
        iter_loss = []
        model.zero_grad()
        # grad_accum = []
        global_protos = self.global_protos
        # g_protos = self.g_protos

        acc0, _ = self.local_test(self.eval_set)
        self.last_model = deepcopy(model)

        # get local prototypes before training, dict:={label: list of sample features}
        local_protos1 = self.get_local_protos()

        # Set optimizer for the local updates, default sgd
        lr = kwargs.get("lr", 0.01)
        optimizer = torch.optim.SGD(
            model.parameters(), lr, momentum=0.5, weight_decay=0.0005
        )

        local_ep_rep = train_steps
        epoch_classifier = 1
        train_steps = int(epoch_classifier + local_ep_rep)

        if train_steps > 0:
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
                data_loader = iter(self.train_set)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = (
                        images.to(self.exe_device),
                        labels.to(self.exe_device),
                    )
                    model.zero_grad()
                    protos, output = model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
                round_loss.append(sum(iter_loss) / len(iter_loss))
                iter_loss = []
            # ---------------------------------------------------------------------------

            acc1, _ = self.local_test(self.eval_set)

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

            for ep in range(local_ep_rep):
                data_loader = iter(self.train_set)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = (
                        images.to(self.exe_device),
                        labels.to(self.exe_device),
                    )
                    model.zero_grad()
                    protos, output = model(images)
                    loss0 = self.criterion(output, labels)
                    loss1 = 0
                    if cur_steps > 0:
                        loss1 = 0
                        protos_new = protos.clone().detach()
                        for i in range(len(labels)):
                            yi = labels[i].item()
                            if yi in global_protos:
                                protos_new[i] = global_protos[yi].detach()
                            else:
                                protos_new[i] = local_protos1[yi].detach()
                        loss1 = self.mse_loss(protos_new, protos)
                    loss = loss0 + self.lam * loss1
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
                round_loss.append(sum(iter_loss) / len(iter_loss))
                iter_loss = []

        # ------------------------------------------------------------------------
        local_protos2 = self.get_local_protos()
        round_loss1 = round_loss[0]
        round_loss2 = round_loss[-1]
        acc2, _ = self.local_test(self.eval_set)

        logs["train-loss"] = round_loss2
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        # logging.info(f"data type of h before send back train results:{type(h_ref)}")
        # logging.info(f"physical device : {self.exe_device}")
        # logging.info(f"physical device type: {type(self.exe_device)}")
        return (
            self.exe_device,
            v,
            h_ref,
            size_label,
            agg_weight,
            model.state_dict(),
            round_loss1,
            round_loss2,
            acc0,
            acc2,
            local_protos2,
            self.w_local_keys,
        )

    def apply_weights(
        self,
        global_weight,
        global_protos,
        new_weight,
        cur_steps: int,
        train_steps: int,
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
                g_classes.append(torch.tensor(i))
                g_protos.append(global_protos[i])
            self.g_classes = torch.stack(g_classes).to(self.device)
            self.g_protos = torch.stack(g_protos)

        update_base_model(self, global_weight)
        update_global_protos(self, global_protos)
        agg_g = kwargs.get("agg_g", 1)
        if agg_g and cur_steps < train_steps:
            update_local_classifier(self, new_weight)


@register_strategy(strategy_name="fed_pac", backend="torch")
class PYUFedPAC(FedPAC):
    pass
