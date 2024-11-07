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

from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy
from secretflow.ml.nn.core.torch import BuilderType

from copy import deepcopy
import torch
from typing import Dict, Tuple
import copy
import numpy as np
import logging
from torch import nn

class FedPAC(BaseTorchModel):
    def __init__(
        self,
        builder_base: BuilderType,
        random_seed: int = None,
        skip_bn: bool = False,
        **kwargs,
    ):
        super().__init__(builder_base, random_seed=random_seed, **kwargs)
        self.num_classes = kwargs.get("num_classes", 10)
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = self.model
        self.w_local_keys = self.local_model.classifier_weight_keys
        self.local_ep_rep = 1
        self.global_protos = {}
        self.g_protos = None
        self.mse_loss = nn.MSELoss()
        self.lam = kwargs.get("lam", 1.0)  # 1.0 for mse_loss
    
    def prior_label(self, dataset):
        py = torch.zeros(self.num_classes)
        total = len(dataset.dataset)
        data_loader = iter(dataset)
        iter_num = len(data_loader)
        for it in range(iter_num):
            images, labels = next(data_loader)
            for i in range(self.num_classes):
                py[i] = py[i] + (i == labels).sum()
        py = py / (total)
        return py

    def size_label(self, dataset):
        py = torch.zeros(self.num_classes)
        total = len(dataset.dataset)
        data_loader = iter(dataset)
        iter_num = len(data_loader)
        for it in range(iter_num):
            images, labels = next(data_loader)
            for i in range(self.num_classes):
                py[i] = py[i] + (i == labels).sum()

        py = py / (total)
        size_label = py * total
        return size_label

    def sample_number(self):
        data_size = len(self.train_set.dataset)
        w = torch.tensor(data_size).to(self.exe_device)
        return w

    def evaluate(self, step_per_epoch=0) -> Tuple[float, float]:
        assert self.model is not None, "Model cannot be none, please give model define"
        assert (
            len(self.model.metrics) > 0
        ), "Metric cannot be none, please give metric by 'TorchModel'"
        self.model.eval()
        model = self.local_model
        device = self.exe_device
        correct = 0
        eval_loader = self.eval_set
        loss_test = []
        self.reset_metrics()
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                model.update_metrics(outputs, labels)
                loss = self.criterion(outputs, labels)
                loss_test.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            result = {}
            self.transform_metrics(result, stage="eval")

        if self.logs is None:
            self.wrapped_metrics.extend(self.wrap_local_metrics())
            return self.wrap_local_metrics()
        else:
            val_result = {}
            for k, v in result.items():
                val_result[f"val_{k}"] = v
            self.logs.update(val_result)
            self.wrapped_metrics.extend(self.wrap_local_metrics(stage="val"))
            return self.wrap_local_metrics(stage="val")

    def get_local_protos(self, step, train_steps):
        model = self.local_model
        local_protos_list = {}
        step_counter = 0
        total_steps = 0
        for inputs, labels in self.train_set:
            if total_steps < step:
                total_steps += 1
                continue
            if step_counter >= train_steps:
                break
            inputs, labels = inputs.to(self.exe_device), labels.to(self.exe_device)
            features, outputs = model(inputs)
            protos = features.clone().detach()
            for i in range(len(labels)):
                if labels[i].item() in local_protos_list.keys():
                    local_protos_list[labels[i].item()].append(protos[i, :])
                else:
                    local_protos_list[labels[i].item()] = [protos[i, :]]
            step_counter += 1
            total_steps += 1

        local_protos = {}
        for [label, proto_list] in local_protos_list.items():
            proto = 0 * proto_list[0]
            for p in proto_list:
                proto += p
            local_protos[label] = proto / len(proto_list)
        return local_protos

    def get_local_protos_with_entire_dataset(self):
        model = self.local_model
        local_protos_list = {}
        for inputs, labels in self.train_set:
            inputs, labels = inputs.to(self.exe_device), labels.to(self.exe_device)
            features, outputs = model(inputs)
            protos = features.clone().detach()
            for i in range(len(labels)):
                if labels[i].item() in local_protos_list.keys():
                    local_protos_list[labels[i].item()].append(protos[i, :])
                else:
                    local_protos_list[labels[i].item()] = [protos[i, :]]

        local_protos = {}
        for [label, proto_list] in local_protos_list.items():
            proto = 0 * proto_list[0]
            for p in proto_list:
                proto += p
            local_protos[label] = proto / len(proto_list)
        return local_protos

    def statistics_extraction(self):
        model = self.local_model
        cls_keys = self.w_local_keys
        g_params = (
            model.state_dict()[cls_keys[0]]
            if isinstance(cls_keys, list)
            else model.state_dict()[cls_keys]
        )
        d = g_params[0].shape[0]
        feature_dict = {}
        datasize = torch.tensor(len(self.train_set.dataset)).to(self.exe_device)
        with torch.no_grad():
            for inputs, labels in self.train_set:
                inputs, labels = inputs.to(self.exe_device), labels.to(self.exe_device)
                features, outputs = model(inputs)
                feat_batch = features.clone().detach()
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi in feature_dict.keys():
                        feature_dict[yi].append(feat_batch[i, :])
                    else:
                        feature_dict[yi] = [feat_batch[i, :]]
        for k in feature_dict.keys():
            feature_dict[k] = torch.stack(feature_dict[k])

        py = self.prior_label(self.train_set).to(self.exe_device)
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.num_classes, d), device=self.exe_device)
        for k in range(self.num_classes):
            if k in feature_dict.keys():
                feat_k = feature_dict[k]
                num_k = feat_k.shape[0]
                feat_k_mu = feat_k.mean(dim=0)
                h_ref[k] = py[k] * feat_k_mu
                v += (
                    py[k] * torch.trace((torch.mm(torch.t(feat_k), feat_k) / num_k))
                ).item()
                v -= (py2[k] * (torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v / datasize.item()
        return v, h_ref

    def get_statistics(
        self,
    ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor]:
        v, h_ref = self.statistics_extraction()
        label_size = self.size_label(self.train_set).to(self.exe_device)
        sample_num = self.sample_number()
        dataset_size = torch.tensor(len(self.train_set.dataset)).to(self.exe_device)
        return v, h_ref, label_size, dataset_size, self.w_local_keys, sample_num

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
            step: current train step
            cur_steps: current train step in the whole training process
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

            loss1 = self.mse_loss(protos_new, protos)
            loss = loss0 + self.lam * loss1
            loss.backward()
            optimizer.step()
            iter_loss.append(loss.item())

        train_loss = sum(iter_loss) / len(iter_loss)
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
