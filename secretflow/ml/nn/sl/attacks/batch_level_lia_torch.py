# Copyright 2024 Ant Group Co., Ltd.
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

from typing import List

import torch
import torch.nn.functional as F

from secretflow import reveal
from secretflow.device import PYU
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.core.torch import BaseModule, module
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.utils import TorchModel


class BatchLevelLabelInferenceAttack(AttackCallback):
    """
    Implemention of batch-level label inference attack: Defending Batch-Level Label Inference and Replacement Attacks in Vertical Federated Learning: https://ieeexplore.ieee.org/abstract/document/9833321.

    Attributes:
        attack_party: attack party.
        victim_party: victim party.
        victim_hidden_size: hidden size of victim, excluding batch size
        dummy_fuse_model: dummy fuse model for attacker
        label: labels for attacker to calculate attack metrics
        attack_epoch: epoch list to attack, if you want to attack all epochs, set it to None
        attack_batch: batch list to attack, if you want to attack all batches, set it to None
        epochs: attack epochs for each batch, 100 for cifar10 to get good attack accuracy
        exec_device: device for calculation, 'cpu' or 'cuda'
    """

    def __init__(
        self,
        attack_party: PYU,
        victim_party: PYU,
        victim_hidden_size: List[int],
        dummy_fuse_model: TorchModel,
        label,
        lr: float = 0.05,
        label_size: List[int] = [10],
        attack_epoch: List[int] = None,
        attack_batch: List[int] = None,
        epochs: int = 100,
        exec_device: str = 'cpu',
        **params,
    ):
        super().__init__(
            **params,
        )
        self.attack_party = attack_party
        self.victim_party = victim_party

        self.attack_epoch = attack_epoch
        self.attack_batch = attack_batch

        self.victim_hidden_size = victim_hidden_size
        self.dummy_fuse_model = dummy_fuse_model
        self.label = label
        self.lr = lr
        self.label_size = label_size
        self.att_epochs = epochs
        self.exec_device = exec_device

        self.logs = {}
        self.metrics = None

        self.attack = False

    def on_train_begin(self, logs=None):
        def init_attacker(
            attack_worker: SLBaseTorchModel,
            victim_hidden_size,
            dummy_fuse_model,
            label,
            lr,
            label_size,
            exec_device,
            att_epochs,
        ):
            attacker = BatchLevelLabelInferenceAttacker(
                attack_worker.model_base,
                victim_hidden_size,
                dummy_fuse_model,
                label,
                lr=lr,
                label_size=label_size,
                epochs=att_epochs,
                exec_device=exec_device,
            )

            attack_worker.attacker = attacker

        self._workers[self.attack_party].apply(
            init_attacker,
            self.victim_hidden_size,
            self.dummy_fuse_model,
            self.label,
            self.lr,
            self.label_size,
            self.exec_device,
            self.att_epochs,
        )

    def on_epoch_begin(self, epoch=None, logs=None):
        if self.attack_epoch is None or (
            epoch is not None and epoch in self.attack_epoch
        ):
            self.attack = True

            def clear_pred_list(worker):
                worker.attacker.pred = []

            self._workers[self.attack_party].apply(clear_pred_list)
        else:
            self.attack = False

    def on_train_batch_begin(self, batch):
        if (
            self.attack
            and self.attack_batch is not None
            and batch not in self.attack_batch
        ):
            self.attack = False

    def on_base_backward_begin(self):
        def lia_attack(worker):
            real_grad = torch.autograd.grad(
                worker._h,
                worker.model_base.parameters(),
                grad_outputs=worker._gradient,
                retain_graph=True,
            )

            worker.attacker.attack_batch_label(real_grad, worker._data_x)

        if self.attack:
            res = self._workers[self.attack_party].apply(lia_attack)

    def on_epoch_end(self, epoch=None, logs=None):
        if self.attack:

            def cal_metrics(worker):
                return worker.attacker.calc_label_recovery_rate()

            recovery_rate = reveal(self._workers[self.attack_party].apply(cal_metrics))
            self.metrics = {'recovery_rate': recovery_rate}
            print(f'epoch {epoch} metrics: {self.metrics}')

    def get_attack_metrics(self):
        return self.metrics


class BatchLevelLabelInferenceAttacker:
    def __init__(
        self,
        base_model: BaseModule,
        victim_hidden_size: List[int],
        dummy_fuse_model_builder: TorchModel,
        label,
        lr: float = 0.05,
        label_size: List[int] = [10],
        epochs: int = 1,
        exec_device: str = 'cpu',
    ):
        super().__init__()

        self.base_model = base_model
        self.pred = []
        self.victim_hidden_size = victim_hidden_size
        self.label_size = label_size
        self.dummy_fuse_model = module.build(
            dummy_fuse_model_builder, device=exec_device
        )
        self.lr = lr
        self.epochs = epochs
        self.exec_device = exec_device
        # for accuracy
        self.label = torch.tensor(label).to(self.exec_device)

    def attack_batch_label(self, real_grad, data_x):
        batch_size = data_x.size()[0]

        # guess and optimize
        dummy_hidden_victim = (
            torch.randn([batch_size] + self.victim_hidden_size)
            .to(self.exec_device)
            .requires_grad_(True)
        )
        dummy_label = (
            torch.randn([batch_size] + self.label_size)
            .to(self.exec_device)
            .requires_grad_(True)
        )

        optimizer = torch.optim.Adam(
            [dummy_hidden_victim, dummy_label]
            + list(self.dummy_fuse_model.parameters()),
            lr=self.lr,
        )

        criterion = torch.nn.CrossEntropyLoss()
        for iters in range(self.epochs):

            def closure():
                optimizer.zero_grad()
                # only support concat agglayer
                dummy_pred = self.dummy_fuse_model(
                    [self.base_model(data_x), dummy_hidden_victim]
                )
                # only multi-classification
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)

                dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                dummy_grad = torch.autograd.grad(
                    dummy_loss, self.base_model.parameters(), create_graph=True
                )

                grad_diff = 0
                for gx, gy in zip(real_grad, dummy_grad):
                    grad_diff += ((gx - gy) ** 2).sum()

                grad_diff.backward()

                return grad_diff

            optimizer.step(closure)

        self.pred.append(dummy_label)

    def calc_label_recovery_rate(self):
        self.pred = torch.concat(self.pred, dim=0)
        success = torch.sum(torch.argmax(self.pred, dim=-1) == self.label).item()
        total = self.label.size()[0]
        return success / total
