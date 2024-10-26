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
import random
from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
import secretflow as sf
import torch
import torch.nn as nn
import torch.optim as optim
from secretflow.device import PYU, PYUObject, proxy
from secretflow.ml.nn.callbacks.attack import AttackCallback
from torch.utils.data import DataLoader, TensorDataset


def convert_to_ndarray(*data: List) -> Union[List[jnp.ndarray], jnp.ndarray]:
    def _convert_to_ndarray(hidden):
        # processing data
        if not isinstance(hidden, jnp.ndarray):
            if isinstance(hidden, torch.Tensor):
                hidden = jnp.array(hidden.detach().cpu().numpy())
            if isinstance(hidden, np.ndarray):
                hidden = jnp.array(hidden)
        return hidden

    if isinstance(data, Tuple) and len(data) == 1:
        # The case is after packing and unpacking using PYU, a tuple of length 1 will be obtained, if 'num_return' is not specified to PYU.
        data = data[0]
    if isinstance(data, (List, Tuple)):
        return [_convert_to_ndarray(d) for d in data]
    else:
        return _convert_to_ndarray(data)


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class VillainAttack(AttackCallback):
    """
    Implementation of backdoor attack algorithm: Villain Attack
    In villain attack, we first infer samples that belong to the target class leveraging embedding swapping, candidate selection. After that, we poison these samples to inject the back.
    Attributes:
        attack_party (PYU): The party performing the attack in the federated learning setup.
        exec_device (str): The device (e.g., "cpu", "gpu") used for executing computations.
        origin_target_idx (int): Index of one original known sample belongs to the target class.
        theta (float): Threshold parameter for label inference in gradient comparison.
        mu (float): Threshold parameter for label inference in gradient comparison.
        beta (float): Parameter controlling the magnitude of the trigger in trigger fabrication. Default 0.4.
        batch_size (int): Number of samples processed per batch. Default 128.
        candi_n (int): Number of candidate samples selected during candidate selection. Default 14.
        infer_epoch (int): The epoch at which the inference phase begins. Default 5.
        attack_epoch (int): The epoch at which the attack phase begins. Default 7.
    """

    def __init__(
        self,
        attack_party: PYU,
        origin_target_idx: int,
        theta: float,
        mu: float = 1.0,
        beta: float = 0.4,
        batch_size: int = 128,
        candi_n: int = 14,
        infer_epoch: int = 5,
        attack_epoch: int = 9,
        exec_device: str = "cpu",
    ):
        self.attack_party = attack_party
        self.exec_device = exec_device
        self.candi_n = candi_n
        self.theta = theta
        self.mu = mu
        self.beta = beta
        self.batch_size = batch_size
        self.infer_epoch = infer_epoch
        self.attack_epoch = attack_epoch
        self.origin_target_idx = origin_target_idx

        self.cnt = 0
        self.epoch = 0
        self.batch = 0
        self.batch_offset = 0

    def backdoor_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        # 使用数据增强技术（如翻转、旋转等）来增强投毒样本
        augmented_data = torch.flip(data, dims=[3])  # 水平翻转
        return augmented_data

    def on_train_begin(self, logs=None):
        def init_worker(attack_worker, origin_target_idx):
            attack_worker.gradients = {}
            attack_worker.top_n_idx = {}
            attack_worker.top_n_hiddens = []
            attack_worker.target_hiddens_np = {}
            attack_worker.target_idx = [origin_target_idx]

            attack_worker.classifier = None
            attack_worker.optimizer = None
            attack_worker.criterion = nn.BCELoss()

        self._workers[self.attack_party].apply(init_worker, self.origin_target_idx)

    def on_epoch_begin(self, epoch=None, logs=None):
        def reset_target_hiddens(attack_worker):
            attack_worker.target_hiddens_np = {}

        self.epoch = epoch
        if self.infer_epoch < epoch < self.attack_epoch:
            self.cnt += 1
        self._workers[self.attack_party].apply(reset_target_hiddens)

    def on_train_batch_begin(self, batch):
        self.batch = batch
        self.batch_offset = batch * self.batch_size

    def on_agglayer_forward_begin(self, hiddens=None):
        def hiddens_swapping(
            attack_worker, hiddens, batch, batch_size, candi_n, exec_device
        ):
            def initialize_classifier(sample_embedding: torch.Tensor):
                """
                Initialize the binary classifier based on the input dimension of the sample embedding.
                """
                input_dim = sample_embedding.size(1)
                classifier = BinaryClassifier(input_dim)
                optimizer = optim.Adam(classifier.parameters(), lr=0.01)
                return classifier, optimizer

            hiddens_np = convert_to_ndarray(hiddens)
            # 先记录用于已知的目标hiddens
            hidden_idx = [
                x for x in range(batch * batch_size, (batch + 1) * batch_size)
            ]
            for idx in np.intersect1d(attack_worker.target_idx, hidden_idx):
                attack_worker.target_hiddens_np[idx] = hiddens_np[
                    idx - batch * batch_size
                ]

            # 使用二分类器预测hiddens是否属于目标标签
            if attack_worker.classifier is None:
                (
                    attack_worker.classifier,
                    attack_worker.optimizer,
                ) = initialize_classifier(hiddens)
            attack_worker.classifier.eval()
            with torch.no_grad():
                predictions = attack_worker.classifier(hiddens_np).squeeze()
            # 选择预测结果最高的前n个hiddens
            attack_worker.top_n_idx[batch] = torch.argsort(
                predictions, descending=True
            )[:candi_n].tolist()
            # 记录hiddens用于训练
            for idx in attack_worker.top_n_idx[batch]:
                attack_worker.top_n_hiddens[batch][idx] = hiddens[idx]
            # 从已知目标样本的hiddens中随机选取替换这n个hiddens
            for idx in attack_worker.top_n_idx:
                hiddens_np[idx] = attack_worker.target_hiddens_np[
                    random.choice(attack_worker.target_idx)
                ]
            self.attack_party._h = torch.tensor(hiddens_np).to(exec_device)

        def trigger_fabrication(attack_worker, target, beta: float):
            def trigger(hidden, beta: float):
                hidden = hiddens[idx]
                delta = torch.std(hidden, dim=0).mean().item()
                # 构造触发器增量 Δ
                delta_pattern = [delta, delta, -delta, -delta] * (hidden.size(0) // 4)
                # 选择 m 个标准差最大的元素作为触发器区域
                std_dev = torch.std(hidden, dim=0)
                m_elements_indices = torch.argsort(std_dev, descending=True)[
                    : len(delta_pattern)
                ]
                # 构造触发器掩码 M
                M = torch.zeros_like(hidden)
                M[m_elements_indices] = 1
                M = M * random.uniform(0.6, 1.2)
                # 触发器 E = M ⊗ (β · Δ)
                E = M @ (beta * torch.tensor(delta_pattern, device=hidden.device))
                return E

            # 计算所有样本中触发器维度中元素的平均标准差
            if attack_worker._training:
                hiddens = attack_worker._h
                for idx in target:
                    hidden = hiddens[idx]
                    E = trigger(hidden, beta)
                    hiddens[idx] = hidden + E
            else:
                hiddens = []
                for h in attack_worker._h:
                    E = trigger(h, beta)
                    hiddens.append(h + E)
            attack_worker._h = hiddens

        if self.infer_epoch <= self.epoch < self.attack_epoch and self.cnt % 2 != 0:
            self._workers[self.attack_party].apply(
                hiddens_swapping,
                hiddens,
                self.batch,
                self.batch_size,
                self.candi_n,
                self.exec_device,
            )

        if self.epoch >= self.attack_epoch:
            idx = []
            for i in self.target_idx:
                idx.append(i - self.batch_offset)
            hidden_idx = [
                x for x in range(self.batch_offset, self.batch_offset + self.batch_size)
            ]
            batch_target = np.intersect1d(idx, hidden_idx)
            self._workers[self.attack_party].apply(
                trigger_fabrication, batch_target, self.beta
            )

    def on_agglayer_backward_end(self, gradients=None):
        def record_gradient(
            attack_worker, index: int, gradient: torch.Tensor, is_target: bool = False
        ):
            if index not in attack_worker.gradients:
                attack_worker.gradients[index] = {"original": None, "target": None}

            if is_target:
                attack_worker.gradients[index]["target"] = gradient
            else:
                attack_worker.gradients[index]["original"] = gradient

        def tune_classifier(attack_worker, batch, train_labels):
            # 使用嵌入交换结果微调分类器
            attack_worker.classifier.train()
            dataset = TensorDataset(attack_worker.top_n_hiddens[batch], train_labels)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
            for data, label in dataloader:
                attack_worker.optimizer.zero_grad()
                output = attack_worker.classifier(data).squeeze()
                loss = attack_worker.criterion(output, label)
                loss.backward()
                attack_worker.optimizer.step()

        def gradient_compare(
            attack_worker, gradients, batch, batch_offset, theta, mu, cnt
        ):
            grad_np = convert_to_ndarray(gradients)
            # 如果是第一次训练，记录原始梯度
            if cnt % 2 == 0:
                for idx in attack_worker.top_n_idx[batch]:
                    idx += batch_offset
                    record_gradient(attack_worker, idx, grad_np[idx])
            # 如果是第二次训练，记录目标梯度并进行判断
            else:
                train_labels = []
                for idx in attack_worker.top_n_idx[batch]:
                    idx += batch_offset
                    attack_worker.record_gradient(idx, grad_np[idx], is_target=True)
                    g_i = attack_worker.gradients[idx]["original"]
                    g_t_i = attack_worker.gradients[idx]["target"]
                    if g_i is not None and g_t_i is not None:
                        norm_g_i = torch.norm(g_i, p=2)
                        norm_g_t_i = torch.norm(g_t_i, p=2)
                        if norm_g_t_i / norm_g_i <= theta and norm_g_i <= mu:
                            attack_worker.target_idx.append(idx)
                            train_labels.append(1)
                        else:
                            train_labels.append(0)
                tune_classifier(attack_worker, batch, train_labels)

        if self.epoch >= self.infer_epoch:
            self._workers[self.attack_party].apply(
                gradient_compare,
                gradients,
                self.batch,
                self.batch_offset,
                self.theta,
                self.mu,
                self.cnt,
            )

    def get_attack_metrics(self, preds, target_class: int):
        preds_plain = []
        for pred in preds:
            preds_plain.append(torch.argmax(pred, dim=1))
        preds_plain = torch.cat(preds_plain, dim=0)

        pred_np: np.ndarray = preds_plain.cpu().numpy()
        poison_pred = pred_np
        poison_pred = poison_pred == target_class
        nums_poison_true = sum(poison_pred)
        acc = nums_poison_true / len(poison_pred)
        return {"acc": acc}
