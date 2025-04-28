# Copyright 2023 Ant Group Co., Ltd.
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

from typing import List, Union

import numpy as np
import torch

from secretflow.device import PYU
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback


class GradReplaceAttack(AttackCallback):
    """
    Implemention of backdoor attack algorithm: Gradient Replace Attack.
    In Gradient Replace Attack, we want to make poison samples' prediction be target label. We assume that same target samples(samples with target label) has already been collected. During training, we replace target sample's input x by poison x, and replace poison sample's gradient by gradient at target sample's position.
    Attributes:
        attack_party: attack party.
        target_idx: list of target samples' indexes, dataloader should return (data, index).
        poison_idx: list of poison samples' indexes.
        poison_input: poison samples' input feature, numpy.ndarray.
        gamma: multiplier for gradient replacement.
        batch_size: batch size, for index calculation.
        blurred: whether to make poison sample noise.
        hidden_size: hidden output's shape, without batch_size.
    """

    def __init__(
        self,
        attack_party: PYU,
        target_idx: List[int],
        poison_idx: List[int],
        poison_input: Union[np.ndarray, List[np.ndarray]],
        gamma: int,
        batch_size: int,
        blurred: bool = False,
        exec_device: str = "cpu",
        **params,
    ):
        super().__init__(
            **params,
        )
        self.attack_party = attack_party
        self.target_idx = target_idx
        self.poison_idx = poison_idx
        self.poison_input = poison_input
        self.gamma = gamma
        self.batch_size = batch_size
        self.blurred = blurred

        self.target_offsets = None
        self.poison_offsets = None
        self.exec_device = exec_device

    def on_train_batch_begin(self, batch):
        data_idx = [
            x for x in range(batch * self.batch_size, (batch + 1) * self.batch_size)
        ]

        target_set = np.intersect1d(data_idx, self.target_idx)
        self.target_offsets = np.where(np.isin(data_idx, target_set))[0]
        poison_set = np.intersect1d(data_idx, self.poison_idx)
        self.poison_offsets = np.where(np.isin(data_idx, poison_set))[0]

    def on_base_forward_begin(self):
        def replace_input(
            attack_worker,
            target_offsets,
            poison_offsets,
            poison_input,
            blurred,
            exec_device,
        ):
            if attack_worker._training:
                t_len = len(target_offsets)
                if t_len > 0:
                    # poison x -> target x
                    if isinstance(attack_worker._data_x, torch.Tensor):
                        if isinstance(poison_input, list):
                            assert len(poison_input) == 1
                            poison_input = poison_input[0]
                        choices = np.random.choice(
                            len(poison_input), (t_len,), replace=True
                        )
                        data_np = attack_worker._data_x.cpu().numpy()
                        data_np[target_offsets] = poison_input[choices]
                        attack_worker._data_x = torch.from_numpy(data_np).to(
                            exec_device
                        )
                    else:
                        choices = np.random.choice(
                            len(poison_input[0]), (t_len,), replace=True
                        )
                        data_np = [data.cpu().numpy() for data in attack_worker._data_x]
                        for i in range(len(data_np)):
                            data_np[i][target_offsets] = poison_input[i][choices]
                        attack_worker._data_x = [
                            torch.from_numpy(data).to(exec_device) for data in data_np
                        ]

                p_len = len(poison_offsets)
                if blurred and p_len > 0:
                    if isinstance(attack_worker._data_x, torch.Tensor):
                        data_np = attack_worker._data_x.cpu().numpy()
                        rnd_shape = (p_len,) + list(data_np.shape[1:])
                        data_np[poison_offsets] = np.random.randn(*rnd_shape)
                        attack_worker._data_x = torch.from_numpy(data_np).to(
                            exec_device
                        )
                    else:
                        data_np = [data.cpu().numpy() for data in attack_worker._data_x]
                        for i in range(len(data_np)):
                            rnd_shape = (p_len,) + list(data_np[i].shape[1:])
                            data_np[i][poison_offsets] = np.random.randn(*rnd_shape)
                        attack_worker._data_x = [
                            torch.from_numpy(data).to(exec_device) for data in data_np
                        ]

        if len(self.target_offsets) > 0 or (
            self.blurred and len(self.poison_offsets) > 0
        ):
            self._workers[self.attack_party].apply(
                replace_input,
                self.target_offsets,
                self.poison_offsets,
                self.poison_input,
                self.blurred,
                self.exec_device,
            )

    def on_base_backward_begin(self):
        def replace_gradient(
            attack_worker, target_offsets, poison_offsets, gamma, exec_device
        ):
            # target grad -> poison grad
            choice = np.random.choice(target_offsets, (1,), replace=True)
            for idx, grad in enumerate(attack_worker._gradient):
                grad_np = grad.cpu().numpy()
                grad_np[poison_offsets] = gamma * grad_np[choice]
                attack_worker._gradient[idx] = torch.from_numpy(grad_np).to(exec_device)

        if len(self.target_offsets) > 0 and len(self.poison_offsets) > 0:
            self._workers[self.attack_party].apply(
                replace_gradient,
                self.target_offsets,
                self.poison_offsets,
                self.gamma,
                self.exec_device,
            )

    def get_attack_metrics(self, preds, target_class: int, eval_poison_set: np.ndarray):
        preds_plain = []
        for pred in preds:
            preds_plain.append(torch.argmax(pred, dim=1))
        preds_plain = torch.cat(preds_plain, dim=0)

        pred_np: np.ndarray = preds_plain.cpu().numpy()
        poison_pred = pred_np[eval_poison_set]
        poison_pred = poison_pred == target_class
        nums_poison_true = sum(poison_pred)
        acc = nums_poison_true / len(poison_pred)
        return {"acc": acc}
