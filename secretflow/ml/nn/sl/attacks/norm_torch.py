# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from secretflow import PYU, PYUObject, reveal
from secretflow.ml.nn.callbacks.attack import AttackCallback


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


def extract_intermidiate_gradient(self, outputs):
    self.backward_gradient(outputs.grad)
    return self.clients[self.target_client_index].grad_from_next_client


def norm_attack(self, my_grad):
    grad_np = convert_to_ndarray(my_grad)
    grad_np_ = grad_np[0]
    grad_norm_np = jnp.sqrt(jnp.sum(jnp.square(grad_np_), axis=1))
    return grad_norm_np


def compute_auc(self, label, epoch_g_norm):
    """Compute the attack leak AUC on the given true label and predict label."""
    label = label.values
    normattack_pred = jnp.concatenate(epoch_g_norm)
    y_true_numpy = label.tolist()
    y_pred_numpy = normattack_pred.tolist()
    return roc_auc_score(y_true_numpy, y_pred_numpy)


class NormAttack(AttackCallback):
    """Norm Attack for label inferences.
    reference: https://arxiv.org/abs/2102.08504
    Notes: this attack is only supported for Binary classification model.
    """

    def __init__(self, attack_party: PYU, label, **kwargs):
        """
        Args:
            attack_party: the attack party pyu, who does not have labels.
            label: True labels for compute the attack AUC.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.attack_party = attack_party
        self.all_g_norm: Dict[int, List[PYUObject]] = {}
        self.label = label
        self.epoch = 0

    def on_epoch_begin(self, epoch=None, logs=None):
        self.all_g_norm[epoch] = []
        self.epoch = epoch

    def on_agglayer_backward_end(self, gradients=None):
        my_grad = gradients[self.attack_party]
        grad_norm_np = self._workers[self.attack_party].apply(norm_attack, my_grad)
        self.all_g_norm[self.epoch].append(grad_norm_np)

    def get_attack_metrics(self):
        metrics = []
        for epoch_g_norm in self.all_g_norm.values():
            metric = self._workers[self.attack_party].apply(
                compute_auc, self.label, epoch_g_norm
            )
            metrics.append(metric)
        metrics = reveal(metrics)

        return {'auc': max(metrics), "auc_list": metrics}
