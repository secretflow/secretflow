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

from secretflow import wait
from secretflow.device import PYU, DeviceObject, PYUObject
from secretflow_fl.ml.nn.callbacks.callback import Callback
from secretflow_fl.ml.nn.core.torch import BuilderType
from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy
from secretflow_fl.security.privacy.mechanism.mechanism_fl import GaussianModelDP


# the client strategy in FedSMP
class FedSMP(BaseTorchModel):
    """
    Implemention of FedSMP algorithm in paper Federated Learning with Sparsified Model Perturbation: Improving Accuracy under Client-Level Differential Privacy: https://ieeexplore.ieee.org/abstract/document/10360319/.

    FedSMP is a novel differentially-private FL scheme which can provide a client-level DP guarantee while maintaining high model accuracy. To mitigate the impact of privacy protection on model accuracy, Fed-SMP leverages a new technique called Sparsified Model Perturbation (SMP) where local models are sparsified first before being perturbed by Gaussian noise.
    """

    def __init__(
        self,
        builder_base: BuilderType,
        random_seed: int = None,
        skip_bn: bool = False,
        **kwargs,
    ):
        super().__init__(builder_base, random_seed=random_seed, skip_bn=skip_bn)

        self.grad_mask = None
        self.compression_ratio = kwargs.get("compression_ratio", 1.0)

    def train_step(
        self,
        weights: list,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params, then do local train

        Args:
            weights: global weight from params server and grad mask
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """
        assert self.model is not None, "Model cannot be none, please give model define"
        self.model.train()
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        if weights is not None:
            self.set_weights(weights)

        dp_strategy = kwargs.get("dp_strategy", None)

        # get grad mask
        if weights:
            self.grad_mask = weights[1]

        # copy the model weights before local training
        init_weights = copy.deepcopy(self.get_weights(return_numpy=True))

        # local training
        num_sample = 0
        dp_strategy = kwargs.get("dp_strategy", None)
        logs = {}
        loss: torch.Tensor = None

        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]

            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)

            if self.model.automatic_optimization:
                self.model.backward_step(loss)

        loss_value = loss.item()
        logs["train-loss"] = loss_value

        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        model_weights = self.get_weights(return_numpy=True)

        # FedSMP DP operation
        grads = [v2 - v1 for v1, v2 in zip(init_weights, model_weights)]

        # add mask to sparsify the grads
        grads = [v2 * v1 for v1, v2 in zip(grads, self.grad_mask)]
        grads = [v / self.compression_ratio for v in grads]

        # add noise
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                grads = dp_strategy.model_gdp(grads)

        # add mask to the grads after the noise injection
        grads = [v2 * v1 for v1, v2 in zip(grads, self.grad_mask)]

        model_weights = [v1 + v2 for v1, v2 in zip(init_weights, grads)]

        return model_weights, num_sample

    def set_weights(self, weights):
        """set weights of client model"""
        if len(weights) == 2:
            self.grad_mask = weights[1]
            weights = weights[0]

        if self.skip_bn:
            self.model.update_weights_not_bn(weights)
        else:
            self.model.update_weights(weights)

    def apply_weights(self, weights, **kwargs):
        """Accept ps model params, then update local model

        Args:
            weights: global weight from params server
        """
        if weights is not None:
            self.set_weights(weights)


@register_strategy(strategy_name='fed_smp', backend='torch')
class PYUFedSMP(FedSMP):
    pass


class FedSMP_server_agg_method:
    """
    The func of the server in FedSMP.
    The server aggregates the params from all clients and generates a new grad mask.

    args:
    model_params_list: the data sent from clients. [[params, mask], ...]
    """

    def __init__(self, compression_ratio):
        self.compression_ratio = compression_ratio

    def _get_dtype(arr):
        if isinstance(arr, np.ndarray):
            return arr.dtype
        else:
            try:
                import tensorflow as tf

                if isinstance(arr, tf.Tensor):
                    return arr.numpy().dtype
            except ImportError:
                return None

    def aggregate(self, model_params_list):

        def average(data, axis, weights=None):
            if isinstance(data[0], (list, tuple)):
                results = []
                for elements in zip(*data):
                    avg = np.average(elements, axis=axis, weights=weights)
                    res_dtype = elements[0].dtype
                    if res_dtype:
                        avg = avg.astype(res_dtype)
                    results.append(avg)
                return results
            else:
                res = np.average(data, axis=axis, weights=weights)
                res_dtype = data[0].dtype
                return res.astype(res_dtype) if res_dtype else res

        params_avg = average(model_params_list, axis=0)

        # the server generate the grad mask
        def generate_grad_mask(compression_ratio, params):
            mask = []
            for v in params:
                submask = np.random.binomial(n=1, p=compression_ratio, size=v.shape)
                mask.append(submask)

            return mask

        grad_mask = generate_grad_mask(self.compression_ratio, params_avg)

        return [(params_avg, grad_mask) for _ in range(len(model_params_list))]
