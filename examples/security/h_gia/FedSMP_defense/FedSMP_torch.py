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
        self.compression_ratio = 1.0
        self.noise_multiplier = kwargs.get("noise_multiplier", 0.0)
        self.l2_norm_clip = kwargs.get("l2_norm_clip", 10000)
        self.num_clients = kwargs.get("num_clients", 1)

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
        if refresh_data:
            self._reset_data_iter()
        if weights is not None:
            self.set_weights(weights)

        dp_strategy = kwargs.get("dp_strategy", None)

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


# the operations of the server in FedSMP
class FedSMPServerCallback(Callback):
    def __init__(self, server, compression_ratio, global_net, **kwargs):
        super().__init__(**kwargs)
        self.server = server
        self.compression_ratio = compression_ratio
        self.global_net = global_net

    def on_train_batch_begin(self, batch):

        # the server generate the grad mask
        def generate_grad_mask(compression_ratio, params) -> PYUObject:
            def _generate_grad_mask(p, params):
                mask = []
                for k, v in params.items():
                    submask = np.random.binomial(
                        n=1, p=1 - compression_ratio, size=v.shape
                    )
                    mask.append(submask)

                return mask

            return self.server(_generate_grad_mask)(compression_ratio, params)

        grad_mask = generate_grad_mask(
            self.compression_ratio, self.global_net.state_dict()
        )

        # send grad mask to all clients
        def receive_grad_mask(worker: BaseTorchModel, grad_mask, compression_ratio):
            worker.grad_mask = grad_mask
            worker.compression_ratio = compression_ratio
            return

        for device, worker in self._workers.items():
            wait(
                worker.apply(
                    receive_grad_mask, grad_mask.to(device), self.compression_ratio
                )
            )

        return
