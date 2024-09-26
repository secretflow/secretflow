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
import numpy as np

from secretflow.ml.nn.callbacks.callback import Callback
from secretflow import reveal

import torch


class CAFEFakeGradients(Callback):
    """
    The method is designed for against CAFE attack: https://arxiv.org/abs/2110.15122.
    Each local worker randomly generates gradients with the normal distribution N,
    sorts them and true gradients in descending order, computes the L2-norm distance
    to find the nearest fake gradient, pairs fake gradients with true gradients by
    the sorted order, and uploads the fake gradients to the server.
    """

    def __init__(
        self,
        exec_device='cpu',
        attack_party=None,
        noise_scale=1.1,
        tau=47,
        v=1000,
        **kwargs
    ):
        """
        Initialize the CAFEFakeGradients.

        Args:
            exec_device (str): The execution device. Default is 'cpu'.
            attack_party (PYU): The party that might be under attack.
            noise_scale (float): Scale for noise generation. Default is 1.1.
            tau (int): Threshold for distance. Default is 47.
            v (int): Number of fake gradients to generate. Default is 1000.
            **kwargs: Additional keyword arguments.
        """
        self.attack_party = attack_party
        self.exec_device = exec_device
        self.noise_scale = noise_scale
        self.tau = tau
        self.v = v
        super().__init__(**kwargs)

    def on_fuse_backward_end(self):
        def fake_gradient(worker, sigma, tau=1.1, M=1, v=10):
            """
            Generate fake gradients for a client.

            Args:
                worker: The worker object.
                sigma (float): Noise parameter.
                tau (float): Threshold for distance. Default is 1.1.
                M (int): Some parameter. Default is 1.
                v (int): Number of fake gradients to generate. Default is 10.

            Returns:
                list: List of fake gradients.
            """
            h_shape = worker._h.shape

            grad_outputs = torch.ones(h_shape).to(worker._h.device)
            real_grad = torch.autograd.grad(
                worker._h,
                worker.model_base.parameters(),
                grad_outputs=grad_outputs,
                retain_graph=True,
            )
            local_gradients = list(real_grad)
            grad_shape_list = [g.shape for g in local_gradients]

            flattened_grad = [g.flatten() for g in local_gradients]

            fake_gradients = []
            for _ in range(v):
                fake_grad = [torch.randn_like(g) * sigma**2 for g in flattened_grad]
                for i in range(len(fake_grad)):
                    fake_grad[i] = torch.sort(fake_grad[i], descending=True)[0]
                fake_gradients.append(fake_grad)
            sorted_indexes = [
                torch.sort(lg, descending=True)[1] for lg in flattened_grad
            ]

            cur = 0
            while cur < 3:
                cur += 1
                min_distance_gradient = None
                min_distance = float('inf')
                for fg in fake_gradients:
                    distance = torch.norm(
                        torch.stack(
                            [
                                torch.norm(fg[i] - flattened_grad[i])
                                for i in range(len(flattened_grad))
                            ]
                        )
                    )
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_gradient = fg

                if min_distance <= tau:
                    break
                else:

                    fake_gradients = []
                    for _ in range(v):
                        fake_grad = [
                            torch.randn_like(g) * sigma**2 for g in flattened_grad
                        ]
                        for i in range(len(fake_grad)):
                            fake_grad[i] = torch.sort(fake_grad[i], descending=True)[0]
                        fake_gradients.append(fake_grad)
            fake_grad = min_distance_gradient
            g = [torch.zeros_like(param) for param in flattened_grad]
            for i in range(len(flattened_grad)):
                l = 0
                for k in sorted_indexes[i]:
                    g[i][k] = torch.min(
                        min_distance_gradient[i][l],
                        torch.max(flattened_grad[i][k], -min_distance_gradient[i][l]),
                    )
                    l += 1
            fake_g = [
                tmp_g.reshape(_shape) for tmp_g, _shape in zip(g, grad_shape_list)
            ]
            return fake_g

        def update_fake_grad(worker, _fake_grad_list):
            """
            Update the fake gradients.

            Args:
                worker: The worker object.
                _fake_grad_list (list): List of fake gradients.
            """
            for fake_grad in _fake_grad_list:
                worker._callback_store['cafe_attack']['true_gradient'].append(fake_grad)

        fake_grad_list = []
        for key in self._workers.keys():
            if key != self.attack_party:
                fake_g = reveal(
                    self._workers[key].apply(fake_gradient, sigma=self.noise_scale)
                )
                fake_grad_list.append(fake_g)

        self._workers[self.attack_party].apply(update_fake_grad, fake_grad_list)
