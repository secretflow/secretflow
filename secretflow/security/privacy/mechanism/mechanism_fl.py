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

from typing import List

import numpy as np
from secretflow.security.privacy.accounting.rdp_accountant import (
    get_privacy_spent_rdp,
    get_rdp,
)


class GaussianModelDP:
    """global model differential privacy perturbation using gaussian noise"""

    def __init__(
        self,
        noise_multiplier: float,
        num_clients: int,
        num_updates: int = None,
        l2_norm_clip: float = 1.0,
        delta: float = None,
        is_secure_generator: bool = False,
        is_clip_each_layer: bool = False,
    ) -> None:
        """
        Args:
            epnoise_multipliers: Epsilon for pure DP.
            num_clients: Number of all clients.
            num_updates:  Number of Clients that participate in the update.
            l2_norm_clip: The clipping norm to apply to the parameters or gradients.
            is_secure_generator: whether use the secure generator to generate noise.
            is_clip_prelayer: The 2norm of each layer is dynamically assigned.
        """
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.num_clients = num_clients
        self.num_updates = num_clients if num_updates is None else num_updates
        self.delta = delta if delta is not None else min(1 / num_clients**2, 1e-5)
        self.is_secure_generator = is_secure_generator
        self.is_clip_each_layer = is_clip_each_layer

    def __call__(self, inputs: List):
        """Add gaussion dp on model parameters or gradients.

        Args:
            inputs: Model parameters.
        """
        assert inputs, 'the inputs of GaussianModelDP should not be empty!'

        model_weights_noised = []
        gradient_norm_all = self.global_norm(inputs)

        if self.is_clip_each_layer:
            for i in range(len(inputs)):
                # clipping
                gradient_norm = self.global_norm([inputs[i]])
                gradient_clipped = inputs[i] * min(
                    1,
                    (self.l2_norm_clip / np.sqrt(gradient_norm * gradient_norm_all)),
                )

                # add noise
                if self.is_secure_generator:
                    import secretflow.security.privacy._lib.random as random

                    noise = random.secure_normal_real(
                        0,
                        self.noise_multiplier * self.l2_norm_clip,
                        size=inputs[i].shape,
                    )
                else:
                    noise = np.random.normal(
                        inputs[i].shape,
                        stddev=self.noise_multiplier * self.l2_norm_clip,
                    )

                model_weights_noised.append(
                    np.add(gradient_clipped, noise / np.sqrt(self.num_updates))
                )
        else:
            scale = min(1, self.l2_norm_clip / gradient_norm_all)
            for i in range(len(inputs)):
                # clipping
                gradient_clipped = inputs[i] * scale

                # add noise
                if self.is_secure_generator:
                    import secretflow.security.privacy._lib.random as random

                    noise = random.secure_normal_real(
                        0,
                        self.noise_multiplier * self.l2_norm_clip,
                        size=inputs[i].shape,
                    ).astype(np.float32)
                else:
                    noise = np.random.normal(
                        loc=0.0,
                        scale=self.noise_multiplier * self.l2_norm_clip,
                        size=inputs[i].shape,
                    ).astype(np.float32)

                model_weights_noised.append(
                    np.add(gradient_clipped, noise / np.sqrt(self.num_updates))
                )
        return model_weights_noised

    def global_norm(self, inputs):
        inputs = [np.linalg.norm(i) ** 2 for i in inputs]

        return np.sqrt(sum(inputs))

    def privacy_spent_rdp(self, step: int, orders: List = None):
        """Get accountant using RDP.

        Args:
            step: The dp current step of model training or prediction.
            orders: An array (or a scalar) of RDP orders.
        """

        if orders is None:
            # order \in [2,128] empirically
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            # optional value
            # orders = (
            #     [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
            #     + list(range(5, 64))
            #     + [128, 256, 512]
            # )

        q = self.num_updates / self.num_clients
        rdp = get_rdp(q, self.noise_multiplier, step, orders)
        eps, _, opt_order = get_privacy_spent_rdp(orders, rdp, target_delta=self.delta)
        return eps, self.delta, opt_order
