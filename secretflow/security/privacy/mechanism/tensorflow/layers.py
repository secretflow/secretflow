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

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List

from secretflow.security.privacy.accounting.rdp_accountant import (
    get_rdp,
    get_privacy_spent_rdp,
)

from secretflow.security.privacy.accounting.gdp_accountant import (
    cal_mu_poisson,
    cal_mu_uniform,
    get_eps_from_mu,
)


class EmbeddingDP(tf.keras.layers.Layer, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def call(self, inputs):
        pass


class GaussianEmbeddingDP(EmbeddingDP):
    """Embedding differential privacy perturbation using gaussian noise"""

    def __init__(
        self,
        noise_multiplier: float,
        batch_size: int,
        num_samples: int,
        l2_norm_clip: float = 1.0,
        delta: float = None,
        is_secure_generator: bool = False,
    ) -> None:
        """
        Args:
            epnoise_multipliers: Epsilon for pure DP.
            batch_size: Batch size.
            num_samples: Number of all samples.
            l2_norm_clip: The clipping norm to apply to the embedding.
            is_secure_generator: whether use the secure generator to generate noise.
        """
        super().__init__()
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.delta = delta if delta is not None else min(1 / num_samples**2, 1e-5)
        self.is_secure_generator = is_secure_generator

    def call(self, inputs):
        """Add gaussion dp on embedding.

        Args:
            inputs: Embedding.
        """
        # clipping
        embed_flat = tf.keras.layers.Flatten()(inputs)
        norm_vec = tf.norm(embed_flat, ord=2, axis=-1)
        ones = tf.ones(shape=norm_vec.shape)
        max_v = tf.linalg.diag(
            1.0 / tf.math.maximum(norm_vec / self.l2_norm_clip, ones)
        )
        embed_flat_clipped = tf.linalg.matmul(max_v, embed_flat)
        embed_clipped = tf.reshape(embed_flat_clipped, inputs.shape)
        # add noise
        if self.is_secure_generator:
            import secretflow.security.privacy._lib.random as random

            noise = random.secure_normal_real(
                0, self.noise_multiplier * self.l2_norm_clip, size=inputs.shape
            )
        else:
            noise = tf.random.normal(
                inputs.shape, stddev=self.noise_multiplier * self.l2_norm_clip
            )

        return tf.add(embed_clipped, noise)

    def privacy_spent_rdp(self, step: int, orders: List = None):
        """Get accountant using RDP.

        Args:
            step: The current step of model training or prediction.
            orders: An array (or a scalar) of RDP orders.
        """

        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        q = self.batch_size / self.num_samples
        rdp = get_rdp(q, self.noise_multiplier, step, orders)
        eps, _, opt_order = get_privacy_spent_rdp(orders, rdp, target_delta=self.delta)
        return eps, self.delta, opt_order

    def privacy_spent_gdp(
        self,
        step: int,
        sampling_type: str,
    ):
        """Get accountant using GDP.

        Args:
            step: The current step of model training or prediction.
            sampling_type: Sampling type, which must be "poisson" or "uniform".
        """

        if sampling_type == 'poisson':
            mu_ideal = cal_mu_poisson(
                step, self.noise_multiplier, self.num_samples, self.batch_size
            )
        elif sampling_type == 'uniform':
            mu_ideal = cal_mu_uniform(
                step, self.noise_multiplier, self.num_samples, self.batch_size
            )
        else:
            raise ValueError('the sampling_type must be "poisson" or "uniform".')

        eps = get_eps_from_mu(mu_ideal, self.delta)
        return eps, self.delta


class LabelDP:
    """Label differential privacy perturbation"""

    def __init__(self, eps: float) -> None:
        """
        Args:
            eps: epsilon for pure DP.
        """
        self._eps = eps

    def __call__(self, inputs: np.ndarray):
        """Random Response. Except for binary classification, inputs only support onehot form.

        Args:
            inputs: the label.
        """
        if not np.sum((inputs == 0) + (inputs == 1)) == inputs.size:
            raise ValueError(
                'Except for binary classification, inputs only support onehot form.'
            )

        if inputs.ndim == 1:
            p_ori = np.exp(self._eps) / (np.exp(self._eps) + 1)
            choice_ori = np.random.binomial(1, p_ori, size=inputs.shape[0])
            outputs = np.abs(1 - choice_ori - inputs)
        elif inputs.ndim == 2:
            p_ori = np.exp(self._eps) / (np.exp(self._eps) + inputs.shape[-1] - 1)
            p_oth = (1 - p_ori) / (inputs.shape[-1] - 1)
            p_array = inputs * (p_ori - p_oth) + np.ones(inputs.shape) * p_oth
            index_rr = np.array(
                [
                    np.random.choice(inputs.shape[-1], p=p_array[i])
                    for i in range(inputs.shape[0])
                ]
            )
            outputs = np.eye(inputs.shape[-1])[index_rr]
        else:
            raise ValueError('the dim of inputs in LabelDP must be less than 2.')

        # TODO(@yushi): Support regression.
        return outputs

    def privacy_spent(self):
        return self._eps
