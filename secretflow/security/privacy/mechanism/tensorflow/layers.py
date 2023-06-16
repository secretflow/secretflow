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

import tensorflow as tf

from secretflow.security.privacy.accounting.budget_accountant import BudgetAccountant


class GaussianEmbeddingDP(BudgetAccountant, tf.keras.layers.Layer):
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
