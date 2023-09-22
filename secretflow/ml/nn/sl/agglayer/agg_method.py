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

"""AggLayer method

"""
from abc import ABC, abstractmethod
from typing import List

import jax.numpy as jnp
from jax import grad


def auto_grad(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        return jnp.sum(jnp.array(ret))

    return wrapper


class AggMethod(ABC):
    """Base class of aggregation method for AggLayer
    The object of subclass will be passed into AggLayer for use.
    custum aggmethod should inherit this class, and override 'forward' and 'backward'
    Since aggmethod can support secure computation and plaintext computation, please use jax-numpy for development
    """

    def __init__(self) -> None:
        self.inputs = None

    @abstractmethod
    def forward(self, data: List, **kwargs) -> jnp.ndarray:
        """
        define how to merge data from parties
        """
        pass

    # Rewrite backward if need to customize the backward logic
    def backward(
        self,
        *gradients,
        parties_num,
        weights,
        inputs,
    ) -> List:
        if isinstance(gradients, tuple) and len(gradients) == 1:
            gradients = gradients[0]
        wrapped_forward = auto_grad(self.forward)
        g_backward = grad(
            wrapped_forward, argnums=tuple([a for a in range(parties_num)])
        )
        df_dx = g_backward(*inputs, weights=weights)
        gradients_list = []
        if isinstance(gradients, (tuple, list)):
            for dx in df_dx:
                party_gradients = []
                for g in gradients:
                    party_gradients.append(jnp.array(dx) * jnp.array(g))
                gradients_list.append(party_gradients)
        else:
            for dx in df_dx:
                gradients_list.append(jnp.array(dx) * jnp.array(gradients))
        return gradients_list


class Average(AggMethod):
    """Built-in Average Aggregation Method."""

    def __init__(self) -> None:
        super().__init__()
        self.g_func = None

    def forward(
        self,
        *data,
        axis=0,
        weights=None,
    ):
        # Here we use jax to ensure that both PYU and SPU can be supported, and support autograd to generate backward funciton
        if isinstance(data, tuple) and len(data) == 1:
            data = list(data)
        self.inputs = data[0]
        if isinstance(data[0], (list, tuple)):
            agg_data = [
                jnp.average(
                    jnp.array(element),
                    axis=axis,
                    weights=jnp.array(weights) if weights is not None else None,
                )
                for element in zip(*data)
            ]
        else:
            agg_data = jnp.average(
                jnp.array(data),
                axis=axis,
                weights=jnp.array(weights) if weights is not None else None,
            )
        return agg_data
