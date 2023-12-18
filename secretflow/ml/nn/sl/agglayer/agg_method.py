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
        inputs,
    ) -> List:
        if isinstance(gradients, tuple) and len(gradients) == 1:
            gradients = gradients[0]
        wrapped_forward = auto_grad(self.forward)
        g_backward = grad(
            wrapped_forward, argnums=tuple([a for a in range(parties_num)])
        )
        df_dx = g_backward(*inputs)
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

    def __init__(self, axis=0, weights=None) -> None:
        super().__init__()
        self.g_func = None
        self.axis = axis
        self.weights = weights

    def forward(
        self,
        *data,
    ):
        # Here we use jax to ensure that both PYU and SPU can be supported, and support autograd to generate backward funciton
        if isinstance(data, tuple) and len(data) == 1:
            data = list(data)
        self.inputs = data[0]
        if isinstance(data[0], (list, tuple)):
            agg_data = [
                jnp.average(
                    jnp.array(element),
                    axis=self.axis,
                    weights=jnp.array(self.weights)
                    if self.weights is not None
                    else None,
                )
                for element in zip(*data)
            ]
        else:
            agg_data = jnp.average(
                jnp.array(data),
                axis=self.axis,
                weights=jnp.array(self.weights) if self.weights is not None else None,
            )
        return agg_data


class Sum(AggMethod):
    """Built-in Sum Aggregation Method."""

    def __init__(self, axis=0) -> None:
        super().__init__()
        self.g_func = None
        self.axis = axis

    def forward(
        self,
        *data,
    ):
        # Here we use jax to ensure that both PYU and SPU can be supported, and support autograd to generate backward funciton
        if isinstance(data, tuple) and len(data) == 1:
            data = list(data)
        self.inputs = data[0]
        if isinstance(data[0], (list, tuple)):
            agg_data = [
                jnp.sum(
                    jnp.array(element),
                    axis=self.axis,
                )
                for element in zip(*data)
            ]
        else:
            agg_data = jnp.sum(
                jnp.array(data),
                axis=self.axis,
            )
        return agg_data


class Concat(AggMethod):
    """Built-in Sum Aggregation Method."""

    def __init__(self, axis=1) -> None:
        super().__init__()
        self.g_func = None
        self.axis = axis

    def forward(
        self,
        *data,
    ):
        # Here we use jax to ensure that both PYU and SPU can be supported, and support autograd to generate backward funciton
        if isinstance(data, tuple) and len(data) == 1:
            data = list(data)
        self.inputs = data[0]
        if isinstance(data[0], (list, tuple)):
            agg_data = [
                jnp.concatenate(
                    jnp.array(element),
                    axis=self.axis,
                )
                for element in zip(*data)
            ]
        else:
            jnp_array = [jnp.array(d) for d in data]
            agg_data = jnp.concatenate(
                jnp_array,
                axis=self.axis,
            )
        return agg_data

    # rewrite backward since gradients and dx shapes are not aligned
    def backward(self, *gradients, parties_num, inputs) -> List:
        if isinstance(gradients, tuple) and len(gradients) == 1:
            gradients = gradients[0]
        wrapped_forward = auto_grad(self.forward)
        g_backward = grad(
            wrapped_forward, argnums=tuple([a for a in range(parties_num)])
        )
        df_dx = g_backward(*inputs)
        gradients_list = []
        split_point = []
        start = 0
        for dx in df_dx:
            split_point.append(start + dx.shape[self.axis])
            start = start + dx.shape[self.axis]
        split_point.pop(-1)
        if isinstance(gradients, (tuple, list)):
            split_gradients = []
            for g in gradients:
                split_gradients.append(jnp.split(g, split_point, axis=self.axis))

            for item in zip(*split_gradients):
                gradients_list.append(list(item))

        else:
            gradients_list = list(jnp.split(gradients, split_point, axis=self.axis))

        return gradients_list
