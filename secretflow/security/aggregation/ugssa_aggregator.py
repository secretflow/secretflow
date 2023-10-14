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


import math
from typing import List

import numpy as np
import torch
from secretflow.device import PYU, DeviceObject, PYUObject
from secretflow.security.aggregation.aggregator import Aggregator


class UGSSAAggregator(Aggregator):
    """Unbaised Gradient Sampling Secure Aggregation (UGSSA) aggregator.

    The computation will be performed in random number with sampling and transforming.

    Warnings:
        UGSSAAggregator is for debugging purpose only.
        You should not use it in production.

    Examples:
      >>> # Alice and bob are both pyu instances.
      >>> aggregator = UGSSAAggregator(alice)
      >>> a = alice(lambda : np.random.rand(2, 5))()
      >>> b = bob(lambda : np.random.rand(2, 5))()
      >>> sum_a_b = aggregator.sum([a, b], axis=0, compress_ratio=0.9)
      >>> # Get the result.
      >>> sf.reveal(sum_a_b)
      array([[0.5954927 , 0.9381409 , 0.99397117, 1.551537  , 0.32698634],
        [1.288345  , 1.1820003 , 1.1769378 , 0.7396539 , 1.215364  ]],
        dtype=float32)
      >>> average_a_b = aggregator.average([a, b], axis=0, compress_ratio=0.9)
      >>> sf.reveal(average_a_b)
      array([[0.29774636, 0.46907046, 0.49698558, 0.7757685 , 0.16349317],
        [0.6441725 , 0.59100014, 0.5884689 , 0.36982694, 0.607682  ]],
        dtype=float32)

    """

    def __init__(self, device: PYU):
        assert isinstance(device, PYU), f"Accepts PYU only but got {type(device)}."
        self.device = device

    @staticmethod
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

    def _sampling(self, grad: torch.tensor, k: int):
        randv = torch.rand_like(grad)
        weight = (1 / randv - 1) * grad ** 2
        weight = torch.where(weight.isnan(), torch.zeros_like(weight), weight)

        sort, idx = weight.sort(descending=True)
        _, topk_idx = sort[:k], idx[:k]

        w_r = grad * topk_idx.bincount(minlength=grad.size(0))

        C = sort[k]

        w_e = w_r + C / w_r
        w_e = torch.where(w_e.isinf(), torch.zeros_like(grad), w_e)
        w_e = torch.where(w_e.isnan(), torch.zeros_like(grad), w_e)

        return w_e

    def _flatten_grads(self, data: List[DeviceObject]):
        grads = []
        for idx, elements in enumerate(data):
            temp_grads = np.array([0.0])
            for thing in elements:
                temp_grads = np.concatenate((temp_grads, thing.flatten()))
            grads.append(temp_grads[1:])
        return grads

    def _reshape_grads_to_network(self, avg: np.ndarray, net_shape: List[np.ndarray]):
        # -> network shape
        results, begin = [], 0
        for layer_shape in net_shape:
            num = math.prod(layer_shape)
            end = begin + num
            temp = avg[begin:end].reshape(layer_shape)
            results.append(temp)
            begin = end
        return results

    def _ugs(self, data, axis=None, weights=None, compress_ratio=0.9, op="avg"):
        # -> flatten gradient: (num_clients, grad)
        flatten_grads = self._flatten_grads(data)

        # -> get network architecture
        net_shape = []
        net_params_num = 0
        for thing in data[0]:
            net_shape.append(thing.shape)
            net_params_num += math.prod(thing.shape)

        # -> unbaised gradient sampling
        results = []
        for elements in flatten_grads:
            results.append(
                self._sampling(
                    torch.tensor(elements), math.ceil(net_params_num * compress_ratio)
                ).numpy()
            )

        # -> average or sum
        if op == "avg":
            result = np.average(results, axis=axis, weights=weights)
        elif op == "sum":
            result = np.sum(results, axis=axis)

        # -> reshape grads to network architecture
        results = self._reshape_grads_to_network(result, net_shape)
        return results

    def sum(self, data: List[DeviceObject], axis=None, compress_ratio=0.9) -> PYUObject:
        """Sum of array elements over a given axis.

        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.mean`.
            compress_ratio: compress ratio :py:math:'float'.

        Returns:
            a device object holds the sum.
        """
        assert data, "Data to aggregate should not be None or empty!"
        data = [d.to(self.device) for d in data]

        def _sum(*data, axis):
            results = self._ugs(
                data, axis=axis, compress_ratio=compress_ratio, op="sum"
            )
            return results

        return self.device(_sum)(*data, axis=axis)

    def average(
        self, data: List[DeviceObject], axis=None, weights=None, compress_ratio=0.9
    ) -> PYUObject:
        """Compute the weighted average after MinMaxSampling along the specified axis.
s
        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.average`.
            weights: optional. Same as the weights argument of :py:meth:`numpy.average`.
            compress_ratio: compress ratio :py:math:'float'.

        Returns:
            a device object holds the weighted average.
        """
        assert data, "Data to aggregate should not be None or empty!"
        data = [d.to(self.device) for d in data]
        if isinstance(weights, (list, tuple)):
            weights = [
                w.to(self.device) if isinstance(w, DeviceObject) else w for w in weights
            ]

        def _average(*data, axis, weights, compress_ratio):
            assert 0.0 <= compress_ratio <= 1.0

            results = self._ugs(
                data,
                axis=axis,
                weights=weights,
                compress_ratio=compress_ratio,
                op="avg",
            )

            return results

        return self.device(_average)(
            *data, axis=axis, weights=weights, compress_ratio=compress_ratio
        )
