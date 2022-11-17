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

import jax.numpy as jnp

from secretflow.device import SPU, DeviceObject, SPUObject
from secretflow.security.aggregation.aggregator import Aggregator


class SPUAggregator(Aggregator):
    """Aggregator based on SPU.

    The computation will be performed on the given SPU device.

    Examples:
      >>> # spu shall be a SPU device instance.
      >>> aggregator = SPUAggregator(spu)
      >>> # Alice and bob are both pyu instances.
      >>> a = alice(lambda : np.random.rand(2, 5))()
      >>> b = bob(lambda : np.random.rand(2, 5))()
      >>> sum_a_b = aggregator.sum([a, b], axis=0)
      >>> # Get the result.
      >>> sf.reveal(sum_a_b)
      array([[0.5954927 , 0.9381409 , 0.99397117, 1.551537  , 0.3269863 ],
        [1.288345  , 1.1820003 , 1.1769378 , 0.7396539 , 1.215364  ]],
        dtype=float32)
      >>> average_a_b = aggregator.average([a, b], axis=0)
      >>> sf.reveal(average_a_b)
      array([[0.29774636, 0.46907043, 0.49698558, 0.7757685 , 0.16349316],
        [0.6441725 , 0.5910001 , 0.5884689 , 0.3698269 , 0.607682  ]],
        dtype=float32)

    """

    def __init__(self, device: SPU):
        assert isinstance(device, SPU), f'Accepts SPU only but got {type(self.device)}.'
        self.device = device

    def sum(self, data: List[DeviceObject], axis=None) -> SPUObject:
        """Sum of array elements over a given axis.

        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.mean`.

        Returns:
            a device object holds the sum.
        """
        assert data, 'Data to aggregate should not be None or empty!'
        data = [d.to(self.device) for d in data]

        def _sum(*data, axis):
            if isinstance(data[0], (list, tuple)):
                return [
                    jnp.sum(jnp.array(element), axis=axis) for element in zip(*data)
                ]
            else:
                return jnp.sum(jnp.array(data), axis=axis)

        return self.device(_sum, static_argnames='axis')(*data, axis=axis)

    def average(self, data: List[DeviceObject], axis=None, weights=None) -> SPUObject:
        """Compute the weighted average along the specified axis.

        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.average`.
            weights: optional. Same as the weights argument of :py:meth:`numpy.average`.

        Returns:
            a device object holds the weighted average.
        """
        assert data, 'Data to aggregate should not be None or empty!'
        data = [d.to(self.device) for d in data]
        if isinstance(weights, (list, tuple)):
            weights = [
                w.to(self.device) if isinstance(w, DeviceObject) else w for w in weights
            ]

        def _average(*data, axis, weights):
            if isinstance(data[0], (list, tuple)):
                return [
                    jnp.average(
                        jnp.array(element),
                        axis=axis,
                        weights=jnp.array(weights) if weights is not None else None,
                    )
                    for element in zip(*data)
                ]
            else:
                return jnp.average(
                    jnp.array(data),
                    axis=axis,
                    weights=jnp.array(weights) if weights is not None else None,
                )

        return self.device(_average, static_argnames='axis')(
            *data, axis=axis, weights=weights
        )
