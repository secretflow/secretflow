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

from dataclasses import dataclass
from typing import List, Union

import jax.numpy as jnp

from secretflow.device import SPU, PYU, DeviceObject
from secretflow.security.aggregation.aggregator import Aggregator


@dataclass
class DeviceAggregator(Aggregator):
    """
    Aggregator based on a device (PYU or SPU).

    Attributes:
        device: a PYU or SPU. The device where the computation hosts.
    """

    device: Union[PYU, SPU]

    def sum(self, data: List[DeviceObject], axis=None) -> DeviceObject:
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

    def average(
        self, data: List[DeviceObject], axis=None, weights=None
    ) -> DeviceObject:
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
