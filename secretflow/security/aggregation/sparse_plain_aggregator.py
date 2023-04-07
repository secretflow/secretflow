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

from secretflow.device import PYU, DeviceObject
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.utils.compressor import sparse_decode


class SparsePlainAggregator(PlainAggregator):
    """Plaintext sparse aggregator.

    The computation will be performed in (decoded) plaintext.

    Warnings:
        SparsePlainAggregator is for debugging purpose only.
        You should not use it in production.

    Examples:
      >>> # Alice and bob are both pyu instances.
      >>> aggregator = SparsePlainAggregator(alice)
      >>> a = alice(lambda : np.random.rand(2, 5))()
      >>> b = bob(lambda : np.random.rand(2, 5))()
      >>> sum_a_b = aggregator.sum([a, b], axis=0)
      >>> # Get the result.
      >>> sf.reveal(sum_a_b)
    """

    def __post_init__(self):
        assert isinstance(
            self.device, PYU
        ), f'Accepts PYU only but got {type(self.device)}.'

    def _zip_decode_data(self, data: List) -> List:
        import sparse as sp

        if isinstance(data[0][0], (sp._coo.core.COO, sp._compressed.compressed.GCXS)):
            decoded_data = [sparse_decode(data=element) for element in zip(*data)]
        elif isinstance(data[0][0], np.ndarray):
            decoded_data = zip(*data)
        else:
            assert (
                False
            ), 'Sparse encoding method not supporterd in SecurePlainAggregator'
        return decoded_data

    def _decode_data(self, data: List) -> List:
        import sparse as sp

        if isinstance(data[0][0], (sp._coo.core.COO, sp._compressed.compressed.GCXS)):
            decoded_data = sparse_decode(data=data)
        elif isinstance(data[0], np.ndarray):
            decoded_data = data
        else:
            assert (
                False
            ), 'Sparse encoding method not supporterd in SecurePlainAggregator'
        return decoded_data

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
                decoded_data = self._zip_decode_data(data)
                return [np.sum(element, axis=axis) for element in decoded_data]
            else:
                decoded_data = self._decode_data(data)
                return np.sum(decoded_data, axis=axis)

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
                decoded_data = self._zip_decode_data(data)
                return [
                    np.average(
                        element,
                        axis=axis,
                        weights=weights if weights is not None else None,
                    )
                    for element in decoded_data
                ]
            else:
                decoded_data = self._decode_data(data)
                return np.average(
                    decoded_data,
                    axis=axis,
                    weights=weights if weights is not None else None,
                )

        return self.device(_average, static_argnames='axis')(
            *data, axis=axis, weights=weights
        )
