import math
from typing import List

import numpy as np

from secretflow.device import PYU, DeviceObject, PYUObject
from secretflow.security.aggregation.aggregator import Aggregator


class LLDPAggregator(Aggregator):
    """Aggregator based on Layer-wise LDP.

    The computation will be performed in LLDPtext.

    Examples:
      >>> # Alice and bob are both pyu instances.
      >>> aggregator = LLDPAggregator(alice)
      >>> a = alice(lambda : np.random.rand(2, 5))()
      >>> b = bob(lambda : np.random.rand(2, 5))()
      >>> sum_a_b = aggregator.sum([a, b], axis=0)
      >>> # Get the result.
      >>> sf.reveal(sum_a_b)
      array([[0.5954927 , 0.9381409 , 0.99397117, 1.551537  , 0.32698634],
        [1.288345  , 1.1820003 , 1.1769378 , 0.7396539 , 1.215364  ]],
        dtype=float32)
      >>> average_a_b = aggregator.average([a, b], axis=0)

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

    def sum(self, data: List[DeviceObject], axis=None) -> PYUObject:
        """Sum of array elements over a given axis.

        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.mean`.

        Returns:
            a device object holds the sum.
        """

        assert data, "Data to aggregate should not be None or empty!"
        data = [d.to(self.device) for d in data]

        def _sum(*data, axis):
            if isinstance(data[0], (list, tuple)):
                return [np.sum(element, axis=axis) for element in zip(*data)]
            else:
                return np.sum(data, axis=axis)

        return self.device(_sum)(*data, axis=axis)

    def average(self, data: List[DeviceObject], axis=None, weights=None) -> PYUObject:
        """Compute the weighted average along the specified axis.

        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.average`.
            weights: optional. Same as the weights argument of :py:meth:`numpy.average`.

        Returns:
            a device object holds the weighted average.
        """
        assert data, "Data to aggregate should not be None or empty!"
        data = [d.to(self.device) for d in data]
        if isinstance(weights, (list, tuple)):
            weights = [
                w.to(self.device) if isinstance(w, DeviceObject) else w for w in weights
            ]

        def _average(*data, axis, weights):  # 打包成一个元组
            client_list = []

            def getSensitivity(w, max=0, min=0):
                for e in w:
                    if max < e.max():
                        max = e.max()
                    if min > e.min():
                        min = e.min()
                return max - min

            if isinstance(data[0], (list, tuple)):
                results = []
                client_num = len(data)
                for j in range(client_num):
                    delta = math.exp(-3)
                    # NOTE(junfeng): unused variable.
                    # epsilon = [80, 80, 40, 40, 30, 30]  # 卷积层不加噪，后三层加噪
                    data_list_l = data[j][:4]
                    data_list_r = data[j][4:]
                    # NOTE(junfeng): unused variable.
                    # sensitivity = getSensitivity(data_list_r)
                    sigma = math.sqrt(2 * math.log(1.25 / delta)) / 28

                    noise = np.random.normal(0, sigma, 1622986)
                    add_noise_data = [0] * 6
                    index = 0
                    pos = 0
                    # 逐层加噪
                    for d in data_list_r:
                        size = d.shape
                        d = d.reshape(-1)
                        add_noise_data[index] = (
                            d + noise[pos : pos + d.shape[0]]
                        ).reshape((size))
                        pos = pos + d.shape[0]
                        index += 1
                    data_list = data_list_l + add_noise_data
                    client_list.append(data_list)

                for elements in zip(*client_list):
                    avg = np.average(elements, axis=axis, weights=weights)
                    res_dtype = self._get_dtype(elements[0])
                    if res_dtype:
                        avg = avg.astype(res_dtype)
                    results.append(avg)
                return results
            else:
                res = np.average(data, axis=axis, weights=weights)
                res_dtype = self._get_dtype(data[0])
                return res.astype(res_dtype) if res_dtype else res

        return self.device(_average)(*data, axis=axis, weights=weights)
