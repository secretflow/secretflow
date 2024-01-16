from typing import List

import numpy as np
from secretflow.device import PYU, DeviceObject, PYUObject
from secretflow.security.aggregation.aggregator import Aggregator


class QPLDPAggregator(Aggregator):

    """Aggregator based on Quantization-PSI-LDP.
    The computation will be performed in QPLDPtext.

    Examples:
      >>> # Alice and bob are both pyu instances.
      >>> aggregator = QPLDPAggregator(alice)
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
            results = []
            grad = []
            gradient = []
            if isinstance(data[0], (list, tuple)):
                client_num = len(data)
                # 梯度数据是一个list
                for i in range(client_num):
                    grad = []
                    for j in range(len(data[0])):
                        grad.append(np.around(data[i][j], decimals=5))  # 量化操作减小通信开销

                    gradient.append(grad)
                gradient = np.array(gradient, dtype=np.float32)

                # 设定PSI服务器，set值是PSI的计算结果。一个数组，里面元素为1的位置代表属于交集元素

                set = np.zeros((gradient[0][-2].shape), dtype=np.float32)
                for i in range(len(gradient[0][-2])):
                    for j in range(len(gradient[0][-2][0])):
                        for k in range(len(gradient) - 1):
                            if gradient[k][-2][i][j] == gradient[k + 1][-2][i][j]:
                                set[i][j] = 1
                            else:
                                break
                # 使用PSI得到交集并添加噪声(FL+Quantization+PSI+LDP)set里元素为零的位置的参数加噪

                for k in range(client_num):
                    for i in range(len(set)):
                        noise = np.random.normal(0, 0.9, size=(len(set), len(set[i])))
                        for j in range(len(set[i])):
                            if set[i][j] == 1:
                                noise[i][j] = 0
                    gradient[k][-2] += noise
                # (FL+Quantization+PSI+LDP)仅对模型倒数第二层进行扰动

                for elements in zip(*gradient):
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
