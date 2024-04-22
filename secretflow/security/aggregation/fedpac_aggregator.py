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
import logging
import numpy as np
import torch
import tensorflow as tf
import copy

# from copy import copy, deepcopy
import math


from secretflow.device import PYU, DeviceObject, PYUObject
from secretflow.security.aggregation.aggregator import Aggregator


class FedPACAggregator(Aggregator):
    """FedPAC aggregator.

    This class provides methods for aggregating data in a federated learning setting using the FedPAC algorithm.

    Examples:
      >>> # Alice and bob are both pyu instances.
      >>> aggregator = FedPACAggregator(alice)
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
        assert isinstance(device, PYU), f'Accepts PYU only but got {type(device)}.'
        self.device = device

    @staticmethod
    def _get_dtype(arr):
        if isinstance(arr, np.ndarray):
            return arr.dtype
        else:
            try:
                if isinstance(arr, tf.Tensor):
                    return arr.numpy().dtype
                elif isinstance(arr, torch.Tensor):
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
        assert data, 'Data to aggregate should not be None or empty!'
        data = [d.to(self.device) for d in data]

        def _sum(*data, axis):
            if isinstance(data[0], (list, tuple)):
                return [np.sum(element, axis=axis) for element in zip(*data)]
            else:
                return np.sum(data, axis=axis)

        return self.device(_sum)(*data, axis=axis)

    def average(self, data: List[DeviceObject], axis=None, weights=None):
        """
        Compute the weighted average along the specified axis.
        Aggregate feature extractor.

        Args:
            data: List of clients local model parameters. List[dict{layer: weight}]
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
                results = []
                for elements in zip(*data):
                    avg = np.average(elements, axis=axis, weights=weights)
                    res_dtype = self._get_dtype(elements[0])
                    if res_dtype:
                        avg = avg.astype(res_dtype)
                    results.append(avg)
                return results

            # fedpac
            elif isinstance(data[0], dict):
                logging.info('feature extraction aggregation')
                weights = torch.tensor(weights)
                weights = weights / (weights.sum(dim=0))

                res = copy.deepcopy(data[0])
                for key in res.keys():
                    res[key] = torch.zeros_like(res[key])
                    for i in range(len(data)):
                        res[key] += data[i][key] * weights[i]
                logging.info(f'res: {res}')
                return res

            else:
                res = np.average(data, axis=axis, weights=weights)
                res_dtype = self._get_dtype(data[0])
                logging.info(f'res: {res}')
                return res.astype(res_dtype) if res_dtype else res

        return self.device(_average)(*data, axis=axis, weights=weights)

    # fedpac
    def global_protos_agg(
        self, clients_protos_list: List, clients_label_sizes_list: List
    ):
        """Compute the weighted average along the specified axis.

        Args:
            local_protos: List of clients local protos. list[dict{label:proto}]
            clints_sizes: List of clients sizes. list[dict{label:size}]

        Returns:
            a device object holds the weighted average.
        """
        assert clients_protos_list, 'Data to aggregate should not be None or empty!'

        agg_protos_label = {}
        agg_sizes_labels = {}
        for i in range(len(clients_protos_list)):
            local_protos = clients_protos_list[i].data
            local_label_sizes = clients_label_sizes_list[i].data
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                    agg_sizes_labels[label].append(local_label_sizes[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
                    agg_sizes_labels[label] = [local_label_sizes[label]]

        for [label, proto_list] in agg_protos_label.items():
            sizes_list = agg_sizes_labels[label]
            proto = 0 * proto_list[0]
            for i in range(len(proto_list)):
                proto += proto_list[i] * sizes_list[i]
            agg_protos_label[label] = proto / sum(sizes_list)

        return agg_protos_label

    # fedpac
    def classifier_weighted_aggregation(
        self, clients_param_list: List, cls_weight_list: List, keys, client_idx: int
    ):
        """Compute the weighted average along the specified axis.

        Args:
            clients_param_list: List of clients local model parameters. List[dict{layer: weight}]
            cls_weight_list: List of clients sizes. list[dict{label:size}]
            keys: keys of the classifier weights
            client_idx: index of the client

        Returns:
            a device object holds the weighted average.
        """
        assert clients_param_list, 'Data to aggregate should not be None or empty!'
        assert cls_weight_list, 'Data to aggregate should not be None or empty!'

        num_users = len(clients_param_list)
        w_0 = copy.deepcopy(clients_param_list[client_idx])
        for key in keys:
            w_0[key] = torch.zeros_like(w_0[key])
        for i in range(num_users):
            for key in keys:
                w_0[key] += cls_weight_list[i] * clients_param_list[i][key]

        wc = sum(cls_weight_list)
        for key in keys:
            w_0[key] = torch.div(w_0[key], wc)
        return w_0
