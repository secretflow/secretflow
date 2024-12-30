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
import torch
import logging
import hdbscan
import sklearn.metrics.pairwise as smp
from secretflow.device import PYU, DeviceObject, PYUObject
from secretflow.security.aggregation.aggregator import Aggregator

logger = logging.getLogger('logger')

class FLAME(Aggregator):
    """FLAME aggregator with defenses against malicious participants."""

    def __init__(self, device: PYU, lamda: float = 0.001):
        assert isinstance(device, PYU), f'Accepts PYU only but got {type(device)}.'
        self.device = device
        self.lamda = lamda

    def _get_dtype(self, arr):
        if isinstance(arr, np.ndarray):
            return arr.dtype
        elif isinstance(arr, torch.Tensor):
            return arr.numpy().dtype
        else:
            return None

    def sum(self, data, axis=None):
        """Sum of array elements over a given axis."""
        assert data, 'Data to aggregate should not be None or empty!'
        data = [d.to(self.server) for d in data]

        def _sum(*data, axis):
            import numpy as np
            if isinstance(data[0], (list, tuple)):
                return [np.sum(element, axis=axis) for element in zip(*data)]
            else:
                return np.sum(data, axis=axis)

        return self.server(_sum)(*data, axis=axis)

    def _defense_average(self, data: List[DeviceObject], weights=None) -> PYUObject:
        """
        Perform FLAME aggregation with clustering and defense.

        Args:
            data: List of device objects representing model updates.
            weights: Optional weights for averaging.

        Returns:
            A device object holding the aggregated result.
        """
        assert data, 'Data to aggregate should not be None or empty!'
        data = [d.to(self.device) for d in data]

        def _flame_aggregation(*data):
            local_params = []
            ed = []

            for d in data:
                if isinstance(d, (np.ndarray, torch.Tensor)):
                    d = d.flatten() if len(d.shape) > 1 else d
                    d = d.cpu().numpy() if isinstance(d, torch.Tensor) else d
                elif isinstance(d, list):
                    d = np.concatenate([np.ravel(x) for x in d], axis=0)
                else:
                    raise ValueError(f"Unsupported data type: {type(d)}")

                # Convert to float64 for compatibility with HDBSCAN
                d = d.astype(np.float64)
                local_params.append(d)
                ed.append(np.linalg.norm(d))

            # Ensure all parameters have the same length
            target_length = len(local_params[0])
            for i, param in enumerate(local_params):
                if len(param) != target_length:
                    raise ValueError(
                        f"Parameter at index {i} has length {len(param)}; expected {target_length}"
                    )

            # Compute pairwise cosine distances
            cd = smp.cosine_distances(np.stack(local_params))

            # Convert cosine distance matrix to float64
            cd = cd.astype(np.float64)

            # HDBSCAN clustering
            cluster = hdbscan.HDBSCAN(
                min_cluster_size=len(data) // 2 + 1,
                min_samples=1,
                allow_single_cluster=True,
                metric="precomputed",
            ).fit(cd)
            cluster_labels = cluster.labels_.tolist()

            # Norm clipping and noise addition
            st = np.median(ed)

            # Ensure lamda is a float
            if not isinstance(self.lamda, (int, float, np.float64)):
                raise ValueError(f"self.lamda must be a scalar, but got {type(self.lamda)}")
            self.lamda = float(self.lamda)

            # Ensure st is a scalar
            if not np.isscalar(st):
                raise ValueError(f"st must be a scalar, but got {type(st)} with value {st}")
            st = float(st)

            clipped_updates = []
            for i, d in enumerate(local_params):
                if cluster_labels[i] == -1:
                    continue  # Skip malicious updates
                scale_factor = min(1.0, st / ed[i])
                clipped_updates.append(d * scale_factor)

            # Aggregate clipped updates
            aggregated_update = np.mean(clipped_updates, axis=0)

            # Add noise
            noise = np.random.normal(0, self.lamda * st, aggregated_update.shape)
            aggregated_update += noise

            return aggregated_update

        return self.device(_flame_aggregation)(*data)

    def average(self, data: List[DeviceObject], axis=None, weights=None) -> PYUObject:
        """
        Override the default average method with FLAME defense.

        Args:
            data: List of device objects.
            axis: Axis for averaging (not used in FLAME defense).
            weights: Weights for averaging (not used in FLAME defense).

        Returns:
            A device object holding the aggregated result.
        """
        return self._defense_average(data, weights=weights)
