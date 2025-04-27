# Copyright 2025 Ant Group Co., Ltd.
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


import numpy as np
from ray import logger
import torch
from hdbscan import HDBSCAN
from scipy.fftpack import dct
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.device import PYU, DeviceObject, PYUObject
from typing import List


def tensorDCT(weight_tensor):
    """
    Perform DCT transformation on the input two-dimensional weight tensor and return the corresponding frequency domain two-dimensional matrix V in the paper.
    The low-frequency component in V is the upper left corner of the matrix
    """
    weight_tensor_cpu = weight_tensor
    # Perform DCT on each row
    dct_rows = torch.tensor(
        [dct(row, type=2, norm='ortho') for row in weight_tensor_cpu]
    )
    # Perform DCT on each column
    dct_matrix = torch.tensor(
        [dct(col.detach().numpy(), type=2, norm='ortho') for col in dct_rows.T]
    ).T
    # The obtained DCT matrix has low-frequency components in the upper left corner
    return dct_matrix


def filtering(V):
    """
    For the frequency domain matrix V, filter out the upper left corner element
    V has M rows and N columns
    V_{i}{j} satisfy i<=M/2,j<=N/2, and i+j<=(M/2+N/2)/2

    """

    F = []
    height, width = V.shape
    subV = V[: int(height / 2), : int(width / 2)]
    for i in range(int(height / 2)):
        for j in range(int(width / 2)):
            if i + j <= int((height + width) / 4):
                F.append(subV[i][j])

    F_tensor = torch.tensor(F)
    return F_tensor


def clustering(F_list):
    """
    Clustering algorithm based on cosine similarity and HDBSCAN (ensuring at least two clusters are generated).

    args:
    F_list: a list of features [F1, F2, ..., Fk]

    return:
    B: The member index set of the largest cluster
    """
    K = len(F_list)

    # Step 1: Initialize the distance matrix
    distances_matrix = torch.zeros((K, K), dtype=torch.float64)  # 使用float64

    # Step 2: Calculate the distance matrix
    for i in range(K):
        for j in range(K):
            # 1 - Cosine similarity
            distances_matrix[i, j] = 1 - torch.nn.functional.cosine_similarity(
                F_list[i].unsqueeze(0), F_list[j].unsqueeze(0), eps=1e-5
            )
            # Ensure symmetry
            distances_matrix[j, i] = distances_matrix[i, j]

    # Convert the distance matrix to NumPy format for use by HDBSCAN
    distances_matrix_np = distances_matrix.numpy()
    # Step 3: Clustering using HDBSCAN (setting the minimum cluster size to ensure at least two clusters)
    clusterer = HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1)
    # At least one neighbor

    cluster_ids = clusterer.fit_predict(distances_matrix_np)

    # Step 4: Find the largest cluster
    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)

    # If no clusters are found, an empty set is returned
    if len(unique_clusters) == 0:
        return set()

    # Find the label of the largest cluster
    max_cluster = unique_clusters[np.argmax(counts)]

    # Step 5: Filter out the index of the largest cluster
    B = set()
    for i in range(K):
        if cluster_ids[i] == max_cluster:
            B.add(i)

    # Returns the ids of the largest cluster
    return B, cluster_ids


class FreqAggregator(PlainAggregator):

    def __init__(self, device: PYU):
        assert isinstance(device, PYU), f'Accepts PYU only but got {type(device)}.'
        self.device = device

    def average(self, data: List[DeviceObject], axis=None, weights=None) -> PYUObject:
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
            if weights is not None:
                F_list = []
                new_sample_num_list = []
                for client_param in data:
                    F = filtering(tensorDCT(client_param[-2]))
                    F_list.append(F)

                ids_no_backdoored, cluster_ids = clustering(F_list=F_list)

                client_param_for_aggregation = []
                for i in ids_no_backdoored:
                    client_param_for_aggregation.append(data[i])
                    new_sample_num_list.append(weights[i])
                print("Nets for aggregation(no backdoor):" + str(ids_no_backdoored))
                print("Nets clusters:" + str(cluster_ids))
                data = client_param_for_aggregation
                weights = new_sample_num_list

            if isinstance(data[0], (list, tuple)):
                results = []
                for elements in zip(*data):
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
