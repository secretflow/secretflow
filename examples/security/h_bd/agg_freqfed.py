#!/usr/bin/env python3
# *_* coding: utf-8 *_*

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
    对输入的二维权重张量，进行DCT变换，返回论文当中对应的频域二维矩阵V
    V当中的低频成分是矩阵的左上角
    """
    weight_tensor_cpu = weight_tensor
    # 对每一行进行DCT
    dct_rows = torch.tensor(
        [dct(row, type=2, norm='ortho') for row in weight_tensor_cpu]
    )
    # 对每一列进行DCT
    dct_matrix = torch.tensor(
        [dct(col.detach().numpy(), type=2, norm='ortho') for col in dct_rows.T]
    ).T

    # 得到的dct矩阵，左上角的是低频成分
    return dct_matrix


def filtering(V):
    """
    对频域矩阵V，筛选出左上角元素
    V的行数为M，列数为N
    V_{i}{j} 满足i<=M/2,j<=N/2,且i+j<=(M/2+N/2)/2

    """
    # 按照freqfed当中选取左上角：
    # 暂时取
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
    基于余弦相似度和HDBSCAN的聚类算法（确保至少生成两个簇）。

    参数:
    F_list: 一个包含特征向量的列表 [F1, F2, ..., Fk]，其中每个Fi是一个PyTorch张量。

    返回:
    B: 最大簇的索引集合。
    """
    K = len(F_list)

    # 步骤1：初始化距离矩阵
    distances_matrix = torch.zeros((K, K), dtype=torch.float64)  # 使用float64

    # 步骤2：计算距离矩阵
    for i in range(K):
        for j in range(K):
            # 1 - 余弦相似度
            distances_matrix[i, j] = 1 - torch.nn.functional.cosine_similarity(
                F_list[i].unsqueeze(0), F_list[j].unsqueeze(0)
            )
            # 保证对称性
            distances_matrix[j, i] = distances_matrix[i, j]

    # 打印距离矩阵查看
    print("距离矩阵：")
    print(distances_matrix.numpy())

    # 将距离矩阵转换为NumPy格式，供HDBSCAN使用
    distances_matrix_np = distances_matrix.numpy()

    # 步骤3：使用HDBSCAN进行聚类（设定最小簇大小，确保至少两个簇）
    clusterer = HDBSCAN(
        metric="precomputed", min_cluster_size=2, min_samples=1
    )  # 至少有一个邻居
    # 如果backdoored nets太少了，可能全部离群，不行

    cluster_ids = clusterer.fit_predict(distances_matrix_np)

    # 步骤4：找到最大簇
    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)

    # 如果没有找到任何簇，返回空集合
    if len(unique_clusters) == 0:
        return set()

    # 找到最大簇的标号
    max_cluster = unique_clusters[np.argmax(counts)]

    # 步骤5：筛选出最大簇的索引
    B = set()  # 用集合存储结果
    for i in range(K):
        if cluster_ids[i] == max_cluster:
            B.add(i)

    # 返回最大簇的ids
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
