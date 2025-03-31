from copy import deepcopy
from typing import List
import numpy as np
import torch
import sklearn.metrics.pairwise as smp
import hdbscan

from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.device import PYU, DeviceObject, PYUObject


class FlameAggregator(PlainAggregator):
    """
    FLAME 聚合器：data[i] 是完整模型 W_i，而不是差分 Δ_i。
    在 aggregator 内部维护 prev_global_model (G_{t-1})，并对 (W_i - G_{t-1}) 进行范数裁剪。
    """

    def __init__(self, device: PYU, prev_global_model=None, lamda=0.001):
        """
        :param device: 聚合所在的 PYU 设备
        :param prev_global_model: 上一个全局模型 G_{t-1}, 格式与 data[i] 相同 (list/tuple of arrays/tensors)
        :param lamda: 高斯噪声系数
        """
        super().__init__()
        assert isinstance(device, PYU), f'Accepts PYU only but got {type(device)}.'
        self.device = device
        self.lamda = lamda
        self.prev_global_model = prev_global_model  # 维护上一个全局模型

    def set_prev_global_model(self, prev_global_model):
        """
        允许外部更新 prev_global_model。
        """
        self.prev_global_model = prev_global_model

    def average(self, data: List[DeviceObject], axis=None, weights=None) -> PYUObject:
        """
        :param data: 每个客户端的完整模型 W_i (list/tuple of np.array/torch.Tensor)，而非差分。
        :param axis, weights: 继承自 PlainAggregator，可选。
        :return: 新的全局模型 (list of numpy arrays)，并在 aggregator 内部更新 self.prev_global_model。
        """
        assert self.prev_global_model is not None, "Must set prev_global_model before calling average()!"

        def _average(*data, axis, weights):
            num_clients = len(data)
            w_list = []  # 存储每个客户端的模型参数
            features = []  # 存储展平后的模型参数，用于计算余弦相似度

            # 1) 提取模型权重并计算特征向量
            for i in range(num_clients):
                w_list.append(data[i])  # data[i] 是完整模型 W_i
                flat_list = [arr.cpu().numpy().flatten() if isinstance(arr, torch.Tensor) else arr.flatten() for arr in
                             data[i]]
                features.append(np.concatenate(flat_list) if flat_list else np.zeros(1))

            features_array = np.stack(features, axis=0)
            cd = smp.cosine_distances(features_array)  # 计算客户端之间的余弦距离

            # 2) HDBSCAN 聚类，过滤异常客户端
            clusterer = hdbscan.HDBSCAN(min_cluster_size=int(num_clients / 2 + 1),
                                        min_samples=1,
                                        allow_single_cluster=True,
                                        metric='precomputed').fit(cd)
            cluster_labels = clusterer.labels_.tolist()

            # 3) 计算每个客户端的欧几里得距离 e_i = ||W_i - G_{t-1}||
            G_prev = self.prev_global_model  # 先前的全局模型

            e_list = []  # 存储每个客户端的 L2 范数
            for i in range(num_clients):
                diff_vector = np.concatenate([(arr_w.cpu().numpy() - arr_g.cpu().numpy()).flatten()
                                              if isinstance(arr_w, torch.Tensor) else (arr_w - arr_g).flatten()
                                              for arr_w, arr_g in zip(w_list[i], G_prev)])
                e_list.append(np.linalg.norm(diff_vector))  # 计算 L2 范数

            e_s = np.median(e_list)  # 取中位数作为阈值

            # 4) 进行范数裁剪
            w_clipped_list = []  # 存储裁剪后的模型参数
            w_clipped_weights = []  # 存储对应的权重

            for i in range(num_clients):
                if cluster_labels[i] == -1:
                    continue  # 跳过异常点
                scale = min(1.0, e_s / e_list[i])  # 计算缩放比例
                w_c = [(arr_g + (arr_w - arr_g) * scale)
                       if isinstance(arr_w, np.ndarray) else arr_g + (arr_w.cpu().numpy() - arr_g.cpu().numpy()) * scale
                       for arr_w, arr_g in zip(w_list[i], G_prev)]
                w_clipped_list.append(w_c)
                w_clipped_weights.append(weights[i] if weights is not None else 1.0)

            # 5) 如果所有客户端都被过滤，返回零模型
            if not w_clipped_list:
                return [np.zeros_like(arr_g.cpu().numpy()) if isinstance(arr_g, torch.Tensor) else np.zeros_like(arr_g)
                        for arr_g in G_prev]

            # 6) 计算加权平均
            aggregated = [np.average([w_clipped_list[c][layer_idx] for c in range(len(w_clipped_list))],
                                     axis=0, weights=w_clipped_weights)
                          for layer_idx in range(len(w_clipped_list[0]))]

            # 7) 添加噪声（高斯噪声，标准差 = lamda * e_s）
            final_agg = [arr + np.random.normal(loc=0.0, scale=self.lamda * e_s, size=arr.shape)
                         for arr in aggregated]

            self.prev_global_model = final_agg  # 更新全局模型
            return final_agg

        return self.device(_average)(*data, axis=axis, weights=weights)
