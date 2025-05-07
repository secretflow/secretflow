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

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np


class TrainDataset(IterableDataset):
    # 训练数据集类
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = cfg.gpu_num

    # 将新闻 ID 转换为新闻索引
    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    # 填充到固定长度
    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    # 数据行映射
    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # 点击新闻处理
        clicked_index, clicked_mask = self.pad_to_fix_len(
            self.trans_to_nindex(click_id), self.cfg.model.his_size
        )
        clicked_input = self.news_input[clicked_index]

        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        return clicked_input, clicked_mask, candidate_input, label

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class TrainGraphDataset(TrainDataset):
    # 图神经网络训练数据集类，继承自 TrainDataset
    def __init__(
        self,
        filename,
        news_index,
        news_input,
        local_rank,
        cfg,
        neighbor_dict,
        news_graph,
        entity_neighbors,
    ):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)

        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors

    def line_mapper(self, line, sum_num_news):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size :]  # 取最近的历史新闻
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # ------------------ 点击新闻 ----------------------
        # ------------------ 新闻子图 ---------------------
        top_k = len(click_id)
        click_idx = self.trans_to_nindex(click_id)  # 将点击的新闻 ID 转换为索引
        source_idx = click_idx
        # 根据跳数构建子图
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(
                    self.neighbor_dict[news_idx][: self.cfg.model.num_neighbors]
                )
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        sub_news_graph, mapping_idx = self.build_subgraph(
            click_idx, top_k, sum_num_news
        )
        padded_maping_idx = F.pad(
            mapping_idx, (self.cfg.model.his_size - len(mapping_idx), 0), "constant", -1
        )

        # ------------------ 候选新闻 ---------------------
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        # ------------------ 实体子图 --------------------
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[
                :, -3 - self.cfg.model.entity_size : -3
            ]  # [5, 5]
            candidate_neighbor_entity = np.zeros(
                (
                    (self.cfg.npratio + 1) * self.cfg.model.entity_size,
                    self.cfg.model.entity_neighbors,
                ),
                dtype=np.int64,
            )  # [5*5, 20]
            # 获取实体邻居
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][
                    :valid_len
                ]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(
                self.cfg.npratio + 1,
                self.cfg.model.entity_size * self.cfg.model.entity_neighbors,
            )  # [5, 5*20]
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate(
                (origin_entity, candidate_neighbor_entity), axis=-1
            )
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        return (
            sub_news_graph,
            padded_maping_idx,
            candidate_input,
            candidate_entity,
            entity_mask,
            label,
            sum_num_news + sub_news_graph.num_nodes,
        )

    # 构建新闻子图
    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset:
            subset = [0]

        subset = torch.tensor(subset, dtype=torch.long, device=device)

        unique_subset, unique_mapping = torch.unique(
            subset, sorted=True, return_inverse=True
        )
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(
            unique_subset,
            self.news_graph.edge_index,
            self.news_graph.edge_attr,
            relabel_nodes=True,
            num_nodes=self.news_graph.num_nodes,
        )

        sub_news_graph = Data(
            x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr
        )

        return sub_news_graph, unique_mapping[:k] + sum_num_nodes

    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:
                    (
                        sub_newsgraph,
                        padded_mapping_idx,
                        candidate_input,
                        candidate_entity,
                        entity_mask,
                        label,
                        sum_num_news,
                    ) = self.line_mapper(line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)

                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))

                    # 当达到 batch_size 时，返回一个 batch
                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)

                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)

                        labels = torch.tensor(labels, dtype=torch.long)
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                        (
                            clicked_graphs,
                            mappings,
                            candidates,
                            labels,
                            candidate_entity_list,
                            entity_mask_list,
                        ) = ([], [], [], [], [], [])
                        sum_num_news = 0

                # 处理剩余的数据
                if len(clicked_graphs) > 0:
                    batch = Batch.from_data_list(clicked_graphs)

                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                    f.seek(0)


class ValidGraphDataset(TrainGraphDataset):
    # 验证数据集类，继承自 TrainGraphDataset
    def __init__(
        self,
        filename,
        news_index,
        news_input,
        local_rank,
        cfg,
        neighbor_dict,
        news_graph,
        entity_neighbors,
        news_entity,
    ):
        super().__init__(
            filename,
            news_index,
            news_input,
            local_rank,
            cfg,
            neighbor_dict,
            news_graph,
            entity_neighbors,
        )
        self.news_graph.x = torch.from_numpy(self.news_input).to(
            local_rank, non_blocking=True
        )  # 将新闻输入转换为张量并移至指定设备
        self.news_entity = news_entity  # 实体信息

    # 数据行映射处理
    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[
            -self.cfg.model.his_size :
        ]  # 获取点击的新闻 ID（最近的历史新闻）

        click_idx = self.trans_to_nindex(click_id)  # 将点击的新闻 ID 转换为新闻索引
        clicked_entity = self.news_entity[click_idx]  # 获取点击新闻对应的实体
        source_idx = click_idx

        # 根据跳数构建子图
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(
                    self.neighbor_dict[news_idx][: self.cfg.model.num_neighbors]
                )  # 获取当前新闻的邻居节点
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        # 构建新闻子图
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

        # ------------------ 实体 --------------------
        labels = np.array(
            [int(i.split('-')[1]) for i in line[4].split()]
        )  # 获取标签信息
        candidate_index = self.trans_to_nindex(
            [i.split('-')[0] for i in line[4].split()]
        )  # 转换候选新闻 ID 为索引
        candidate_input = self.news_input[candidate_index]  # 获取候选新闻输入

        # ------------------ 实体子图 --------------------
        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]  # 获取候选新闻对应的实体
            candidate_neighbor_entity = np.zeros(
                (
                    len(candidate_index) * self.cfg.model.entity_size,
                    self.cfg.model.entity_neighbors,
                ),
                dtype=np.int64,
            )
            # 获取候选新闻的实体邻居
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue  # 如果实体 ID 为 0，则跳过
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue  # 如果该实体没有邻居，则跳过
                valid_len = min(
                    entity_dict_length, self.cfg.model.entity_neighbors
                )  # 取实体邻居的有效长度
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][
                    :valid_len
                ]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(
                len(candidate_index),
                self.cfg.model.entity_size * self.cfg.model.entity_neighbors,
            )  # 重塑为候选新闻实体邻居矩阵

            entity_mask = candidate_neighbor_entity.copy()  # 实体掩码
            entity_mask[entity_mask > 0] = 1  # 标记有效的实体邻居

            candidate_entity = np.concatenate(
                (origin_entity, candidate_neighbor_entity), axis=-1
            )  # 合并原始实体和邻居实体
        else:
            candidate_entity = np.zeros(1)  # 如果不使用实体，将实体设置为零
            entity_mask = np.zeros(1)  # 同样将实体掩码设置为零

        # 将子图数据转换为 Batch 数据
        batch = Batch.from_data_list([sub_news_graph])

        return (
            batch,
            mapping_idx,
            clicked_entity,
            candidate_input,
            candidate_entity,
            entity_mask,
            labels,
        )

    # 数据迭代器
    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:  # 确保当前行有点击新闻信息
                (
                    batch,
                    mapping_idx,
                    clicked_entity,
                    candidate_input,
                    candidate_entity,
                    entity_mask,
                    labels,
                ) = self.line_mapper(line)
            yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels


class NewsDataset(Dataset):
    # 新闻数据集类
    def __init__(self, data):
        self.data = data  # 存储数据

    # 获取指定索引的新闻数据
    def __getitem__(self, idx):
        return self.data[idx]

    # 获取数据集的长度
    def __len__(self):
        return self.data.shape[0]  # 返回数据的行数（即样本数）
