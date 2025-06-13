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

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pickle

from dataload.dataset import *  # 导入自定义数据集模块


def load_data(cfg, mode='train', model=None, local_rank=0):
    """
    加载训练、验证或测试数据。

    :param cfg: 配置文件
    :param mode: 数据模式，可选 'train', 'val', 'test'
    :param model: 模型对象，只有在验证或测试时需要
    :param local_rank: 当前进程的GPU编号
    :return: 返回DataLoader
    """

    # 根据模式选择数据路径
    data_dir = {
        "train": cfg.dataset.train_dir,
        "val": cfg.dataset.val_dir,
        "test": cfg.dataset.test_dir,
    }

    # ------------- 加载新闻数据 -------------
    # 载入新闻索引和新闻内容
    news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))

    # ------------- 加载行为数据 -------------
    if mode == 'train':
        # 训练模式下，加载行为数据
        target_file = (
            Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv"
        )

        if cfg.model.use_graph:
            # 如果使用图结构模型，加载新闻图数据
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            # 如果是无向图，转换成无向图
            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(
                    news_graph.edge_index, news_graph.edge_attr
                )
            print(f"[{mode}] News Graph Info: {news_graph}")

            # 加载新闻邻居字典
            news_neighbors_dict = pickle.load(
                open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb")
            )

            # 如果模型使用实体图，还需要加载实体邻居数据
            if cfg.model.use_entity:
                entity_neighbors = pickle.load(
                    open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb")
                )
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            # 创建训练图数据集
            dataset = TrainGraphDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
                neighbor_dict=news_neighbors_dict,
                news_graph=news_graph,
                entity_neighbors=entity_neighbors,
            )
            dataloader = DataLoader(dataset, batch_size=None)  # 创建DataLoader

        else:
            # 不使用图结构模型时，直接加载普通的训练数据集
            dataset = TrainDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
            )

            # 创建DataLoader，按GPU数进行批量划分
            dataloader = DataLoader(
                dataset, batch_size=int(cfg.batch_size / cfg.gpu_num), pin_memory=True
            )
        return dataloader
    elif mode in ['val', 'test']:
        # 转换新闻数据为嵌入
        news_dataset = NewsDataset(news_input)
        news_dataloader = DataLoader(
            news_dataset,
            batch_size=int(cfg.batch_size * cfg.gpu_num),
            num_workers=cfg.num_workers,
        )

        stacked_news = []
        with torch.no_grad():
            # 在验证和测试模式下，计算新闻的嵌入表示
            for news_batch in tqdm(
                news_dataloader,
                desc=f"[{local_rank}] Processing validation News Embedding",
            ):
                # 如果使用图模型，计算嵌入
                if cfg.model.use_graph:
                    batch_emb = (
                        model.module.client.local_news_encoder(
                            news_batch.long().unsqueeze(0).to(local_rank)
                        )
                        .squeeze(0)
                        .detach()
                    )
                else:
                    batch_emb = (
                        model.module.client.local_news_encoder(
                            news_batch.long().unsqueeze(0).to(local_rank)
                        )
                        .squeeze(0)
                        .detach()
                    )
                stacked_news.append(batch_emb)

        # 拼接所有新闻的嵌入表示
        news_emb = torch.cat(stacked_news, dim=0).cpu().numpy()

        if cfg.model.use_graph:
            # 如果使用图结构模型，加载图数据
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")
            news_neighbors_dict = pickle.load(
                open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb")
            )

            # 如果是无向图，转换成无向图
            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(
                    news_graph.edge_index, news_graph.edge_attr
                )
            print(f"[{mode}] News Graph Info: {news_graph}")

            # 如果使用实体图，加载实体邻居数据
            if cfg.model.use_entity:
                # entity_graph = torch.load(Path(data_dir[mode]) / "entity_graph.pt")
                entity_neighbors = pickle.load(
                    open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb")
                )
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            # 验证模式下，使用图数据集
            if mode == 'val':
                dataset = ValidGraphDataset(
                    filename=Path(data_dir[mode])
                    / f"behaviors_np{cfg.npratio}_{local_rank}.tsv",
                    news_index=news_index,
                    news_input=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    neighbor_dict=news_neighbors_dict,
                    news_graph=news_graph,
                    news_entity=news_input[:, -8:-3],
                    entity_neighbors=entity_neighbors,
                )

            # 创建DataLoader
            dataloader = DataLoader(dataset, batch_size=None)

        else:
            # 不使用图结构时，使用普通的验证数据集
            if mode == 'val':
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / f"behaviors_{local_rank}.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                )
            else:
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / f"behaviors.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                )

            # 创建DataLoader
            dataloader = DataLoader(
                dataset, batch_size=1, collate_fn=lambda b: collate_fn(b, local_rank)
            )
        return dataloader


def collate_fn(tuple_list, local_rank):
    """
    定义如何将批次中的样本合并为一个batch。

    :param tuple_list: 一个包含样本的列表，每个样本是一个元组
    :param local_rank: 当前进程的GPU编号
    :return: 合并后的batch数据
    """
    # 分别提取clicked_news、clicked_mask、candidate_news等数据
    clicked_news = [x[0] for x in tuple_list]
    clicked_mask = [x[1] for x in tuple_list]
    candidate_news = [x[2] for x in tuple_list]
    clicked_index = [x[3] for x in tuple_list]
    candidate_index = [x[4] for x in tuple_list]

    # 如果样本包含标签，则一并返回
    if len(tuple_list[0]) == 6:
        labels = [x[5] for x in tuple_list]
        return (
            clicked_news,
            clicked_mask,
            candidate_news,
            clicked_index,
            candidate_index,
            labels,
        )
    else:
        return (
            clicked_news,
            clicked_mask,
            candidate_news,
            clicked_index,
            candidate_index,
        )
