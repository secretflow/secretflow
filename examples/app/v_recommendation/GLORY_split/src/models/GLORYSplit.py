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
import torch.nn as nn
from torch_geometric.nn import Sequential, GatedGraphConv
from secretflow import PYUObject, proxy

from models.base.layers import *
from models.component.candidate_encoder import *
from models.component.click_encoder import ClickEncoder
from models.component.entity_encoder import EntityEncoder, GlobalEntityEncoder
from models.component.nce_loss import NCELoss
from models.component.news_encoder import *
from models.component.user_encoder import *


class GLORY(nn.Module):
    def __init__(self, cfg, glove_emb=None, entity_emb=None):
        super().__init__()

        self.cfg = cfg
        self.use_entity = cfg.model.use_entity  # 是否使用实体信息

        self.news_dim = cfg.model.head_num * cfg.model.head_dim  # 新闻的维度
        self.entity_dim = cfg.model.entity_emb_dim  # 实体的维度

        # -------------------------- 模型定义 --------------------------
        # 新闻编码器
        self.local_news_encoder = NewsEncoder(cfg, glove_emb)

        # GCN（图卷积网络）
        self.global_news_encoder = Sequential(
            'x, index',
            [
                (
                    GatedGraphConv(self.news_dim, num_layers=3, aggr='add'),
                    'x, index -> x',
                ),  # 采用 GatedGraphConv 进行图卷积
            ],
        )

        # 实体编码器
        if self.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()  # 使用预训练的实体嵌入
            self.entity_embedding_layer = nn.Embedding.from_pretrained(
                pretrain, freeze=False, padding_idx=0
            )  # 实体嵌入层

            # 局部实体编码器
            self.local_entity_encoder = Sequential(
                'x, mask',
                [
                    (self.entity_embedding_layer, 'x -> x'),
                    (
                        EntityEncoder(cfg),
                        'x, mask -> x',
                    ),  # 使用 EntityEncoder 进行实体编码
                ],
            )

            # 全局实体编码器
            self.global_entity_encoder = Sequential(
                'x, mask',
                [
                    (self.entity_embedding_layer, 'x -> x'),
                    (
                        GlobalEntityEncoder(cfg),
                        'x, mask -> x',
                    ),  # 使用 GlobalEntityEncoder 进行全局实体编码
                ],
            )

        # 点击编码器
        self.click_encoder = ClickEncoder(cfg)

        # 用户编码器
        self.user_encoder = UserEncoder(cfg)

        # 候选新闻编码器
        self.candidate_encoder = CandidateEncoder(cfg)

        # 点击预测器
        self.click_predictor = DotProduct()

        # NCE损失函数
        self.loss_fn = NCELoss()

        # 另一个损失函数
        self.loss_fn_ = NCELoss()

    def forward(
        self,
        subgraph,
        mapping_idx,
        candidate_news,
        candidate_entity,
        entity_mask,
        label=None,
    ):
        # -------------------------------------- 处理点击新闻 ----------------------------------
        mask = mapping_idx != -1  # 创建掩码，表示哪些新闻是有效的
        mapping_idx[mapping_idx == -1] = 0  # 将无效的新闻映射为0

        batch_size, num_clicked, token_dim = (
            mapping_idx.shape[0],
            mapping_idx.shape[1],
            candidate_news.shape[-1],
        )
        clicked_entity = subgraph.x[mapping_idx, -8:-3]  # 获取点击新闻的实体信息

        # 新闻编码器 + 图卷积网络（GCN）
        x_flatten = subgraph.x.view(1, -1, token_dim)  # 将新闻节点的特征展平
        x_encoded = self.local_news_encoder(x_flatten).view(
            -1, self.news_dim
        )  # 对新闻特征进行局部编码

        # 拆到服务器端的操作
        graph_emb = self.global_news_encoder(x_encoded, subgraph.edge_index)
        clicked_graph_emb = (
            graph_emb[mapping_idx, :]
            .masked_fill(~mask.unsqueeze(-1), 0)
            .view(batch_size, num_clicked, self.news_dim)
        )

        # 获取点击新闻的原始特征和图卷积后的特征
        clicked_origin_emb = (
            x_encoded[mapping_idx, :]
            .masked_fill(~mask.unsqueeze(-1), 0)
            .view(batch_size, num_clicked, self.news_dim)
        )

        # ------------------- 实体信息处理 -------------------
        if self.use_entity:
            clicked_entity = self.local_entity_encoder(
                clicked_entity, None
            )  # 对点击新闻的实体进行编码
        else:
            clicked_entity = None

        # 点击新闻的总嵌入（结合原始特征和图卷积特征）
        clicked_total_emb = self.click_encoder(
            clicked_origin_emb, clicked_graph_emb, clicked_entity
        )
        user_emb = self.user_encoder(clicked_total_emb, mask)  # 获取用户的嵌入

        # ----------------------------------------- 处理候选新闻 ------------------------------------
        # 对候选新闻进行编码
        cand_title_emb = self.local_news_encoder(candidate_news)  # 编码候选新闻标题
        if self.use_entity:
            # 分离候选新闻的原始实体和邻居实体
            origin_entity, neighbor_entity = candidate_entity.split(
                [
                    self.cfg.model.entity_size,
                    self.cfg.model.entity_size * self.cfg.model.entity_neighbors,
                ],
                dim=-1,
            )

            # 对候选新闻的实体进行编码
            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(
                neighbor_entity, entity_mask
            )

        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None

        # 将候选新闻的嵌入信息通过候选新闻编码器合并
        cand_final_emb = self.candidate_encoder(
            cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb
        )

        # ----------------------------------------- 计算分数 ------------------------------------
        score = self.click_predictor(
            cand_final_emb, user_emb
        )  # 计算候选新闻与用户的匹配度

        loss = self.loss_fn(score, label)  # 计算损失
        loss_ = self.loss_fn_(cand_final_emb, user_emb_)

        loss_all = loss + self.lambda_ * loss_
        # 年前主要改的这里哈
        return loss_all, score

    def validation_process(
        self,
        subgraph,
        mappings,
        clicked_entity,
        candidate_emb,
        candidate_entity,
        entity_mask,
    ):
        # 验证过程
        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        # 通过全局新闻编码器对新闻进行编码
        title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
        clicked_graph_emb = title_graph_emb[mappings, :].view(
            batch_size, num_news, news_dim
        )
        clicked_origin_emb = subgraph.x[mappings, :].view(
            batch_size, num_news, news_dim
        )

        # ------------------- Attention Pooling -------------------
        if self.use_entity:
            clicked_entity_emb = self.local_entity_encoder(
                clicked_entity.unsqueeze(0), None
            )
        else:
            clicked_entity_emb = None

        # 结合点击新闻的原始特征、图卷积特征和实体特征进行编码
        clicked_final_emb = self.click_encoder(
            clicked_origin_emb, clicked_graph_emb, clicked_entity_emb
        )

        # 获取用户的嵌入
        user_emb = self.user_encoder(clicked_final_emb)  # 用户的嵌入

        # ----------------------------------------- 处理候选新闻 ------------------------------------
        if self.use_entity:
            cand_entity_input = candidate_entity.unsqueeze(0)
            entity_mask = entity_mask.unsqueeze(0)
            origin_entity, neighbor_entity = cand_entity_input.split(
                [
                    self.cfg.model.entity_size,
                    self.cfg.model.entity_size * self.cfg.model.entity_neighbors,
                ],
                dim=-1,
            )

            # 对候选新闻的实体进行编码
            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(
                neighbor_entity, entity_mask
            )

        else:
            cand_origin_entity_emb = None
            cand_neighbor_entity_emb = None

        # 合并候选新闻的嵌入信息
        cand_final_emb = self.candidate_encoder(
            candidate_emb.unsqueeze(0), cand_origin_entity_emb, cand_neighbor_entity_emb
        )

        # ----------------------------------------- 计算分数 ------------------------------------
        scores = (
            self.click_predictor(cand_final_emb, user_emb).view(-1).cpu().tolist()
        )  # 计算候选新闻的分数

        return scores


###########################################################################################################################################################################


class GLORYServer(nn.Module):
    '''
    服务器只计算global gnn
    '''

    def __init__(
        self,
        cfg,
        glove_emb=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.news_dim = cfg.model.head_num * cfg.model.head_dim  # 新闻的维度

        # GCN（图卷积网络）
        self.global_news_encoder = Sequential(
            'x, index',
            [
                (
                    GatedGraphConv(self.news_dim, num_layers=3, aggr='add'),
                    'x, index -> x',
                ),  # 采用 GatedGraphConv 进行图卷积
            ],
        )

    def forward(self, x_encoded, edge_index, mapping_idx):
        # 服务器端计算图卷积特征
        graph_emb = self.global_news_encoder(x_encoded, edge_index)
        # clicked_graph_emb = graph_emb[mapping_idx, :]

        return graph_emb


class GLORYClient(nn.Module):
    '''
    客户端负责本地新闻的语义计算和后续全局图向量传入的计算
    '''

    def __init__(
        self,
        cfg,
        entity_emb,
        glove_emb=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.news_dim = cfg.model.head_num * cfg.model.head_dim  # 新闻的维度
        self.local_news_encoder = NewsEncoder(cfg, glove_emb)
        self.click_encoder = ClickEncoder(cfg)
        self.user_encoder = UserEncoder(cfg)
        self.candidate_encoder = CandidateEncoder(cfg)
        self.click_predictor = DotProduct()
        self.use_entity = cfg.model.use_entity
        self.entity_dim = cfg.model.entity_emb_dim  # 实体的维度
        self.loss_fn = NCELoss()
        # 实体编码器
        if self.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()  # 使用预训练的实体嵌入
            self.entity_embedding_layer = nn.Embedding.from_pretrained(
                pretrain, freeze=False, padding_idx=0
            )  # 实体嵌入层

            # 局部实体编码器
            self.local_entity_encoder = Sequential(
                'x, mask',
                [
                    (self.entity_embedding_layer, 'x -> x'),
                    (
                        EntityEncoder(cfg),
                        'x, mask -> x',
                    ),  # 使用 EntityEncoder 进行实体编码
                ],
            )
            self.global_entity_encoder = Sequential(
                'x, mask',
                [
                    (self.entity_embedding_layer, 'x -> x'),
                    (
                        GlobalEntityEncoder(cfg),
                        'x, mask -> x',
                    ),  # 使用 GlobalEntityEncoder 进行全局实体编码
                ],
            )

    def forward(
        self,
        subgraph,
        mapping_idx,
        candidate_news,
        candidate_entity,
        entity_mask,
        label=None,
    ):
        # ----------------- 计算客户端的 x_encoded -----------------
        x_encoded, mask, mapping_idx, batch_size, num_clicked, clicked_entity = (
            self.process_news(
                subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask
            )
        )

        # 服务器端返回 clicked_graph_emb
        # 假设服务器端已经计算好了 clicked_graph_emb
        clicked_graph_emb = self.get_clicked_graph_emb_from_server(
            x_encoded, subgraph.edge_index, mapping_idx
        )

        # ----------------- 合并特征 -----------------
        clicked_origin_emb = (
            x_encoded[mapping_idx, :]
            .masked_fill(~mask.unsqueeze(-1), 0)
            .view(batch_size, num_clicked, self.news_dim)
        )
        user_emb = self.combine_embeddings(
            clicked_origin_emb,
            clicked_graph_emb,
            clicked_entity,
            mask,
            batch_size,
            num_clicked,
        )

        # ------------计算候选新闻向量并且预测点击 -----------------
        score = self.predict_click(
            user_emb, candidate_news, candidate_entity, entity_mask
        )

        # ----------------- 计算损失 -----------------
        loss = self.loss_fn(score, label)

        return loss, score

    def combine_embeddings(
        self,
        clicked_origin_emb,
        clicked_graph_emb,
        clicked_entity,
        mask,
        batch_size,
        num_clicked,
    ):
        # 合并原始特征和图卷积特征
        clicked_total_emb = self.click_encoder(
            clicked_origin_emb, clicked_graph_emb, clicked_entity
        )
        user_emb = self.user_encoder(clicked_total_emb, mask)  # 用户嵌入
        return user_emb

    def predict_click(self, user_emb, candidate_news, candidate_entity, entity_mask):
        # 计算点击预测分数
        # --------------------------------------------1-----------------------------------------------------------
        # ----------------------------------------- 处理候选新闻 ------------------------------------
        # 对候选新闻进行编码
        cand_title_emb = self.local_news_encoder(candidate_news)  # 编码候选新闻标题
        if self.use_entity:
            # 分离候选新闻的原始实体和邻居实体
            origin_entity, neighbor_entity = candidate_entity.split(
                [
                    self.cfg.model.entity_size,
                    self.cfg.model.entity_size * self.cfg.model.entity_neighbors,
                ],
                dim=-1,
            )

            # 对候选新闻的实体进行编码
            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(
                neighbor_entity, entity_mask
            )

        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None

        # 将候选新闻的嵌入信息通过候选新闻编码器合并
        cand_final_emb = self.candidate_encoder(
            cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb
        )
        score = self.click_predictor(cand_final_emb, user_emb)
        return score

    def process_news(
        self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask
    ):
        # 处理新闻的编码过程
        mask = mapping_idx != -1
        mapping_idx[mapping_idx == -1] = 0
        batch_size, num_clicked, token_dim = (
            mapping_idx.shape[0],
            mapping_idx.shape[1],
            candidate_news.shape[-1],
        )
        clicked_entity = subgraph.x[mapping_idx, -8:-3]  # 获取点击新闻的实体信息
        x_flatten = subgraph.x.view(1, -1, token_dim)  # 展平
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)  # 编码
        if self.use_entity:
            clicked_entity = self.local_entity_encoder(
                clicked_entity, None
            )  # 对点击新闻的实体进行编码
        else:
            clicked_entity = None
        return x_encoded, mask, mapping_idx, batch_size, num_clicked, clicked_entity


class GLORYSplit(nn.Module):
    '''
    完整的模型
    '''

    def __init__(self, cfg, glove_emb=None, entity_emb=None):
        super().__init__()
        self.cfg = cfg
        self.client = GLORYClient(
            cfg,
            entity_emb,
            glove_emb,
        )
        self.server = GLORYServer(cfg, glove_emb)
        self.use_entity = cfg.model.use_entity  # 假设配置中有一个 `use_entity` 字段

    def forward(
        self,
        subgraph,
        mapping_idx,
        candidate_news,
        candidate_entity,
        entity_mask,
        label=None,
    ):
        # 客户端计算 x_encoded 和其他特征
        x_encoded, mask, mapping_idx, batch_size, num_clicked, clicked_entity = (
            self.client.process_news(
                subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask
            )
        )

        clicked_origin_emb = (
            x_encoded[mapping_idx, :]
            .masked_fill(~mask.unsqueeze(-1), 0)
            .view(batch_size, num_clicked, self.client.news_dim)
        )
        # 服务器端计算 clicked_graph_emb
        graph_emb = self.server(x_encoded, subgraph.edge_index, mapping_idx)

        clicked_graph_emb = (
            graph_emb[mapping_idx, :]
            .masked_fill(~mask.unsqueeze(-1), 0)
            .view(batch_size, num_clicked, self.client.news_dim)
        )

        # 客户端合并特征并进行点击预测

        user_emb = self.client.combine_embeddings(
            clicked_origin_emb,
            clicked_graph_emb,
            clicked_entity,
            mask,
            batch_size,
            num_clicked,
        )
        score = self.client.predict_click(
            user_emb, candidate_news, candidate_entity, entity_mask
        )

        # 计算损失
        loss = self.client.loss_fn(score, label)

        return loss, score

    def validation_process(
        self,
        subgraph,
        mappings,
        clicked_entity,
        candidate_emb,
        candidate_entity,
        entity_mask,
    ):
        # 验证过程
        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        # 通过全局新闻编码器对新闻进行编码
        title_graph_emb = self.server.global_news_encoder(
            subgraph.x, subgraph.edge_index
        )
        clicked_graph_emb = title_graph_emb[mappings, :].view(
            batch_size, num_news, news_dim
        )
        clicked_origin_emb = subgraph.x[mappings, :].view(
            batch_size, num_news, news_dim
        )

        # ------------------- Attention Pooling -------------------
        if self.use_entity:
            clicked_entity_emb = self.client.local_entity_encoder(
                clicked_entity.unsqueeze(0), None
            )
        else:
            clicked_entity_emb = None

        # 结合点击新闻的原始特征、图卷积特征和实体特征进行编码
        clicked_final_emb = self.client.click_encoder(
            clicked_origin_emb, clicked_graph_emb, clicked_entity_emb
        )

        # 获取用户的嵌入
        user_emb = self.client.user_encoder(clicked_final_emb)  # 用户的嵌入

        # ----------------------------------------- 处理候选新闻 ------------------------------------
        if self.use_entity:
            cand_entity_input = candidate_entity.unsqueeze(0)
            entity_mask = entity_mask.unsqueeze(0)
            origin_entity, neighbor_entity = cand_entity_input.split(
                [
                    self.cfg.model.entity_size,
                    self.cfg.model.entity_size * self.cfg.model.entity_neighbors,
                ],
                dim=-1,
            )

            # 对候选新闻的实体进行编码
            cand_origin_entity_emb = self.client.local_entity_encoder(
                origin_entity, None
            )
            cand_neighbor_entity_emb = self.client.global_entity_encoder(
                neighbor_entity, entity_mask
            )

        else:
            cand_origin_entity_emb = None
            cand_neighbor_entity_emb = None

        # 合并候选新闻的嵌入信息
        cand_final_emb = self.client.candidate_encoder(
            candidate_emb.unsqueeze(0), cand_origin_entity_emb, cand_neighbor_entity_emb
        )

        # ----------------------------------------- 计算分数 ------------------------------------
        scores = (
            self.client.click_predictor(cand_final_emb, user_emb)
            .view(-1)
            .cpu()
            .tolist()
        )  # 计算候选新闻的分数

        return scores
