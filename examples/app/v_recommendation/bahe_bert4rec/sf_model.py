# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from random import sample
import math
from transformers import AlbertModel, AlbertTokenizer


class EmbeddingLayer(nn.Module):
    """
    嵌入层：将用户和物品的 ID 映射为低维向量。
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, ids):
        return self.embedding(ids)


# class MultiHeadAttention(nn.Module):
#     """
#     多头自注意力机制。
#     """
#     def __init__(self, embed_dim, num_heads, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(embed_dim, embed_dim)

#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)

#         # 线性变换并分头
#         query = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         key = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         value = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

#         # 计算注意力分数
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)

#         # 计算注意力权重
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)


#         # 加权求和
#         attn_output = torch.matmul(attn_weights, value)
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
#         return self.out(attn_output)
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # 确保所有线性层使用相同的输入输出维度
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 打印维度信息用于调试
        print(
            f"Input shapes - Query: {query.shape}, Key: {key.shape}, Value: {value.shape}"
        )

        # 线性变换
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # 重塑为多头形式
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # 计算注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)

        return self.out(attn_output)


class TransformerBlock(nn.Module):
    """
    Transformer 模块：包含多头自注意力和前馈神经网络。
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 多头自注意力
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈神经网络
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x


class BERT4Rec(nn.Module):
    """
    BERT4Rec 模型：基于 Transformer 的序列推荐模型，支持多域推荐。
    """

    # def __init__(self, user_vocab_size, item_vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_len, num_domains=2, dropout=0.1):
    #     super(BERT4Rec, self).__init__()
    #     self.user_embedding = EmbeddingLayer(user_vocab_size, embed_dim)
    #     self.item_embedding = EmbeddingLayer(item_vocab_size, embed_dim)
    #     self.position_embedding = nn.Embedding(seq_len, embed_dim)
    #     self.domain_embedding = nn.Embedding(num_domains, embed_dim)  # 多域嵌入

    #     # Transformer 编码器
    #     self.transformer_blocks = nn.ModuleList([
    #         TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
    #     ])

    #     # 预测层
    #     self.predict_layer = nn.Linear(embed_dim, 1)
    def __init__(
        self,
        user_length,
        user_emb_dim,
        item_length,
        item_emb_dim,
        seq_len,
        hid_dim,
        bs,
        isInC,
        isItC,
        threshold1,
        threshold2,
    ):
        super(BERT4Rec, self).__init__()
        self.user_embedding = EmbeddingLayer(user_length, user_emb_dim)
        self.item_embedding = EmbeddingLayer(item_length, item_emb_dim)
        self.position_embedding = nn.Embedding(
            seq_len + 1, item_emb_dim
        )  # +1 确保序列长度足够
        self.domain_embedding = nn.Embedding(2, item_emb_dim)

        # Transformer编码器
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(item_emb_dim, num_heads=8, ff_dim=hid_dim)
                for _ in range(2)
            ]
        )

        self.predict_layer = nn.Linear(item_emb_dim, 1)
        self.isInC = isInC
        self.isItC = isItC
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def process_sequence(self, item_ids, seq, domain_ids, user_embedding):
        batch_size, seq_len = seq.size()

        # 获取嵌入
        item_emb = self.item_embedding(seq)  # [batch_size, seq_len, embed_dim]
        pos_emb = self.position_embedding(
            torch.arange(seq_len, device=seq.device)
        )  # [seq_len, embed_dim]
        domain_emb = self.domain_embedding(domain_ids)  # [batch_size, embed_dim]

        # 调整维度
        domain_emb = domain_emb.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch_size, seq_len, embed_dim]
        pos_emb = pos_emb.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, seq_len, embed_dim]

        # 融合嵌入
        seq_emb = item_emb + pos_emb + domain_emb

        # 如果有用户嵌入，也添加进去
        if user_embedding is not None:
            user_emb = user_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            seq_emb = seq_emb + user_emb

        # Transformer处理
        mask = (seq > 0).unsqueeze(1).unsqueeze(2)
        for block in self.transformer_blocks:
            seq_emb = block(seq_emb, mask)

        # 只取最后一个时间步的预测
        last_hidden = seq_emb[:, -1, :]  # [batch_size, embed_dim]

        # 预测当前item的概率
        item_ids_emb = self.item_embedding(item_ids)  # [batch_size, embed_dim]
        logits = torch.sum(last_hidden * item_ids_emb, dim=-1)  # [batch_size]
        return torch.sigmoid(logits)

    def forward(self, user_embedding, item_ids, seq_d1, seq_d2, domain_ids):
        output_d1 = self.process_sequence(
            item_ids, seq_d1, domain_ids, user_embedding
        )  # [batch_size]
        output_d2 = self.process_sequence(
            item_ids, seq_d2, domain_ids, user_embedding
        )  # [batch_size]

        # 根据 domain_ids 选择对应的输出
        selected_output = torch.where(
            domain_ids == 0, output_d1, output_d2
        )  # [batch_size]

        return selected_output


class AtomicBehaviorEmbedding(nn.Module):
    def __init__(self, albert_model_name='albert-base-v2'):
        super(AtomicBehaviorEmbedding, self).__init__()
        self.albert = AlbertModel.from_pretrained(albert_model_name)
        self.tokenizer = AlbertTokenizer.from_pretrained(albert_model_name)

    def forward(self, behavior_texts):
        # 将批次中的所有行为文本处理为单个列表
        all_texts = []
        batch_lengths = []
        for user_texts in behavior_texts:
            all_texts.extend(user_texts)
            batch_lengths.append(len(user_texts))

        # 分批处理文本以节省内存
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,  # 减小最大长度以节省内存
                return_tensors='pt',
            )

            with torch.no_grad():  # 使用no_grad减少内存使用
                outputs = self.albert(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
                all_embeddings.append(embeddings)

        # 连接所有嵌入
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # 重组为批次
        start_idx = 0
        batch_embeddings = []
        for length in batch_lengths:
            if length > 0:
                user_embeddings = all_embeddings[start_idx : start_idx + length].mean(
                    dim=0
                )
            else:
                # 对于没有行为的用户，使用零向量
                user_embeddings = torch.zeros(768, device=all_embeddings.device)
            batch_embeddings.append(user_embeddings)
            start_idx += length

        return torch.stack(batch_embeddings)  # [batch_size, 768]


class BehaviorInteractionModel(nn.Module):
    """
    行为交互模块：使用 Transformer 编码器学习行为间的交互。
    """

    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(BehaviorInteractionModel, self).__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, behavior_embeddings):
        """
        输入：原子行为嵌入表（形状：[batch_size, num_behaviors, embed_dim]）
        输出：用户嵌入（形状：[batch_size, embed_dim]）
        """
        for block in self.transformer_blocks:
            behavior_embeddings = block(behavior_embeddings)
        # 取最后一个时间步的输出作为用户嵌入
        user_embedding = behavior_embeddings[:, -1, :]
        return user_embedding


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=128):  # 改为128维度
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(
            embed_dim,
            embed_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, behavior_texts):
        embeddings = []
        for texts in behavior_texts:
            text_embeds = []
            for text in texts:
                words = text.split()
                word_ids = torch.tensor([hash(w) % 30000 for w in words])
                word_embeds = self.embedding(word_ids)
                text_embed = word_embeds.mean(0)
                text_embeds.append(text_embed)
            if text_embeds:
                user_embed = torch.stack(text_embeds).mean(0)
            else:
                user_embed = torch.zeros(128)  # 改为128维度
            embeddings.append(user_embed)
        return torch.stack(embeddings)


class BAHE(nn.Module):
    """
    BAHE 模型：结合原子行为嵌入和行为交互模块。
    """

    def __init__(
        self, albert_model_name, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1
    ):
        super(BAHE, self).__init__()
        self.atomic_embedding = SimpleTextEncoder(vocab_size=30000, embed_dim=embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)  # 添加投影层
        self.interaction_model = BehaviorInteractionModel(
            embed_dim, num_heads, ff_dim, num_layers, dropout
        )

    def forward(self, behavior_texts):
        # 获取文本嵌入
        behavior_embeddings = self.atomic_embedding(behavior_texts)
        # 添加序列维度并投影到正确的维度
        behavior_embeddings = behavior_embeddings.unsqueeze(
            1
        )  # [batch_size, 1, embed_dim]
        behavior_embeddings = self.projection(behavior_embeddings)
        # 通过交互模型
        user_embedding = self.interaction_model(behavior_embeddings)
        return user_embedding.squeeze(1)  # 移除序列维度


class BERT4RecEncoder(nn.Module):
    def __init__(
        self, user_length, user_emb_dim, item_length, item_emb_dim, seq_len, hid_dim
    ):
        super().__init__()
        self.user_embedding = EmbeddingLayer(user_length, user_emb_dim)
        self.item_embedding = EmbeddingLayer(item_length, item_emb_dim)
        self.position_embedding = nn.Embedding(seq_len + 1, item_emb_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(item_emb_dim, num_heads=8, ff_dim=hid_dim)
                for _ in range(2)
            ]
        )

    def forward(self, item_ids, seq, domain_ids, user_embedding):
        batch_size, seq_len = seq.size()

        # 获取嵌入
        item_emb = self.item_embedding(seq)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=seq.device))

        # 融合嵌入
        seq_emb = item_emb + pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        if user_embedding is not None:
            user_emb = user_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            seq_emb = seq_emb + user_emb

        # Transformer处理
        mask = (seq > 0).unsqueeze(1).unsqueeze(2)
        for block in self.transformer_blocks:
            seq_emb = block(seq_emb, mask)

        return seq_emb[:, -1, :]  # 返回最后一个时间步的表示


class BERT4RecFusion(nn.Module):
    """BERT4Rec融合器部分，在服务器端运行"""

    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.transformer = TransformerBlock(
            embed_dim, num_heads, ff_dim
        )  # 改名为transformer
        self.predict_layer = nn.Linear(embed_dim, 1)

    def forward(
        self,
        user_embedding_d1,
        user_embedding_d2,
        seq_embedding_d1,
        seq_embedding_d2,
        domain_ids,
    ):
        # 根据domain_id选择正确的表示
        selected_user_embedding = torch.where(
            domain_ids.unsqueeze(-1) == 0, user_embedding_d1, user_embedding_d2
        )

        selected_seq_embedding = torch.where(
            domain_ids.unsqueeze(-1) == 0, seq_embedding_d1, seq_embedding_d2
        )

        # 融合表示
        fused_embedding = self.transformer(
            torch.cat([selected_user_embedding, selected_seq_embedding], dim=1)
        )

        # 生成预测
        output = self.predict_layer(fused_embedding).squeeze(-1)
        return output
