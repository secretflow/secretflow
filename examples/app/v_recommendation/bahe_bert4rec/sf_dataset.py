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

import os
import random
from typing import DefaultDict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from collections import defaultdict
import json


def seq_padding(seq, length_enc, long_length, pad_id):
    if len(seq) >= long_length:
        long_mask = 1
    else:
        long_mask = 0
    if len(seq) >= length_enc:
        enc_in = seq[-length_enc + 1 :]
    else:
        enc_in = [pad_id] * (length_enc - len(seq) - 1) + seq

    return enc_in, long_mask


class DualDomainSeqDataset(data.Dataset):
    def __init__(
        self, seq_len, isTrain, neg_nums, long_length, pad_id, csv_path, domain_id
    ):
        self.seq_len = seq_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.pad_id = pad_id
        self.domain_id = domain_id  # 这是一个固定值，表示当前数据集的域

        # 读取CSV文件
        df = pd.read_csv(csv_path)
        # 只保留对应域的数据
        df = df[df['domain_id'] == domain_id]
        self.user_nodes = df['user_id'].values
        self.seq_d1 = df['seq_d1'].values
        self.seq_d2 = df['seq_d2'].values

        # 构建商品池
        self.item_pool_d1 = self.__build_i_set__(self.seq_d1)
        self.item_pool_d2 = self.__build_i_set__(self.seq_d2)

    def __build_i_set__(self, sequences):
        """构建商品池集合"""
        item_set = set()
        for seq in sequences:
            try:
                items = json.loads(seq)
                item_set.update(items)
            except:
                continue
        return item_set

    def __len__(self):
        return len(self.user_nodes)

    def __generate_behavior_texts__(self, seq_d1_raw, seq_d2_raw, domain_id):
        """根据原始序列生成行为文本"""
        behavior_texts = []

        # 获取原始序列（未填充的）
        if domain_id == 0:  # 电子产品域
            main_seq = seq_d1_raw[-5:]  # 只取最后5个交互
            for item_id in main_seq:
                behavior_texts.append(f"用户在电子产品域购买了商品{item_id}")
            # 如果有跨域行为，添加跨域信息
            if seq_d2_raw:  # 如果在服装域也有行为
                cross_items = seq_d2_raw[-2:]  # 取最后2个跨域行为
                for item_id in cross_items:
                    behavior_texts.append(f"用户在服装域也购买过商品{item_id}")
        else:  # 服装域
            main_seq = seq_d2_raw[-5:]
            for item_id in main_seq:
                behavior_texts.append(f"用户在服装域购买了商品{item_id}")
            # 添加跨域信息
            if seq_d1_raw:  # 如果在电子产品域也有行为
                cross_items = seq_d1_raw[-2:]
                for item_id in cross_items:
                    behavior_texts.append(f"用户在电子产品域也购买过商品{item_id}")

        if not behavior_texts:  # 如果没有行为记录
            behavior_texts.append("这是一个新用户")

        return behavior_texts

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        seq_d1_raw = json.loads(self.seq_d1[idx])  # 原始序列
        seq_d2_raw = json.loads(self.seq_d2[idx])  # 原始序列

        if len(seq_d1_raw) != 0 and len(seq_d2_raw) != 0:
            overlap_label = 1
        else:
            overlap_label = 0

        domain_id_old = self.domain_id
        label = list()

        # 初始化序列变量
        seq_d1_tmp = seq_d1_raw[:]  # 创建副本
        seq_d2_tmp = seq_d2_raw[:]  # 创建副本

        if domain_id_old == 0:
            neg_items_set = self.item_pool_d1 - set(seq_d1_raw)
            item = seq_d1_raw[-1] if seq_d1_raw else None
            if item is not None:
                seq_d1_tmp = seq_d1_raw[:-1]
                label.append(1)
                while item in seq_d1_tmp:
                    seq_d1_tmp.remove(item)
            if self.isTrain:
                neg_samples = (
                    random.sample(list(neg_items_set), 1) if neg_items_set else [0]
                )
                label.append(0)
            else:
                neg_samples = random.sample(
                    list(neg_items_set), min(self.neg_nums, len(neg_items_set))
                )
                neg_samples += [0] * (
                    self.neg_nums - len(neg_samples)
                )  # 补齐不足的部分
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 0
        else:
            neg_items_set = self.item_pool_d2 - set(seq_d2_raw)
            item = seq_d2_raw[-1] if seq_d2_raw else None
            if item is not None:
                seq_d2_tmp = seq_d2_raw[:-1]
                label.append(1)
                while item in seq_d2_tmp:
                    seq_d2_tmp.remove(item)
            if self.isTrain:
                neg_samples = (
                    random.sample(list(neg_items_set), 1) if neg_items_set else [0]
                )
                label.append(0)
            else:
                neg_samples = random.sample(
                    list(neg_items_set), min(self.neg_nums, len(neg_items_set))
                )
                neg_samples += [0] * (
                    self.neg_nums - len(neg_samples)
                )  # 补齐不足的部分
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 1

        # 序列填充
        seq_d1_tmp, long_tail_mask_d1 = seq_padding(
            seq_d1_tmp, self.seq_len + 1, self.long_length, self.pad_id
        )
        seq_d2_tmp, long_tail_mask_d2 = seq_padding(
            seq_d2_tmp, self.seq_len + 1, self.long_length, self.pad_id
        )

        # 生成行为文本
        behavior_texts = self.__generate_behavior_texts__(
            seq_d1_raw, seq_d2_raw, domain_id
        )

        if item is None:
            item = 0  # 设置默认值

        label = np.array([domain_id], dtype=np.float32)  # 直接用 0 或 1 表示标签
        sample = {
            'user_node': np.array([user_node], dtype=np.int64),
            'i_node': np.array([item], dtype=np.int64),
            'seq_d1': np.array([seq_d1_tmp], dtype=np.int64),
            'seq_d2': np.array([seq_d2_tmp], dtype=np.int64),
            'long_tail_mask_d1': np.array([long_tail_mask_d1], dtype=np.float32),
            'long_tail_mask_d2': np.array([long_tail_mask_d2], dtype=np.float32),
            'domain_id': np.array([domain_id], dtype=np.int64),
            'overlap_label': np.array([overlap_label], dtype=np.float32),
            'label': np.array(label, dtype=np.float32),
            'neg_samples': np.array(neg_samples, dtype=np.int64),
            'behavior_texts': behavior_texts,
        }

        return sample


def collate_fn_enhance(batch):
    # 将所有张量转换为Long类型
    user_node = torch.cat(
        [torch.LongTensor(sample['user_node']) for sample in batch], dim=0
    )
    i_node = torch.cat([torch.LongTensor(sample['i_node']) for sample in batch], dim=0)
    seq_d1 = torch.cat([torch.LongTensor(sample['seq_d1']) for sample in batch], dim=0)
    seq_d2 = torch.cat([torch.LongTensor(sample['seq_d2']) for sample in batch], dim=0)
    long_tail_mask_d1 = torch.cat(
        [torch.Tensor(sample['long_tail_mask_d1']) for sample in batch], dim=0
    )
    long_tail_mask_d2 = torch.cat(
        [torch.Tensor(sample['long_tail_mask_d2']) for sample in batch], dim=0
    )
    label = torch.stack(
        [torch.Tensor(sample['label']) for sample in batch], dim=0
    ).squeeze(1)
    domain_id = torch.cat(
        [torch.LongTensor(sample['domain_id']) for sample in batch], dim=0
    )
    overlap_label = torch.cat(
        [torch.Tensor(sample['overlap_label']) for sample in batch], dim=0
    )
    neg_samples = torch.stack(
        [torch.LongTensor(sample['neg_samples']) for sample in batch], dim=0
    )

    # 收集所有behavior_texts
    behavior_texts = [sample['behavior_texts'] for sample in batch]

    data = {
        'user_node': user_node,
        'i_node': i_node,
        'seq_d1': seq_d1,
        'seq_d2': seq_d2,
        'long_tail_mask_d1': long_tail_mask_d1,
        'long_tail_mask_d2': long_tail_mask_d2,
        'label': label,
        'domain_id': domain_id,
        'overlap_label': overlap_label,
        'neg_samples': neg_samples,
        'behavior_texts': behavior_texts,
    }
    return data


def generate_corr_seq(real_seq, fake_seq):
    seq = list()
    for i in range(len(real_seq)):
        seq.append(real_seq[i])
        seq.append(fake_seq[i])
    return seq


# Bob端数据集（域1）
class BobDataset(DualDomainSeqDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 过滤出域1数据（domain_id=0）
        self.data = self.data[self.data['domain_id'] == 0]


# Alice端数据集（域2）
class AliceDataset(DualDomainSeqDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 过滤出域2数据（domain_id=1）
        self.data = self.data[self.data['domain_id'] == 1]
