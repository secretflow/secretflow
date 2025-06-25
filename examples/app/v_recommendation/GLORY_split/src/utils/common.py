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

"""
Common utils and tools.
"""
import pickle
import random

import pandas as pd
import torch
import numpy as np
import pyrootutils
from pathlib import Path
import torch.distributed as dist

import importlib
from omegaconf import DictConfig, ListConfig


def seed_everything(seed):
    '''
    分别对CPU,GPU上的操作以及numpy上的操作进行固定
    '''
    # 设置 PyTorch 的随机种子，影响 CPU 上的所有操作
    torch.manual_seed(seed)

    # 设置所有 GPU 上的随机种子，影响 CUDA 操作
    torch.cuda.manual_seed_all(seed)

    # 设置 CUDNN（NVIDIA GPU 深度学习库）为确定性模式，保证每次运行相同的结果
    # 这样设置可以使得卷积操作的结果具有可重复性
    torch.backends.cudnn.deterministic = True  # backend是命名空间，表示与硬件有关的后端的各种设置，backend模块中有很多子模块，分别对应不同的计算后端

    # 设置 CUDNN 的自动优化关闭，使得每次运行时都使用相同的算法（以确保可重复性）
    # 设置为 False 会启用自动选择最快的卷积算法，但可能导致不同的结果
    torch.backends.cudnn.benchmark = False

    # 设置 Python 自带的 random 模块的种子，用于生成随机数的部分
    # 例如，random.shuffle()、random.choice() 等会受到影响
    random.seed(seed)

    # 设置 numpy 的随机种子，影响 numpy 中的随机操作，例如 np.random.rand()
    np.random.seed(seed)


def load_model(cfg):
    framework = getattr(
        importlib.import_module(f"models.{cfg.model.model_name}"), cfg.model.model_name
    )

    if cfg.model.use_entity:
        entity_dict = pickle.load(
            open(Path(cfg.dataset.val_dir) / "entity_dict.bin", "rb")
        )
        entity_emb_path = Path(cfg.dataset.val_dir) / "combined_entity_embedding.vec"
        entity_emb = load_pretrain_emb(entity_emb_path, entity_dict, 100)
    else:
        entity_emb = None

    if cfg.dataset.dataset_lang == 'english':
        word_dict = pickle.load(
            open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb")
        )
        glove_emb = load_pretrain_emb(
            cfg.path.glove_path, word_dict, cfg.model.word_emb_dim
        )
    else:
        word_dict = pickle.load(
            open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb")
        )
        glove_emb = len(word_dict)
    model = framework(cfg, glove_emb=glove_emb, entity_emb=entity_emb)

    return model


def save_model(cfg, model, optimizer=None, mark=None):
    file_path = Path(
        f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{mark}.pth"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': (
                optimizer.state_dict() if optimizer is not None else None
            ),
        },
        file_path,
    )
    print(f"Model Saved. Path = {file_path}")


def load_pretrain_emb(embedding_file_path, target_dict, target_dim):
    embedding_matrix = np.zeros(shape=(len(target_dict) + 1, target_dim))
    have_item = []
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                itme = line[0].decode()
                if itme in target_dict:
                    index = target_dict[itme]
                    tp = [float(x) for x in line[1:]]
                    embedding_matrix[index] = np.array(tp)
                    have_item.append(itme)
    print('-----------------------------------------------------')
    print(f'Dict length: {len(target_dict)}')
    print(f'Have words: {len(have_item)}')
    miss_rate = (
        (len(target_dict) - len(have_item)) / len(target_dict)
        if len(target_dict) != 0
        else 0
    )
    print(f'Missing rate: {miss_rate}')
    return embedding_matrix


def reduce_mean(result, nprocs):
    rt = result.detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def pretty_print(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key) + '\t' + str(value))


# def get_root():
#   return pyrootutils.setup_root(
#       search_from=__file__,
#       indicator=[".git", "README.md"],
#       pythonpath=True,
#      dotenv=True,
#   )
