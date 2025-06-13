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

import collections
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle
from collections import Counter
import numpy as np
import torch
import json
import itertools


def update_dict(target_dict, key, value=None):
    """
    更新字典，添加键值对，如果键不存在，则插入一个新项。

    参数:
        target_dict(dict): 目标字典
        key(string): 目标键
        value(Any, optional): 如果没有提供值，则自动生成一个新值（len(dict)+1）
    """
    if key not in target_dict:
        if value is None:
            target_dict[key] = (
                len(target_dict) + 1
            )  # 如果没有提供值，使用字典的当前长度加1作为新值
        else:
            target_dict[key] = value


def get_sample(all_elements, num_sample):
    """
    从列表中随机选择 `num_sample` 个元素。如果样本数大于列表长度，则重复列表进行抽样。

    参数:
        all_elements(list): 所有元素列表
        num_sample(int): 需要抽样的数量

    返回:
        list: 抽样后的元素列表
    """
    if num_sample > len(all_elements):
        return random.sample(
            all_elements * (num_sample // len(all_elements) + 1), num_sample
        )
    else:
        return random.sample(all_elements, num_sample)


def prepare_distributed_data(cfg, mode="train"):
    """
    准备分布式训练的数据，将数据根据 GPU 数量分成多个文件。

    参数:
        cfg: 配置对象，包含数据路径等信息
        mode(string): 数据处理模式，可为 'train', 'val', 'test'

    返回:
        int: 处理后的数据条数
    """
    data_dir = {
        "train": cfg.dataset.train_dir,
        "val": cfg.dataset.val_dir,
        "test": cfg.dataset.test_dir,
    }

    # 检查是否已处理过目标文件，如果存在且不需要重新处理，则直接返回
    target_file = os.path.join(data_dir[mode], f"behaviors_np{cfg.npratio}_0.tsv")
    if os.path.exists(target_file) and not cfg.reprocess:
        return 0
    print(f'Target_file is not exist. New behavior file in {target_file}')

    behaviors = []
    behavior_file_path = os.path.join(data_dir[mode], 'behaviors.tsv')

    # 处理训练数据
    if mode == 'train':
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                iid, uid, time, history, imp = line.strip().split('\t')
                impressions = [x.split('-') for x in imp.split(' ')]
                pos, neg = [], []
                for news_ID, label in impressions:
                    if label == '0':
                        neg.append(news_ID)
                    elif label == '1':
                        pos.append(news_ID)
                if len(pos) == 0 or len(neg) == 0:
                    continue
                for pos_id in pos:
                    neg_candidate = get_sample(neg, cfg.npratio)  # 从负样本中随机选取
                    neg_str = ' '.join(neg_candidate)
                    new_line = (
                        '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                    )
                    behaviors.append(new_line)
        random.shuffle(behaviors)  # 打乱数据顺序

        # 根据 GPU 数量将数据分配到不同的文件中
        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % cfg.gpu_num].append(line)

    # 处理验证集和测试集数据
    elif mode in ['val', 'test']:
        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        behavior_file_path = os.path.join(data_dir[mode], 'behaviors.tsv')
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                behaviors_per_file[i % cfg.gpu_num].append(line)

    print(f'[{mode}] Writing files...')
    for i in range(cfg.gpu_num):
        processed_file_path = os.path.join(
            data_dir[mode], f'behaviors_np{cfg.npratio}_{i}.tsv'
        )
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors_per_file[i])

    return len(behaviors)


def read_raw_news(cfg, file_path, mode='train'):
    """
    读取原始的新闻数据文件 (news.tsv)，并处理成适合训练的格式。

    参数:
        cfg: 配置对象
        file_path(Path): 文件路径
        mode(string, optional): 模式，'train' 或 'test'

    返回:
        tuple: 返回新闻数据、新闻索引、分类字典、子分类字典、词汇字典
    """
    import nltk

    nltk.download('punkt')  # 下载 NLTK 需要的分词数据

    data_dir = {
        "train": cfg.dataset.train_dir,
        "val": cfg.dataset.val_dir,
        "test": cfg.dataset.test_dir,
    }

    # 如果是验证集或测试集，则加载已处理好的数据字典
    if mode in ['val', 'test']:
        news_dict = pickle.load(open(Path(data_dir["train"]) / "news_dict.bin", "rb"))
        entity_dict = pickle.load(
            open(Path(data_dir["train"]) / "entity_dict.bin", "rb")
        )
        news = pickle.load(open(Path(data_dir["train"]) / "nltk_news.bin", "rb"))
    else:
        news = {}
        news_dict = {}
        entity_dict = {}

    category_dict = {}
    subcategory_dict = {}
    word_cnt = Counter()  # 用于统计词频

    num_line = len(open(file_path, encoding='utf-8').readlines())  # 获取文件的行数
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing raw news"):
            # 分割每一行
            split_line = line.strip('\n').split('\t')
            news_id, category, subcategory, title, abstract, url, t_entity_str, _ = (
                split_line
            )
            update_dict(target_dict=news_dict, key=news_id)

            # 处理实体信息
            if t_entity_str:
                entity_ids = [obj["WikidataId"] for obj in json.loads(t_entity_str)]
                [
                    update_dict(target_dict=entity_dict, key=entity_id)
                    for entity_id in entity_ids
                ]
            else:
                entity_ids = t_entity_str

            tokens = word_tokenize(
                title.lower(), language=cfg.dataset.dataset_lang
            )  # 对标题进行分词

            update_dict(
                target_dict=news,
                key=news_id,
                value=[tokens, category, subcategory, entity_ids, news_dict[news_id]],
            )

            if mode == 'train':
                # 在训练模式下更新分类字典、子分类字典，并统计词频
                update_dict(target_dict=category_dict, key=category)
                update_dict(target_dict=subcategory_dict, key=subcategory)
                word_cnt.update(tokens)

        if mode == 'train':
            # 对词汇进行过滤，只保留出现频率大于指定值的词
            word = [k for k, v in word_cnt.items() if v > cfg.model.word_filter_num]
            word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
            return (
                news,
                news_dict,
                category_dict,
                subcategory_dict,
                entity_dict,
                word_dict,
            )
        else:  # 验证集和测试集
            return news, news_dict, None, None, entity_dict, None


def read_parsed_news(
    cfg,
    news,
    news_dict,
    category_dict=None,
    subcategory_dict=None,
    entity_dict=None,
    word_dict=None,
):
    # 获取新闻的总数，增加1是因为从1开始索引
    news_num = len(news) + 1
    # 初始化分类、子分类、新闻索引、实体的数组
    news_category, news_subcategory, news_index = [
        np.zeros((news_num, 1), dtype='int32') for _ in range(3)
    ]
    news_entity = np.zeros((news_num, 5), dtype='int32')

    # 初始化标题的数组
    news_title = np.zeros((news_num, cfg.model.title_size), dtype='int32')

    # 遍历所有新闻
    for _news_id in tqdm(news, total=len(news), desc="Processing parsed news"):
        _title, _category, _subcategory, _entity_ids, _news_index = news[_news_id]

        # 填充新闻的分类和子分类信息
        news_category[_news_index, 0] = (
            category_dict[_category] if _category in category_dict else 0
        )
        news_subcategory[_news_index, 0] = (
            subcategory_dict[_subcategory] if _subcategory in subcategory_dict else 0
        )
        news_index[_news_index, 0] = news_dict[_news_id]

        # 处理实体信息
        entity_index = [
            entity_dict[entity_id] if entity_id in entity_dict else 0
            for entity_id in _entity_ids
        ]
        news_entity[_news_index, : min(cfg.model.entity_size, len(_entity_ids))] = (
            entity_index[: cfg.model.entity_size]
        )

        # 处理标题的单词信息
        for _word_id in range(min(cfg.model.title_size, len(_title))):
            if _title[_word_id] in word_dict:
                news_title[_news_index, _word_id] = word_dict[_title[_word_id]]

    return news_title, news_entity, news_category, news_subcategory, news_index


def prepare_preprocess_bin(cfg, mode):
    data_dir = {
        "train": cfg.dataset.train_dir,
        "val": cfg.dataset.val_dir,
        "test": cfg.dataset.test_dir,
    }

    if cfg.reprocess is True:
        # 加载原始新闻数据并进行处理
        (
            nltk_news,
            nltk_news_dict,
            category_dict,
            subcategory_dict,
            entity_dict,
            word_dict,
        ) = read_raw_news(
            file_path=Path(data_dir[mode]) / "news.tsv",
            cfg=cfg,
            mode=mode,
        )

        if mode == "train":
            # 如果是训练模式，保存处理结果
            pickle.dump(
                category_dict, open(Path(data_dir[mode]) / "category_dict.bin", "wb")
            )
            pickle.dump(
                subcategory_dict,
                open(Path(data_dir[mode]) / "subcategory_dict.bin", "wb"),
            )
            pickle.dump(word_dict, open(Path(data_dir[mode]) / "word_dict.bin", "wb"))
        else:
            # 如果是验证或测试模式，加载训练时保存的字典
            category_dict = pickle.load(
                open(Path(data_dir["train"]) / "category_dict.bin", "rb")
            )
            subcategory_dict = pickle.load(
                open(Path(data_dir["train"]) / "subcategory_dict.bin", "rb")
            )
            word_dict = pickle.load(
                open(Path(data_dir["train"]) / "word_dict.bin", "rb")
            )

        # 保存实体字典和处理后的新闻数据
        pickle.dump(entity_dict, open(Path(data_dir[mode]) / "entity_dict.bin", "wb"))
        pickle.dump(nltk_news, open(Path(data_dir[mode]) / "nltk_news.bin", "wb"))
        pickle.dump(nltk_news_dict, open(Path(data_dir[mode]) / "news_dict.bin", "wb"))

        # 解析新闻数据并合并特征
        nltk_news_features = read_parsed_news(
            cfg,
            nltk_news,
            nltk_news_dict,
            category_dict,
            subcategory_dict,
            entity_dict,
            word_dict,
        )
        news_input = np.concatenate([x for x in nltk_news_features], axis=1)
        pickle.dump(
            news_input, open(Path(data_dir[mode]) / "nltk_token_news.bin", "wb")
        )
        print("Glove token preprocess finish.")
    else:
        print(f'[{mode}] All preprocessed files exist.')


def prepare_news_graph(cfg, mode='train'):
    data_dir = {
        "train": cfg.dataset.train_dir,
        "val": cfg.dataset.val_dir,
        "test": cfg.dataset.test_dir,
    }

    nltk_target_path = Path(data_dir[mode]) / "nltk_news_graph.pt"

    reprocess_flag = False
    if nltk_target_path.exists() is False:
        reprocess_flag = True

    if (reprocess_flag == False) and (cfg.reprocess == False):
        print(f"[{mode}] All graphs exist !")
        return

    # -----------------------------------------构建新闻图------------------------------------------------
    behavior_path = Path(data_dir['train']) / "behaviors.tsv"
    origin_graph_path = Path(data_dir['train']) / "nltk_news_graph.pt"

    # 加载新闻字典和处理后的新闻数据
    news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    nltk_token_news = pickle.load(
        open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb")
    )

    # ------------------- 构建图 -------------------------------
    if mode == 'train':
        edge_list, user_set = [], set()
        num_line = len(open(behavior_path, encoding='utf-8').readlines())
        with open(behavior_path, 'r', encoding='utf-8') as f:
            for line in tqdm(
                f,
                total=num_line,
                desc=f"[{mode}] Processing behaviors news to News Graph",
            ):
                line = line.strip().split('\t')

                # 检查用户是否已处理过
                used_id = line[1]
                if used_id in user_set:
                    continue
                else:
                    user_set.add(used_id)

                # 处理历史新闻，创建边
                history = line[3].split()
                if len(history) > 1:
                    long_edge = [news_dict[news_id] for news_id in history]
                    edge_list.append(long_edge)

        # 统计图中的节点数
        node_feat = nltk_token_news
        target_path = nltk_target_path
        num_nodes = len(news_dict) + 1

        short_edges = []
        for edge in tqdm(
            edge_list, total=len(edge_list), desc=f"Processing news edge list"
        ):
            # 根据配置选择图的类型（轨迹图或共现图）
            if cfg.model.use_graph_type == 0:
                # 轨迹图
                for i in range(len(edge) - 1):
                    short_edges.append((edge[i], edge[i + 1]))
            elif cfg.model.use_graph_type == 1:
                # 共现图
                for i in range(len(edge) - 1):
                    for j in range(i + 1, len(edge)):
                        short_edges.append((edge[i], edge[j]))
                        short_edges.append((edge[j], edge[i]))
            else:
                assert False, "Wrong"

        # 计算边的权重并创建边的索引
        edge_weights = Counter(short_edges)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor(
            [edge_weights[edge] for edge in unique_edges], dtype=torch.long
        )

        # 保存图数据
        data = Data(
            x=torch.from_numpy(node_feat),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )

        torch.save(data, target_path)
        print(data)
        print(
            f"[{mode}] Finish News Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}"
        )

    elif mode in ['test', 'val']:
        # 加载已存在的图数据
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr
        node_feat = nltk_token_news

        # 保存图数据
        data = Data(
            x=torch.from_numpy(node_feat),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(news_dict) + 1,
        )

        torch.save(data, nltk_target_path)
        print(
            f"[{mode}] Finish nltk News Graph Construction, \nGraph Path: {nltk_target_path}\nGraph Info: {data}"
        )


def prepare_neighbor_list(cfg, mode='train', target='news'):
    # --------------------------------Neighbors List-------------------------------------------
    print(f"[{mode}] Start to process neighbors list")

    # 设置数据目录，根据模式选择训练、验证或测试目录
    data_dir = {
        "train": cfg.dataset.train_dir,
        "val": cfg.dataset.val_dir,
        "test": cfg.dataset.test_dir,
    }

    # 定义邻居字典和权重字典的路径
    neighbor_dict_path = Path(data_dir[mode]) / f"{target}_neighbor_dict.bin"
    weights_dict_path = Path(data_dir[mode]) / f"{target}_weights_dict.bin"

    reprocess_flag = False
    # 检查是否需要重新处理数据
    for file_path in [neighbor_dict_path, weights_dict_path]:
        if file_path.exists() is False:
            reprocess_flag = True

    # 如果数据已经存在且无需重新处理，则跳过
    if (
        (reprocess_flag == False)
        and (cfg.reprocess == False)
        and (cfg.reprocess_neighbors == False)
    ):
        print(f"[{mode}] All {target} Neighbor dict exist !")
        return

    # 根据目标选择相应的图数据和字典
    if target == 'news':
        target_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    elif target == 'entity':
        target_graph_path = Path(data_dir[mode]) / "entity_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    else:
        assert False, f"[{mode}] Wrong target {target} "

    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr

    # 如果图是无向的，转换为无向图
    if cfg.model.directed is False:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    # 初始化邻居字典和权重字典
    neighbor_dict = collections.defaultdict(list)
    neighbor_weights_dict = collections.defaultdict(list)

    # 为每个节点（除了节点0）处理邻居信息
    for i in range(1, len(target_dict) + 1):
        dst_edges = torch.where(edge_index[1] == i)[0]  # 以i为目标节点，查找邻居
        neighbor_weights = edge_attr[dst_edges]
        neighbor_nodes = edge_index[0][dst_edges]  # 邻居作为源节点
        sorted_weights, indices = torch.sort(neighbor_weights, descending=True)
        # 根据权重排序邻居
        neighbor_dict[i] = neighbor_nodes[indices].tolist()
        neighbor_weights_dict[i] = sorted_weights.tolist()

    # 保存邻居字典和权重字典
    pickle.dump(neighbor_dict, open(neighbor_dict_path, "wb"))
    pickle.dump(neighbor_weights_dict, open(weights_dict_path, "wb"))
    print(
        f"[{mode}] Finish {target} Neighbor dict \nDict Path: {neighbor_dict_path}, \nWeight Dict: {weights_dict_path}"
    )


def prepare_entity_graph(cfg, mode='train'):
    data_dir = {
        "train": cfg.dataset.train_dir,
        "val": cfg.dataset.val_dir,
        "test": cfg.dataset.test_dir,
    }

    target_path = Path(data_dir[mode]) / "entity_graph.pt"
    reprocess_flag = False
    # 如果图文件不存在，则需要重新处理
    if target_path.exists() is False:
        reprocess_flag = True
    # 如果不需要重新处理数据，则跳过
    if (
        (reprocess_flag == False)
        and (cfg.reprocess == False)
        and (cfg.reprocess_neighbors == False)
    ):
        print(f"[{mode}] Entity graph exists!")
        return

    entity_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
    origin_graph_path = Path(data_dir['train']) / "entity_graph.pt"

    # 训练模式下重新构建实体图
    if mode == 'train':
        target_news_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        news_graph = torch.load(target_news_graph_path)
        print("news_graph,", news_graph)
        entity_indices = news_graph.x[:, -8:-3].numpy()  # 获取实体索引
        print("entity_indices, ", entity_indices.shape)

        entity_edge_index = []
        # -------- Inter-news -----------------
        # 处理新闻之间的实体关联，注释掉的部分为未处理的代码
        # for entity_idx in entity_indices:
        #     entity_idx = entity_idx[entity_idx > 0]
        #     edges = list(itertools.combinations(entity_idx, r=2))
        #     entity_edge_index.extend(edges)

        news_edge_src, news_edge_dest = news_graph.edge_index
        edge_weights = news_graph.edge_attr.long().tolist()
        for i in range(news_edge_src.shape[0]):
            src_entities = entity_indices[news_edge_src[i]]  # 获取源节点的实体
            dest_entities = entity_indices[news_edge_dest[i]]  # 获取目标节点的实体
            src_entities_mask = src_entities > 0  # 过滤掉没有实体的节点
            dest_entities_mask = dest_entities > 0
            src_entities = src_entities[src_entities_mask]
            dest_entities = dest_entities[dest_entities_mask]
            # 计算实体之间的边
            edges = (
                list(itertools.product(src_entities, dest_entities)) * edge_weights[i]
            )
            entity_edge_index.extend(edges)

        # 计算边的权重并创建边的索引
        edge_weights = Counter(entity_edge_index)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor(
            [edge_weights[edge] for edge in unique_edges], dtype=torch.long
        )

        # --- Entity Graph Undirected
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        # 构建图数据对象
        data = Data(
            x=torch.arange(len(entity_dict) + 1),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(entity_dict) + 1,
        )

        # 保存图数据
        torch.save(data, target_path)
        print(
            f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}"
        )
    elif mode in ['val', 'test']:
        # 加载已经保存的图数据
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr

        # 构建图数据对象
        data = Data(
            x=torch.arange(len(entity_dict) + 1),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(entity_dict) + 1,
        )

        # 保存图数据
        torch.save(data, target_path)
        print(
            f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}"
        )


def prepare_preprocessed_data(cfg):
    # 准备分布式数据
    prepare_distributed_data(cfg, "train")
    prepare_distributed_data(cfg, "val")

    # 准备预处理的二进制数据
    prepare_preprocess_bin(cfg, "train")
    prepare_preprocess_bin(cfg, "val")
    prepare_preprocess_bin(cfg, "test")

    # 准备新闻图
    prepare_news_graph(cfg, 'train')
    prepare_news_graph(cfg, 'val')
    prepare_news_graph(cfg, 'test')

    # 准备新闻的邻居列表
    prepare_neighbor_list(cfg, 'train', 'news')
    prepare_neighbor_list(cfg, 'val', 'news')
    prepare_neighbor_list(cfg, 'test', 'news')

    # 准备实体图
    prepare_entity_graph(cfg, 'train')
    prepare_entity_graph(cfg, 'val')
    prepare_entity_graph(cfg, 'test')

    # 准备实体的邻居列表
    prepare_neighbor_list(cfg, 'train', 'entity')
    prepare_neighbor_list(cfg, 'val', 'entity')
    prepare_neighbor_list(cfg, 'test', 'entity')

    # 处理实体向量
    data_dir = {
        "train": cfg.dataset.train_dir,
        "val": cfg.dataset.val_dir,
        "test": cfg.dataset.test_dir,
    }
    train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"
    test_entity_emb_path = Path(data_dir['test']) / "entity_embedding.vec"

    val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"
    test_combined_path = Path(data_dir['test']) / "combined_entity_embedding.vec"

    # 合并训练集、验证集和测试集的实体嵌入向量
    os.system(
        "cat "
        + f"{train_entity_emb_path} {val_entity_emb_path}"
        + f" > {val_combined_path}"
    )
    os.system(
        "cat "
        + f"{train_entity_emb_path} {test_entity_emb_path}"
        + f" > {test_combined_path}"
    )
