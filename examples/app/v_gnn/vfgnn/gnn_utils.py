# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import pickle
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, TensorDataset

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.utils.simulation.datasets import dataset


class GraphTrainLoader(DataLoader):
    def __init__(
        self,
        inputs,
        fanout=[10, 25],
        batch_size=256,
        shuffle=False,
        drop_last=False,
        **kwargs,
    ):
        assert (
            len(inputs) >= 2
        ), f"input should be at least list of [nodes, edges, indices] Optional are [y, sample_weights]"

        self.node, self.edge, self.indices = inputs[0], inputs[1], inputs[2]

        # if indices is not set, use the whole dataset
        if self.indices is None:
            logging.info(
                "`Indices is not set. Manually configured to the full dataset."
            )
            self.indices = torch.arange(self.node.shape[0])

        if len(inputs) >= 4:
            self.y = inputs[3]
            self.has_y = True
        else:
            self.y = None
            self.has_y = False

        # sampel weight
        if len(inputs) >= 5:
            self.s_w = inputs[4]
            self.has_s_w = True
        else:
            self.s_w = None
            self.has_s_w = False

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.fanout = fanout

        # create graph
        self.graph_loader = self.construct_graph_loader(**kwargs)

        # construct sample weight loader
        self.sw_loader = self.construct_sw_loader(**kwargs) if self.has_s_w else None

        self.reset_loader()

        self.count = 0

        super().__init__(
            dataset=None,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )  # do not need to construct the dataset

    def reset_loader(self):
        self.graph_iter = iter(self.graph_loader)
        self.sw_iter = iter(self.sw_loader) if self.has_s_w else None

    def construct_graph_loader(self, **kwargs):
        graph = construct_dgl_graph_from_adjmat(self.node, self.edge, self.y)

        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fi) for fi in self.fanout]
        )

        indices = (
            torch.tensor(np.where(self.indices)).squeeze(axis=0)
            if torch.max(torch.tensor(self.indices)) == 1
            else self.indices
        )
        graph_loader = dgl.dataloading.NodeDataLoader(
            graph,
            indices,
            sampler,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            **kwargs,
        )

        return graph_loader

    @staticmethod
    def cycle(iterable):
        # FIXME: use cycle to manually repeat over the iterable object
        while True:
            for x in iterable:
                yield x

    def construct_sw_loader(self, **kwargs):
        assert self.s_w is not None, "Sample weight is not given!"
        self.s_w = torch.from_numpy(self.s_w)

        if self.s_w.dtype is torch.bool:
            self.s_w = self.s_w.int()

        sw_ds = TensorDataset(self.s_w)
        sw_loader = DataLoader(
            sw_ds,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            **kwargs,
        )
        return sw_loader

    def __len__(self):
        if torch.max(torch.tensor(self.indices)) == 1:
            return math.ceil(
                torch.sum(torch.from_numpy(self.indices)) / self.batch_size
            )
        else:
            return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self):
        self.reset_loader()
        while True:
            input_nodes, batch_nodes, blocks = next(self.graph_iter)

            feats = blocks[0].srcdata['features']
            data_x = (blocks, feats)

            if self.has_y:
                train_y = blocks[-1].dstdata['labels']
                if self.has_s_w:
                    train_sample_weight = next(self.sw_iter)
                    yield data_x, train_y, train_sample_weight
                else:
                    yield data_x, train_y
            else:
                yield data_x


class GraphEvalLoader(DataLoader):
    def __init__(
        self,
        inputs,
        fanout=[10, 25],
        batch_size=256,
        shuffle=False,
        drop_last=False,
        **kwargs,
    ):
        assert (
            len(inputs) >= 2
        ), f"input should be at least list of [nodes, edges, indices] Optional are [y, sample_weights]"

        self.node, self.edge, self.indices = inputs[0], inputs[1], inputs[2]

        if len(inputs) >= 4:
            self.y = inputs[3]
            self.has_y = True
        else:
            self.y = None
            self.has_y = False

        # sampel weight
        if len(inputs) >= 5:
            self.s_w = inputs[4]
            self.has_s_w = True
        else:
            self.s_w = None
            self.has_s_w = False

        self.shuffle = shuffle

        self.batch_size = batch_size
        if (
            batch_size != self.node.shape[0] and self.indices is None
        ):  # mismatch between given `batch_size` and input shape
            logging.warning(
                f'Shape mismatch between given batch size {batch_size} and input size {self.node.shape[0]}. Force to use {self.node.shape[0]}.'
            )
            self.batch_size = self.node.shape[0]  # force batch size

        # if indices is not set, use the whole dataset
        if self.indices is None:
            logging.info(
                "`Indices is not set. Manually configured to the full dataset."
            )
            self.indices = torch.arange(self.node.shape[0])

        self.drop_last = drop_last
        self.fanout = fanout

        # create graph
        self.graph_loader = self.construct_graph_loader(**kwargs)

        # construct sample weight loader
        self.sw_loader = self.construct_sw_loader(**kwargs) if self.has_s_w else None

        self.reset_loader()

        super().__init__(
            dataset=None,  # do not need to construct the dataset
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

    def reset_loader(self):
        self.graph_iter = iter(self.graph_loader)
        self.sw_iter = iter(self.sw_loader) if self.has_s_w else None

    def construct_graph_loader(self, **kwargs):
        graph = construct_dgl_graph_from_adjmat(self.node, self.edge, self.y)

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.fanout))

        indices = (
            torch.tensor(np.where(self.indices)).squeeze(axis=0)
            if torch.max(torch.tensor(self.indices)) == 1
            else self.indices
        )
        graph_loader = dgl.dataloading.NodeDataLoader(
            graph,
            indices,
            sampler,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            **kwargs,
        )

        return graph_loader

    def construct_sw_loader(self, **kwargs):
        assert self.s_w is not None, "Sample weight is not given!"
        self.s_w = torch.from_numpy(self.s_w)

        if self.s_w.dtype is torch.bool:
            self.s_w = self.s_w.int()

        sw_ds = TensorDataset(self.s_w)
        sw_loader = DataLoader(
            sw_ds,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            **kwargs,
        )
        return sw_loader

    def __len__(self):
        if torch.max(torch.tensor(self.indices)) == 1:
            return math.ceil(
                torch.sum(torch.from_numpy(self.indices)) / self.batch_size
            )
        else:
            return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self):
        self.reset_loader()
        while True:
            input_nodes, batch_nodes, blocks = next(self.graph_iter)
            feats = blocks[0].srcdata['features']
            data_x = (blocks, feats)

            if self.has_y:
                train_y = blocks[-1].dstdata['labels']
                if self.has_s_w:
                    train_sample_weight = next(self.sw_iter)
                    yield data_x, train_y, train_sample_weight
                else:
                    yield data_x, train_y
            else:
                yield data_x


# Modified from https://github.com/xinleihe/GNNStealing
def load_dgl_data(dataset_name):
    """
    https://docs.dgl.ai/api/python/dgl.data.html#node-prediction-datasets

    We can select dataset from:
        Citation datasets: Cora, Citeseer, Pubmed
    """
    if dataset_name == "Cora":
        data = dgl.data.CoraGraphDataset(reverse_edge=False, force_reload=True)
    elif dataset_name == "Citeseer":
        data = dgl.data.CiteseerGraphDataset(reverse_edge=False, force_reload=True)
    elif dataset_name == "Pubmed":
        data = dgl.data.PubmedGraphDataset(reverse_edge=False, force_reload=True)
    elif dataset_name == "Amazon":
        data = dgl.data.AmazonCoBuyComputerDataset()
    elif dataset_name == 'Photo':  # not working
        data = dgl.data.AmazonCoBuyPhotoDataset(force_reload=True)
    elif dataset_name == 'Physics':
        data = dgl.data.CoauthorPhysicsDataset()
    elif dataset_name == "Coauthor":
        data = dgl.data.CoauthorCSDataset()
    elif dataset_name == "Reddit":  # not working
        data = dgl.data.RedditDataset()
    elif dataset_name == 'AIFB':  # not working
        data = dgl.data.AIFBDataset()
    elif dataset_name == 'CoraFull':
        data = dgl.data.CoraFullDataset()

    g = data[0]
    g = dgl.add_self_loop(g)

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    logging.info('Total labels: {}'.format(len(g.ndata['labels'])))
    if dataset_name in [
        'Photo',
        'Amazon',
    ]:  # workaround of incorrect value returned by DGL
        return g, len(np.unique(g.ndata['labels']))
    return g, data.num_classes


def split_graph(g, frac_list=[0.6, 0.2, 0.2]):
    train_subset, val_subset, test_subset = dgl.data.utils.split_dataset(
        g, frac_list=frac_list, shuffle=True
    )
    train_g = g.subgraph(train_subset.indices)
    val_g = g.subgraph(val_subset.indices)
    test_g = g.subgraph(test_subset.indices)

    if 'features' not in train_g.ndata:
        train_g.ndata['features'] = train_g.ndata['feat']
    if 'labels' not in train_g.ndata:
        train_g.ndata['labels'] = train_g.ndata['label']

    if 'features' not in val_g.ndata:
        val_g.ndata['features'] = val_g.ndata['feat']
    if 'labels' not in train_g.ndata:
        val_g.ndata['labels'] = val_g.ndata['label']

    if 'features' not in test_g.ndata:
        test_g.ndata['features'] = test_g.ndata['feat']
    if 'labels' not in train_g.ndata:
        test_g.ndata['labels'] = test_g.ndata['label']
    return train_g, val_g, test_g


def load_data_sf(dataset_name: str, parties, local_path=None):
    """
    Data loader for SecretFlow.

    Wrap the loaded np.ndarray to FedNdarray.

    Args:
        dataset_name (str): Name for the dataset. Support ['cora', 'pubmebd', 'citeseer'].
        parties: The device list for all the parties. Specifically, this is for vertical setting.
        local_path: Default is none. If not none, we will load data locally according to this path.

    Returns:
        A list of FedNdarray, including both vertically and horizontally data.
    """
    n_party = len(parties)

    (
        _,
        split_adjs,
        split_feats,
        edge,
        nodes,
        labels,
        y_train,
        y_val,
        y_test,
        idx_train,
        idx_val,
        idx_test,
    ) = load_data_sl(dataset_name, n_party, local_path)

    # convert bool mask to idx list
    # idx_train = np.array(np.where(idx_train)).squeeze(axis=0)
    # idx_val = np.array(np.where(idx_val)).squeeze(axis=0)
    # idx_test = np.array(np.where(idx_test)).squeeze(axis=0)

    edge_arr = FedNdarray(
        partitions={
            party: party(lambda: split_adjs[i])() for i, party in enumerate(parties)
        },
        partition_way=PartitionWay.VERTICAL,
    )

    feat_arr = FedNdarray(
        partitions={
            party: party(lambda: split_feats[i])() for i, party in enumerate(parties)
        },
        partition_way=PartitionWay.VERTICAL,
    )

    Y_train_arr = FedNdarray(
        partitions={parties[0]: parties[0](lambda: y_train)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    Y_val_arr = FedNdarray(
        partitions={parties[0]: parties[0](lambda: y_val)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    Y_test_arr = FedNdarray(
        partitions={parties[0]: parties[0](lambda: y_test)()},
        partition_way=PartitionWay.HORIZONTAL,
    )

    idx_train_arr = FedNdarray(
        partitions={party: party(lambda: idx_train)() for party in parties},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_val_arr = FedNdarray(
        partitions={party: party(lambda: idx_val)() for party in parties},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_test_arr = FedNdarray(
        partitions={party: party(lambda: idx_test)() for party in parties},
        partition_way=PartitionWay.HORIZONTAL,
    )

    return (
        edge_arr,
        feat_arr,
        Y_train_arr,
        Y_val_arr,
        Y_test_arr,
        idx_train_arr,
        idx_val_arr,
        idx_test_arr,
    )


def construct_dgl_graph_from_adjmat(
    features,
    edge_adj,
    labels=None,
):
    n_nodes = features.shape[0]
    src, dst = np.nonzero(edge_adj)

    dgl_graph = dgl.graph((src, dst), num_nodes=n_nodes)
    dgl_graph.ndata['features'] = torch.tensor(features.astype(np.float32))
    if labels is not None:
        dgl_graph.ndata['labels'] = torch.tensor(labels.astype(np.float32))

    return dgl_graph


def load_data_sl(dataset_name: str, n_party=2, local_path=None):
    """
    Data loader for split learning.

    Args:
        dataset_name (str): Name for the dataset. Support ['cora', 'pubmebd', 'citeseer'].
        n_party: The number to seperate the input datasets vertically.
        local_path: Default is none. If not none, we will load data locally according to this path.

    Returns:
        A list of np.ndarray, split features and edges.
    """
    if dataset_name in ['cora', 'pubmed', 'citeseer']:
        if not local_path:
            dataset_zip = dataset(dataset_name)
        else:
            dataset_zip = os.path.join(
                os.path.expanduser('~'), local_path, f'{dataset_name}.zip'
            )

        extract_path = str(Path(dataset_zip).parent)
        logging.info(f'extract_path is {extract_path}, dataset is {dataset_name}')

        with zipfile.ZipFile(dataset_zip, 'r') as zip_f:
            zip_f.extractall(extract_path)

        file_names = [
            os.path.join(extract_path, f'ind.{dataset_name}.{name}')
            for name in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        ]

        objects = []
        for name in file_names:
            with open(name, 'rb') as f:
                objects.append(pickle.load(f, encoding='latin1'))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        with open(
            os.path.join(extract_path, f"ind.{dataset_name}.test.index"), 'r'
        ) as f:
            test_idx_reorder = f.readlines()
        test_idx_reorder = list(map(lambda s: int(s.strip()), test_idx_reorder))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1
            )
            zero_ind = list(set(test_idx_range_full) - set(test_idx_reorder))
            tx_extended = scipy.sparse.lil_matrix(
                (len(test_idx_range_full), x.shape[1])
            )
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty_extended[
                zero_ind - min(test_idx_range),
                np.random.randint(0, y.shape[1], len(zero_ind)),
            ] = 1
            ty = ty_extended

        nodes = scipy.sparse.vstack((allx, tx)).tolil()
        nodes[test_idx_reorder, :] = nodes[test_idx_range, :]

        # load dense adjacency matrix
        edge = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        # edge = edge.toarray() + np.eye(edge.shape[1])   # add self-loop

        def normalize_adj(adj):
            """Symmetrically normalize adjacency matrix."""
            adj = scipy.sparse.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
            return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        # test 1000
        # train #class * 20
        # val 500
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        def sample_mask(idx, length):
            mask = np.zeros(length)
            mask[idx] = 1
            return np.array(mask, dtype=bool)

        idx_train = sample_mask(idx_train, labels.shape[0])
        idx_val = sample_mask(idx_val, labels.shape[0])
        idx_test = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[idx_train, :] = labels[idx_train, :]
        y_val[idx_val, :] = labels[idx_val, :]
        y_test[idx_test, :] = labels[idx_test, :]

        nodes = nodes.toarray()

        # raw 2-party split
        # features_split_pos = round(nodes.shape[1] / 2)
        # nodes_alice, nodes_bob = (
        #     nodes[:, :features_split_pos],
        #     nodes[:, features_split_pos:],
        # )

        # updated evenly features split
        split_feats, n_feats = evenly_split_node_features(nodes, n_party)

        # evenly split edges
        # NOTE: by default, we add self-loop for each party
        split_adjs, n_nodes = evenly_split_edges(edge, n_party)

        pass
    else:
        logging.info(f"Dataset Loading undefined for {dataset_name}")
        exit()

    # save data
    temp_dir = tempfile.mkdtemp()
    saved_files = [
        os.path.join(temp_dir, name)
        for name in [
            'edge_alice.npy',
            'edge_bob.npy',
            'x_alice.npy',
            'x_bob.npy',
            'y_train.npy',
            'y_val.npy',
            'y_test.npy',
            'idx_train.npy',
            'idx_val.npy',
            'idx_test.npy',
        ]
    ]

    return (
        saved_files,
        split_adjs,
        split_feats,
        edge,
        nodes,
        labels,
        y_train,
        y_val,
        y_test,
        idx_train,
        idx_val,
        idx_test,
    )


def sample_edges(adj):
    # convert dense adj to sparse type
    adj = scipy.sparse.csr_matrix(adj)

    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    dic = defaultdict(list)
    for u in range(n_nodes):
        begg, endd = indptr[u : u + 2]
        dic[u] = indices[begg:endd]

    edge_set = []
    existing_set = set([])
    # construct edge set
    for u in range(n_nodes):
        for v in dic[u]:
            if v > u:
                edge_set.append([u, v])
                existing_set.add(",".join([str(u), str(v)]))

    n_samples = len(edge_set)

    unlink = []
    while len(unlink) < n_samples:
        row = np.random.randint(0, n_nodes - 1)
        col = np.random.randint(0, n_nodes - 1)
        if row > col:
            row, col = col, row
        edge_str = ",".join([str(row), str(col)])
        if (row != col) and (edge_str not in existing_set):
            unlink.append([row, col])
            existing_set.add(edge_str)
    return edge_set, unlink


def evenly_split_edges(adj, n_parts=2):
    # convert dense adj to sparse type
    adj = scipy.sparse.csr_matrix(adj)

    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    dic = defaultdict(list)
    for u in range(n_nodes):
        begg, endd = indptr[u : u + 2]
        dic[u] = indices[begg:endd]

    edge_set = []

    # construct edge set
    for u in range(n_nodes):
        for v in dic[u]:
            if v > u:
                edge_set.append((u, v))

    n_samples = len(edge_set)
    seg_size = math.floor(n_samples / n_parts)

    # random shuffle edges
    np.random.shuffle(edge_set)
    split_edges = []

    for i in range(n_parts):
        if i == n_parts - 1:
            split_edges.append(edge_set[seg_size * i :])
        else:
            split_edges.append(edge_set[seg_size * i : seg_size * (i + 1)])

    split_adjs = []
    for i in range(n_parts):
        g = nx.DiGraph()
        g.add_nodes_from(list(range(n_nodes)))
        g.add_edges_from(split_edges[i])

        # add symmetric edges
        dgl_g = dgl.from_networkx(g)
        dgl_g = dgl.add_reverse_edges(dgl_g)

        # add self-loop
        dgl_g = dgl.add_self_loop(dgl_g)

        g = dgl.to_networkx(dgl_g)
        split_adjs.append(nx.adjacency_matrix(g).toarray())

    logging.info(
        f'sampling done! len(edge_set) = {len(edge_set)}, len(split_edges) = {len(split_edges)}, len(split_edges[0])={len(split_edges[0])}'
    )

    return split_adjs, list(range(n_nodes))


def evenly_split_node_features(features, n_party=2):
    n_feats = features.shape[1]
    seg_size = math.floor(n_feats / n_party)

    split_feats = []

    for i in range(n_party):
        if i == n_party - 1:
            split_feats.append(features[:, seg_size * i :])
        else:
            split_feats.append(features[:, seg_size * i : seg_size * (i + 1)])

    return split_feats, n_feats


def construct_dgl_graph_from_numpy(data):
    (
        edge_adj,
        features,
        labels,
        y_train,
        y_val,
        y_test,
        idx_train,
        idx_val,
        idx_test,
    ) = data

    n_classes = len(np.unique(torch.LongTensor(np.where(labels)[1]).unsqueeze(-1)))
    n_nodes = features.shape[0]
    logging.info(f'Node num: {n_nodes}, #class: {n_classes}')

    src, dst = np.nonzero(edge_adj)

    dgl_graph = dgl.graph((src, dst), num_nodes=n_nodes)
    dgl_graph.ndata['features'] = torch.from_numpy(features.astype(np.float32))
    dgl_graph.ndata['labels'] = torch.from_numpy(labels.astype(np.float32))

    return dgl_graph, n_classes


def test_utils():
    def test_load_dgl():
        test_ds = ['Cora', 'Pubmed', 'Citeseer']

        for ds in test_ds:
            g, n_classes = load_dgl_data(ds)
            logging.info(g, n_classes)

    def test_split(dataset):
        g, n_classes = load_dgl_data(dataset)
        train_g, val_g, test_g = split_graph(g, frac_list=[0.6, 0.2, 0.2])
        logging.info(
            train_g.number_of_nodes(), val_g.number_of_nodes(), test_g.number_of_nodes()
        )
        logging.info(
            train_g.ndata['labels'], val_g.number_of_nodes(), test_g.number_of_nodes()
        )

    for ds in ['cora', 'pubmed', 'citeseer']:
        load_data_sl(ds, '.secretflow/datasets')


if __name__ == '__main__':
    test_utils()
