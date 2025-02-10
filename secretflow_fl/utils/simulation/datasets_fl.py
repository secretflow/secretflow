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

import os
import random
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy

from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.data.vertical import read_csv as v_read_csv
from secretflow.device.device.pyu import PYU
from secretflow.security.aggregation import Aggregator
from secretflow.security.compare import Comparator
from secretflow.utils import secure_pickle as pickle
from secretflow.utils.simulation.data import create_ndarray
from secretflow.utils.simulation.data.dataframe import create_df
from secretflow.utils.simulation.datasets import (
    _DATASETS,
    get_dataset,
    _CACHE_DIR,
    unzip,
)


def load_cora(
    parts: List[PYU], data_dir: str = None, add_self_loop: bool = True
) -> Tuple[
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
]:
    """Load the cora dataset for split learning GNN.

    Args:
        parts (List[PYU]): parties that the paper features will be partitioned
            evenly.

    Returns:
        A tuple of FedNdarray: edge, x, Y_train, Y_val, Y_valid, index_train,
        index_val, index_test. Note that Y is bound to the first participant.
    """
    assert parts, "Parts shall not be None or empty!"
    if data_dir is None:
        data_dir = os.path.join(_CACHE_DIR, "cora")
        if not Path(data_dir).is_dir():
            filepath = get_dataset(_DATASETS["cora"])
            unzip(filepath, data_dir)

    file_names = [
        os.path.join(data_dir, f"ind.cora.{name}")
        for name in ["y", "tx", "ty", "allx", "ally", "graph"]
    ]

    objects = []
    for name in file_names:
        with open(name, "rb") as f:
            objects.append(pickle.load(f, encoding="latin1"))

    y, tx, ty, allx, ally, graph = tuple(objects)

    with open(os.path.join(data_dir, f"ind.cora.test.index"), "r") as f:
        test_idx_reorder = f.readlines()
    test_idx_reorder = list(map(lambda s: int(s.strip()), test_idx_reorder))
    test_idx_range = np.sort(test_idx_reorder)

    nodes = scipy.sparse.vstack((allx, tx)).tolil()
    nodes[test_idx_reorder, :] = nodes[test_idx_range, :]
    edge_sparse = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    def sample_mask(idx, length):
        mask = np.zeros(length)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    def edge_dense(edge: np.ndarray):
        if add_self_loop:
            return edge + np.eye(edge.shape[1])
        else:
            return edge.toarray()

    nodes = nodes.toarray()
    edge_arr = FedNdarray(
        partitions={part: part(edge_dense)(edge_sparse) for part in parts},
        partition_way=PartitionWay.HORIZONTAL,
    )

    feature_split_idxs = np.rint(np.linspace(0, nodes.shape[1], len(parts) + 1)).astype(
        np.int32
    )
    x_arr = FedNdarray(
        partitions={
            part: part(
                lambda: nodes[:, feature_split_idxs[i] : feature_split_idxs[i + 1]]
            )()
            for i, part in enumerate(parts)
        },
        partition_way=PartitionWay.VERTICAL,
    )
    Y_train_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: y_train)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    Y_val_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: y_val)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    Y_test_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: y_test)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_train_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: train_mask)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_val_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: val_mask)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_test_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: test_mask)()},
        partition_way=PartitionWay.HORIZONTAL,
    )

    return (
        edge_arr,
        x_arr,
        Y_train_arr,
        Y_val_arr,
        Y_test_arr,
        idx_train_arr,
        idx_val_arr,
        idx_test_arr,
    )


def load_pubmed(
    parts: List[PYU], data_dir: str = None, add_self_loop: bool = True
) -> Tuple[
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
]:
    """Load the pubmed dataset for split learning GNN.
    Datasource: https://github.com/kimiyoung/planetoid/tree/master/data

    Args:
        parts (List[PYU]): parties that the paper features will be partitioned
            evenly.

    Returns:
        A tuple of FedNdarray: edge, x, Y_train, Y_val, Y_valid, index_train,
        index_val, index_test. Note that Y is bound to the first participant.
    """
    assert parts, "Parts shall not be None or empty!"
    if data_dir is None:
        data_dir = os.path.join(_CACHE_DIR, "pubmed")
        if not Path(data_dir).is_dir():
            filepath = get_dataset(_DATASETS["pubmed"])
            unzip(filepath, data_dir)

    file_names = [
        os.path.join(data_dir, f"ind.pubmed.{name}")
        for name in ["y", "tx", "ty", "allx", "ally", "graph"]
    ]

    objects = []
    for name in file_names:
        with open(name, "rb") as f:
            objects.append(pickle.load(f, encoding="latin1"))

    y, tx, ty, allx, ally, graph = tuple(objects)

    with open(os.path.join(data_dir, f"ind.pubmed.test.index"), "r") as f:
        test_idx_reorder = f.readlines()
    test_idx_reorder = list(map(lambda s: int(s.strip()), test_idx_reorder))
    test_idx_range = np.sort(test_idx_reorder)

    nodes = scipy.sparse.vstack((allx, tx)).tolil()
    nodes[test_idx_reorder, :] = nodes[test_idx_range, :]
    edge_sparse = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # test 1000
    # train #class * 20 = 7 * 20 = 140
    # val 500
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    def sample_mask(idx, length):
        mask = np.zeros(length)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    def edge_dense(edge: np.ndarray):
        if add_self_loop:
            return edge + np.eye(edge.shape[1])
        else:
            return edge.toarray()

    nodes = nodes.toarray()
    edge_arr = FedNdarray(
        partitions={part: part(edge_dense)(edge_sparse) for part in parts},
        partition_way=PartitionWay.HORIZONTAL,
    )

    feature_split_idxs = np.rint(np.linspace(0, nodes.shape[1], len(parts) + 1)).astype(
        np.int32
    )
    x_arr = FedNdarray(
        partitions={
            part: part(
                lambda: nodes[:, feature_split_idxs[i] : feature_split_idxs[i + 1]]
            )()
            for i, part in enumerate(parts)
        },
        partition_way=PartitionWay.VERTICAL,
    )
    Y_train_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: y_train)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    Y_val_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: y_val)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    Y_test_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: y_test)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_train_arr = FedNdarray(
        partitions={part: part(lambda: train_mask)() for part in parts},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_val_arr = FedNdarray(
        partitions={part: part(lambda: val_mask)() for part in parts},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_test_arr = FedNdarray(
        partitions={part: part(lambda: test_mask)() for part in parts},
        partition_way=PartitionWay.HORIZONTAL,
    )

    return (
        edge_arr,
        x_arr,
        Y_train_arr,
        Y_val_arr,
        Y_test_arr,
        idx_train_arr,
        idx_val_arr,
        idx_test_arr,
    )


def load_citeseer(
    parts: List[PYU], data_dir: str = None, add_self_loop: bool = True
) -> Tuple[
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
    FedNdarray,
]:
    """Load the citeseer dataset for split learning GNN.
    Datasource: https://github.com/kimiyoung/planetoid/tree/master/data

    Args:
        parts (List[PYU]): parties that the paper features will be partitioned
            evenly.

    Returns:
        A tuple of FedNdarray: edge, x, Y_train, Y_val, Y_valid, index_train,
        index_val, index_test. Note that Y is bound to the first participant.
    """
    assert parts, "Parts shall not be None or empty!"
    if data_dir is None:
        data_dir = os.path.join(_CACHE_DIR, "citeseer")
        if not Path(data_dir).is_dir():
            filepath = get_dataset(_DATASETS["citeseer"])
            unzip(filepath, data_dir)

    file_names = [
        os.path.join(data_dir, f"ind.citeseer.{name}")
        for name in ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    ]

    objects = []
    for name in file_names:
        with open(name, "rb") as f:
            objects.append(pickle.load(f, encoding="latin1"))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    with open(os.path.join(data_dir, f"ind.citeseer.test.index"), "r") as f:
        test_idx_reorder = f.readlines()
    test_idx_reorder = list(map(lambda s: int(s.strip()), test_idx_reorder))
    test_idx_range = np.sort(test_idx_reorder)

    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    zero_ind = list(set(test_idx_range_full) - set(test_idx_reorder))
    tx_extended = scipy.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
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
    edge_sparse = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # test 1000
    # train #class * 20 = 6 * 20 = 120
    # val 500
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    def sample_mask(idx, length):
        mask = np.zeros(length)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    def edge_dense(edge: np.ndarray):
        if add_self_loop:
            return edge + np.eye(edge.shape[1])
        else:
            return edge.toarray()

    nodes = nodes.toarray()
    edge_arr = FedNdarray(
        partitions={part: part(edge_dense)(edge_sparse) for part in parts},
        partition_way=PartitionWay.HORIZONTAL,
    )

    feature_split_idxs = np.rint(np.linspace(0, nodes.shape[1], len(parts) + 1)).astype(
        np.int32
    )
    x_arr = FedNdarray(
        partitions={
            part: part(
                lambda: nodes[:, feature_split_idxs[i] : feature_split_idxs[i + 1]]
            )()
            for i, part in enumerate(parts)
        },
        partition_way=PartitionWay.VERTICAL,
    )
    Y_train_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: y_train)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    Y_val_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: y_val)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    Y_test_arr = FedNdarray(
        partitions={parts[0]: parts[0](lambda: y_test)()},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_train_arr = FedNdarray(
        partitions={part: part(lambda: train_mask)() for part in parts},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_val_arr = FedNdarray(
        partitions={part: part(lambda: val_mask)() for part in parts},
        partition_way=PartitionWay.HORIZONTAL,
    )
    idx_test_arr = FedNdarray(
        partitions={part: part(lambda: test_mask)() for part in parts},
        partition_way=PartitionWay.HORIZONTAL,
    )

    return (
        edge_arr,
        x_arr,
        Y_train_arr,
        Y_val_arr,
        Y_test_arr,
        idx_train_arr,
        idx_val_arr,
        idx_test_arr,
    )


def load_ml_1m(
    part: Dict[PYU, List],
    data_dir: str = None,
    shuffle: bool = False,
    num_sample: int = -1,
):
    """Load the movie lens 1M dataset for split learning.

    Args:
        parts (Dict[PYU, List]): party map features columns
        data_dir: data dir if data has been downloaded
        shuffle: whether need shuffle
        num_sample: num of samples, default -1 for all

    Returns:
        A tuple of FedNdarray: edge, x, Y_train, Y_val, Y_valid, index_train,
        index_val, index_test. Note that Y is bound to the first participant.
    """

    def _load_data(filename, columns):
        data = {}
        with open(filename, "r", encoding="unicode_escape") as f:
            for line in f:
                ls = line.strip("\n").split("::")
                data[ls[0]] = dict(zip(columns[1:], ls[1:]))
        return data

    def _shuffle_data(filename):
        shuffled_filename = f"{filename}.shuffled"
        with open(filename, "r") as f:
            lines = f.readlines()
        random.shuffle(lines)
        with open(shuffled_filename, "w") as f:
            f.writelines(lines)
        return shuffled_filename

    def _parse_example(feature, columns, index):
        if "Title" in feature.keys():
            feature["Title"] = feature["Title"].replace(",", "_")
        if "Genres" in feature.keys():
            feature["Genres"] = feature["Genres"].replace("|", " ")
        values = []
        values.append(str(index))
        for c in columns:
            values.append(feature[c])
        return ",".join(values)

    if data_dir is None:
        data_dir = os.path.join(_CACHE_DIR, "ml-1m")
        if not Path(data_dir).is_dir():
            filepath = get_dataset(_DATASETS["ml-1m"])
            unzip(filepath, data_dir)
    extract_dir = os.path.join(data_dir, "ml-1m")
    users_data = _load_data(
        extract_dir + "/users.dat",
        columns=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    )
    movies_data = _load_data(
        extract_dir + "/movies.dat", columns=["MovieID", "Title", "Genres"]
    )
    ratings_columns = ["UserID", "MovieID", "Rating", "Timestamp"]
    index = 0

    csv_writer_container = {}

    fed_csv = {}
    for device, columns in part.items():
        file_name = os.path.join(
            data_dir, device.party + f"_{uuid.uuid4().int}" + ".csv"
        )
        fed_csv[device] = file_name
        _csv_writer = open(file_name, "w")
        csv_writer_container[device] = _csv_writer
        _csv_writer.write("ID," + ",".join(columns) + "\n")
    if shuffle:
        shuffled_filename = _shuffle_data(extract_dir + "/ratings.dat")
        f = open(shuffled_filename, "r", encoding="unicode_escape")
    else:
        f = open(extract_dir + "/ratings.dat", "r", encoding="unicode_escape")

    for line in f:
        ls = line.strip().split("::")
        rating = dict(zip(ratings_columns, ls))
        rating.update(users_data.get(ls[0]))
        rating.update(movies_data.get(ls[1]))
        for device, columns in part.items():
            parse_f = _parse_example(rating, columns, index)
            csv_writer_container[device].write(parse_f + "\n")
        index += 1
        if num_sample > 0 and index >= num_sample:
            break
    for w in csv_writer_container.values():
        w.close()
    try:
        vdf = v_read_csv(
            fed_csv,
            keys="ID",
            drop_keys="ID",
        )
    finally:
        for v in fed_csv.values():
            os.remove(v)
    return vdf


def load_criteo(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    axis=1,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
    num_samples: int = 410000,
) -> Union[VDataFrame, HDataFrame]:
    df = load_criteo_unpartitioned(num_samples)
    if isinstance(parts, List):
        assert len(parts) == 2
        parts = {parts[0]: (14, 40), parts[1]: (0, 14)}
    return create_df(
        source=df,
        parts=parts,
        axis=axis,
        shuffle=False,
        aggregator=aggregator,
        comparator=comparator,
    )


def load_criteo_unpartitioned(num_samples):
    filepath = get_dataset(_DATASETS["criteo"])
    dtypes = {"Label": "int"}
    dtypes.update({f"I{i}": "float" for i in range(1, 14)})
    dtypes.update({f"C{i}": "str" for i in range(1, 27)})
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=list(dtypes.keys()),
        dtype=dtypes,
        nrows=num_samples,
    )
    return df


def load_cifar10(
    parts: List[PYU], data_dir: str = None, axis=1, aggregator=None, comparator=None
) -> ((FedNdarray, FedNdarray), (FedNdarray, FedNdarray)):
    import torch.utils.data as torch_data
    from torchvision import datasets, transforms

    assert axis == 1, f"only support axis = 1 split cifar10 yet."
    assert len(parts) == 2
    alice, bob = parts[0], parts[1]
    if data_dir is None:
        data_dir = _CACHE_DIR + "/cifar10"
    train_dataset = datasets.CIFAR10(
        data_dir, True, transform=transforms.ToTensor(), download=True
    )
    train_loader = torch_data.DataLoader(
        dataset=train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    test_dataset = datasets.CIFAR10(
        data_dir, False, transform=transforms.ToTensor(), download=True
    )
    test_loader = torch_data.DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    train_data, train_labels = next(iter(train_loader))
    train_plain_data = train_data.numpy()
    train_plain_label = train_labels.numpy()
    train_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, :, :, 0:16])(train_plain_data),
            bob: bob(lambda x: x[:, :, :, 16:32])(train_plain_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    train_label = bob(lambda x: x)(train_plain_label)
    test_data, test_labels = next(iter(test_loader))
    test_plain_data = test_data.numpy()
    test_plain_label = test_labels.numpy()
    test_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, :, :, 0:16])(test_plain_data),
            bob: bob(lambda x: x[:, :, :, 16:32])(test_plain_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    test_label = bob(lambda x: x)(test_plain_label)
    return (train_data, train_label), (test_data, test_label)


def load_mnist_unpartitioned(normalized_x: bool = True, categorical_y: bool = False):
    filepath = get_dataset(_DATASETS['mnist'])
    with np.load(filepath) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    if normalized_x:
        x_train, x_test = x_train / 255, x_test / 255

    if categorical_y:
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = encoder.fit_transform(y_test.reshape(-1, 1))
    return ((x_train, y_train), (x_test, y_test))


def load_mnist(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    normalized_x: bool = True,
    categorical_y: bool = False,
    is_torch=False,
    axis: int = 0,
) -> Tuple[Tuple[FedNdarray, FedNdarray], Tuple[FedNdarray, FedNdarray]]:
    """Load mnist dataset to federated ndarrays.

    This dataset has a training set of 60,000 examples, and a test set of 10,000 examples.
    Each example is a 28x28 grayscale image of the 10 digits.
    For the original dataset please refer to `MNIST <http://yann.lecun.com/exdb/mnist/>`_.

    Args:
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict {PYU: value}, the value shall be one of the followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the right-side.
        normalized_x: optional, normalize x if True. Default to True.
        categorical_y: optional, do one hot encoding to y if True. Default to True.
        is_torch: torch need new axis.
        axis: the axis of the data, 0 for HORIZONTAL, 1 for VERTICAL.
    Returns:
        A tuple consists of two tuples, (x_train, y_train) and (x_test, y_test).
    """
    ((x_train, y_train), (x_test, y_test)) = load_mnist_unpartitioned(
        normalized_x, categorical_y
    )
    return (
        (
            create_ndarray(x_train, parts=parts, axis=axis, is_torch=is_torch),
            create_ndarray(y_train, parts=parts, axis=axis, is_label=True),
        ),
        (
            create_ndarray(x_test, parts=parts, axis=axis, is_torch=is_torch),
            create_ndarray(y_test, parts=parts, axis=axis, is_label=True),
        ),
    )


def load_cifar10_unpartitioned(
    normalized_x: bool = True, categorical_y: bool = False, data_dir: str = None
):
    from torchvision import datasets

    if data_dir is None:
        data_dir = _CACHE_DIR + "/cifar10"
    train_dataset = datasets.CIFAR10(data_dir, True, transform=None, download=True)
    test_dataset = datasets.CIFAR10(data_dir, False, transform=None, download=True)
    x_train, y_train = train_dataset.data, train_dataset.targets
    x_test, y_test = test_dataset.data, test_dataset.targets
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = x_train.transpose((0, 3, 1, 2))
    x_test = x_test.transpose((0, 3, 1, 2))
    if normalized_x:
        x_train, x_test = x_train / 255, x_test / 255

    if categorical_y:
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = encoder.fit_transform(y_test.reshape(-1, 1))
    return ((x_train, y_train), (x_test, y_test))


def load_cifar10_horiontal(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    normalized_x: bool = True,
    categorical_y: bool = False,
    is_torch=False,
    axis: int = 0,
) -> Tuple[Tuple[FedNdarray, FedNdarray], Tuple[FedNdarray, FedNdarray]]:
    """Load cifar10 dataset to federated ndarrays.

    This dataset has a training set of 60,000 examples, and a test set of 10,000 examples.
    Args:
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict {PYU: value}, the value shall be one of the followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the right-side.
        normalized_x: optional, normalize x if True. Default to True.
        categorical_y: optional, do one hot encoding to y if True. Default to True.
        is_torch: torch need new axis.
        axis: the axis of the data, 0 for HORIZONTAL, 1 for VERTICAL.
    Returns:
        A tuple consists of two tuples, (x_train, y_train) and (x_test, y_test).
    """
    ((x_train, y_train), (x_test, y_test)) = load_cifar10_unpartitioned(
        normalized_x, categorical_y
    )
    return (
        (
            create_ndarray(x_train, parts=parts, axis=axis, is_torch=is_torch),
            create_ndarray(y_train, parts=parts, axis=axis, is_label=True),
        ),
        (
            create_ndarray(x_test, parts=parts, axis=axis, is_torch=is_torch),
            create_ndarray(y_test, parts=parts, axis=axis, is_label=True),
        ),
    )
