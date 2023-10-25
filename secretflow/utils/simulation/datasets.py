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

import hashlib
import os
import pickle
import random
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import requests
import scipy

from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.data.vertical import read_csv as v_read_csv
from secretflow.device.device.pyu import PYU
from secretflow.security.aggregation import Aggregator
from secretflow.security.compare import Comparator
from secretflow.utils.hash import sha256sum
from secretflow.utils.simulation.data.dataframe import create_df, create_vdf
from secretflow.utils.simulation.data.ndarray import create_ndarray

_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.secretflow/datasets')

_Dataset = namedtuple('_Dataset', ['filename', 'url', 'sha256'])

_DATASETS = {
    'iris': _Dataset(
        'iris.csv',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/iris/iris.csv',
        '92cae857cae978e0c25156265facc2300806cf37eb8700be094228b374f5188c',
    ),
    'dermatology': _Dataset(
        'dermatology.csv',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/dermatology/dermatology.csv',
        '76b63f6c2be12347b1b76f485c6e775e36d0ab5412bdff0e9df5a9885f5ae11e',
    ),
    'bank_marketing': _Dataset(
        'bank.csv',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/bank_marketing/bank.csv',
        'dc8d576e9bda0f41ee891251bd84bab9a39ce576cba715aac08adc2374a01fde',
    ),
    'mnist': _Dataset(
        'mnist.npz',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/mnist/mnist.npz',
        '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1',
    ),
    'linear': _Dataset(
        'linear.csv',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/linear/linear.csv',
        'bf269b267eb9e6985ae82467a4e1ece420de90f3107633cb9b9aeda6632c0052',
    ),
    'cora': _Dataset(
        'cora.zip',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/cora/cora.zip',
        'd7018f2d7d2b693abff6f6f7ccaf9d70e2e428ca068830863f19a37d8575fd01',
    ),
    'bank_marketing_full': _Dataset(
        'bank-full.csv',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/bank_marketing/bank-full.csv',
        'd1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d',
    ),
    'ml-1m': _Dataset(
        'ml-1m.zip',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/movielens/ml-1m.zip',
        'a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20',
    ),
    'pubmed': _Dataset(
        'pubmed.zip',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/pubmed/pubmed.zip',
        '04a5aa8b3b3432d617d35286e42011b64d58ac362a107d2c257d9da85bf0c021',
    ),
    'citeseer': _Dataset(
        'citeseer.zip',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/citeseer/citeseer.zip',
        '8f0f1aba42c7be5818dc43d96913713a2ffc1c0d9dc09bef30d0432d2c102b49',
    ),
    'drive_cleaned': _Dataset(
        'drive_cleaned_demo.csv',
        'https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/drive_cleaned/drive_cleaned_demo.csv',
        'b461458e642596e934f6b8f6d5d4040fdfc21467c83ed79c86a3e6ef8e777bca',
    ),
}


def unzip(file, extract_path=None):
    if not extract_path:
        extract_path = str(Path(file).parent)
    with zipfile.ZipFile(file, 'r') as zip_f:
        zip_f.extractall(extract_path)


def download(url: str, filepath: str, sha256: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    content = requests.get(url, stream=True).content
    h = hashlib.sha256()
    h.update(content)
    actual_sha256 = h.hexdigest()
    assert (
        sha256 == actual_sha256
    ), f'Failed to check sha256 of {url}, expected {sha256}, got {actual_sha256}.'

    with open(filepath, 'wb') as f:
        f.write(content)


def get_dataset(dataset: _Dataset, cache_dir: str = None):
    if not cache_dir:
        cache_dir = _CACHE_DIR

    filepath = f'{cache_dir}/{dataset.filename}'
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    import filelock

    with filelock.FileLock(f'{filepath}.lock'):
        needdownload = not Path(filepath).exists()
        if not needdownload:
            sha256 = sha256sum(filepath)
            if sha256 != dataset.sha256:
                os.remove(filepath)
                needdownload = True

        if needdownload:
            assert (
                dataset.url
            ), f'{dataset.filename} does not exist locally, please give a download url.'

            download(dataset.url, filepath, dataset.sha256)
        return filepath


def dataset(name: str, cache_dir: str = None) -> str:
    """Get the specific dataset file path.

    Args:
        name: the dataset name, should be one of ['iris', 'dermatology',
            'bank_marketing', 'mnist', 'linear'].

    Returns:
        the dataset file path.
    """
    assert name and isinstance(name, str), 'Name shall be a valid string.'
    name = name.lower()
    return get_dataset(_DATASETS[name], cache_dir)


def load_iris(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    axis=0,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
) -> Union[HDataFrame, VDataFrame]:
    """Load iris dataset to federated dataframe.

    This dataset includes columns:
        1. sepal_length
        2. sepal_width
        3. petal_length
        4. petal_width
        5. class

    This dataset originated from `Iris <https://archive.ics.uci.edu/ml/datasets/iris>`_.

    Args:
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict {PYU: value}, the value shall be one of the followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the right-side.
        axis: optional; optional, the value is 0 or 1.
            0 means split by row and returns a horizontal partitioning
            federated DataFrame. 1 means split by column returns a vertical
            partitioning federated DataFrame.
        aggregator: optional, shall be provided only when axis is 0. For details,
            please refer to `secretflow.data.horizontal.HDataFrame`.
        comparator:  optional, shall be provided only when axis is 0. For details,
            please refer to `secretflow.data.horizontal.HDataFrame`.

    Returns:
        return a HDataFrame if axis is 0 else VDataFrame.
    """
    filepath = get_dataset(_DATASETS['iris'])
    return create_df(
        source=filepath,
        parts=parts,
        axis=axis,
        shuffle=False,
        aggregator=aggregator,
        comparator=comparator,
    )


def load_dermatology(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    axis=0,
    class_starts_from_zero: bool = True,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
) -> Union[HDataFrame, VDataFrame]:
    """Load dermatology dataset to federated dataframe.

    This dataset consists of dermatology cancer diagnosis.
    For the original dataset please refer to
    `Dermatology <https://archive.ics.uci.edu/ml/datasets/dermatology>`_.

    Args:
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict {PYU: value}, the value shall be one of the followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the right-side.
        axis: optional, the value could be 0 or 1.
            0 means split by row and returns a horizontal partitioning
            federated DataFrame. 1 means split by column returns a vertical
            partitioning federated DataFrame.
        class_starts_from_zero: optional, class starts from zero if True.
        aggregator: optional, shall be provided only when axis is 0. For details,
            please refer to `secretflow.data.horizontal.HDataFrame`.
        comparator:  optional, shall be provided only when axis is 0. For details,
            please refer to `secretflow.data.horizontal.HDataFrame`.

    Returns:
        return a HDataFrame if axis is 0 else VDataFrame.
    """
    filepath = get_dataset(_DATASETS['dermatology'])
    df = pd.read_csv(filepath)
    if class_starts_from_zero:
        df['class'] = df['class'] - 1
    return create_df(
        source=df,
        parts=parts,
        axis=axis,
        shuffle=False,
        aggregator=aggregator,
        comparator=comparator,
    )


def load_bank_marketing(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    axis=0,
    full=False,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
) -> Union[HDataFrame, VDataFrame]:
    """Load bank marketing dataset to federated dataframe.

    This dataset is related with direct marketing campaigns.
    For the original dataset please refer to
    `Bank marketing <https://archive.ics.uci.edu/ml/datasets/bank+marketing>`_.

    Args:
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict {PYU: value}, the value shall be one of the followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the right-side.
        axis: optional, the value is 0 or 1.
            0 means split by row and returns a horizontal partitioning
            federated DataFrame. 1 means split by column returns a vertical
            partitioning federated DataFrame.
        full: optional. indicates whether to load to full version of dataset.
        aggregator: optional, shall be provided only when axis is 0. For details,
            please refer to `secretflow.data.horizontal.HDataFrame`.
        comparator:  optional, shall be provided only when axis is 0. For details,
            please refer to `secretflow.data.horizontal.HDataFrame`.

    Returns:
        return a HDataFrame if axis is 0 else VDataFrame.
    """
    if full:
        filepath = get_dataset(_DATASETS['bank_marketing_full'])
    else:
        filepath = get_dataset(_DATASETS['bank_marketing'])
    return create_df(
        lambda: pd.read_csv(filepath, sep=';'),
        parts=parts,
        axis=axis,
        shuffle=False,
        aggregator=aggregator,
        comparator=comparator,
    )


def load_mnist(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    normalized_x: bool = True,
    categorical_y: bool = False,
    is_torch: bool = False,
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

    Returns:
        A tuple consists of two tuples, (x_train, y_train) and (x_test, y_test).
    """
    filepath = get_dataset(_DATASETS['mnist'])
    with np.load(filepath) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    if normalized_x:
        x_train, x_test = x_train / 255, x_test / 255

    if categorical_y:
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(sparse=False)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = encoder.fit_transform(y_test.reshape(-1, 1))

    return (
        (
            create_ndarray(x_train, parts=parts, axis=0, is_torch=is_torch),
            create_ndarray(y_train, parts=parts, axis=0),
        ),
        (
            create_ndarray(x_test, parts=parts, axis=0, is_torch=is_torch),
            create_ndarray(y_test, parts=parts, axis=0),
        ),
    )


def load_linear(parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]]) -> VDataFrame:
    """Load the linear dataset to federated dataframe.

    This dataset is random generated and includes columns:
        1) id
        2) 20 features: [x1, x2, x3, ..., x19, x20]
        3) y

    Args:
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict {PYU: value}, the value shall be one of the followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the right-side.

    Returns:
        return a VDataFrame.
    """
    filepath = get_dataset(_DATASETS['linear'])
    return create_vdf(source=filepath, parts=parts, shuffle=False)


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
    assert parts, 'Parts shall not be None or empty!'
    if data_dir is None:
        data_dir = os.path.join(_CACHE_DIR, 'cora')
        if not Path(data_dir).is_dir():
            filepath = get_dataset(_DATASETS['cora'])
            unzip(filepath, data_dir)

    file_names = [
        os.path.join(data_dir, f'ind.cora.{name}')
        for name in ['y', 'tx', 'ty', 'allx', 'ally', 'graph']
    ]

    objects = []
    for name in file_names:
        with open(name, 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))

    y, tx, ty, allx, ally, graph = tuple(objects)

    with open(os.path.join(data_dir, f"ind.cora.test.index"), 'r') as f:
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
    assert parts, 'Parts shall not be None or empty!'
    if data_dir is None:
        data_dir = os.path.join(_CACHE_DIR, 'pubmed')
        if not Path(data_dir).is_dir():
            filepath = get_dataset(_DATASETS['pubmed'])
            unzip(filepath, data_dir)

    file_names = [
        os.path.join(data_dir, f'ind.pubmed.{name}')
        for name in ['y', 'tx', 'ty', 'allx', 'ally', 'graph']
    ]

    objects = []
    for name in file_names:
        with open(name, 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))

    y, tx, ty, allx, ally, graph = tuple(objects)

    with open(os.path.join(data_dir, f"ind.pubmed.test.index"), 'r') as f:
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
    assert parts, 'Parts shall not be None or empty!'
    if data_dir is None:
        data_dir = os.path.join(_CACHE_DIR, 'citeseer')
        if not Path(data_dir).is_dir():
            filepath = get_dataset(_DATASETS['citeseer'])
            unzip(filepath, data_dir)

    file_names = [
        os.path.join(data_dir, f'ind.citeseer.{name}')
        for name in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    ]

    objects = []
    for name in file_names:
        with open(name, 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    with open(os.path.join(data_dir, f"ind.citeseer.test.index"), 'r') as f:
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
        data_dir = os.path.join(_CACHE_DIR, 'ml-1m')
        if not Path(data_dir).is_dir():
            filepath = get_dataset(_DATASETS['ml-1m'])
            unzip(filepath, data_dir)
    extract_dir = os.path.join(data_dir, 'ml-1m')
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
        file_name = os.path.join(data_dir, device.party + ".csv")
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

    return v_read_csv(
        fed_csv,
        keys="ID",
        drop_keys="ID",
    )
