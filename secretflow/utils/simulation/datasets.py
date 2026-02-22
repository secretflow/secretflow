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
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import requests

from secretflow.data.horizontal import HDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.device.device.pyu import PYU
from secretflow.security.aggregation import Aggregator
from secretflow.security.compare import Comparator
from secretflow.utils.hash import sha256sum
from secretflow.utils.simulation.data.dataframe import create_df, create_vdf

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".secretflow", "datasets")

_Dataset = namedtuple("_Dataset", ["filename", "url", "sha256"])

_DATASETS = {
    "iris": _Dataset(
        "iris.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/iris/iris.csv",
        "92cae857cae978e0c25156265facc2300806cf37eb8700be094228b374f5188c",
    ),
    "dermatology": _Dataset(
        "dermatology.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/dermatology/dermatology.csv",
        "76b63f6c2be12347b1b76f485c6e775e36d0ab5412bdff0e9df5a9885f5ae11e",
    ),
    "bank_marketing": _Dataset(
        "bank.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/bank_marketing/bank.csv",
        "dc8d576e9bda0f41ee891251bd84bab9a39ce576cba715aac08adc2374a01fde",
    ),
    "mnist": _Dataset(
        "mnist.npz",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/mnist/mnist.npz",
        "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
    ),
    "linear": _Dataset(
        "linear.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/linear/linear.csv",
        "bf269b267eb9e6985ae82467a4e1ece420de90f3107633cb9b9aeda6632c0052",
    ),
    "cora": _Dataset(
        "cora.zip",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/cora/cora.zip",
        "d7018f2d7d2b693abff6f6f7ccaf9d70e2e428ca068830863f19a37d8575fd01",
    ),
    "bank_marketing_full": _Dataset(
        "bank-full.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/bank_marketing/bank-full.csv",
        "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d",
    ),
    "ml-1m": _Dataset(
        "ml-1m.zip",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/movielens/ml-1m.zip",
        "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
    ),
    "pubmed": _Dataset(
        "pubmed.zip",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/pubmed/pubmed.zip",
        "04a5aa8b3b3432d617d35286e42011b64d58ac362a107d2c257d9da85bf0c021",
    ),
    "citeseer": _Dataset(
        "citeseer.zip",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/citeseer/citeseer.zip",
        "8f0f1aba42c7be5818dc43d96913713a2ffc1c0d9dc09bef30d0432d2c102b49",
    ),
    "drive_cleaned": _Dataset(
        "drive_cleaned.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/drive_cleaned/drive_cleaned.csv",
        "324477fec24716097fbf0338d792d254f2a1d5f87faefb23f1842ecbb035930e",
    ),
    "criteo": _Dataset(
        "criteo.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/criteo/criteo.csv",
        "5e6bc83ed1413a6cef82e82f91fe2584514a6084b889d24178ce8adc7397c849",
    ),
    "creditcard": _Dataset(
        "creditcard.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/creditcard/creditcard.csv",
        "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89",
    ),
    "creditcard_small": _Dataset(
        "creditcard_small.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/creditcard/creditcard_small.csv",
        "0ff315b83ac183c9ac877c91a630a4dab717abc2f9882c87376a00a8cde5a8d3",
    ),
    "fremtpl2freq": _Dataset(
        "fremtpl2freq.csv",
        "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/tweedie/freMTPL2freq.csv",
        "c029e69d140f80a8d5bcc3dfcf94b1438d7f838d4d4d8263639780d26b1c5cc6",
    ),
}


def unzip(file, extract_path=None):
    if not extract_path:
        extract_path = str(Path(file).parent)
    with zipfile.ZipFile(file, "r") as zip_f:
        zip_f.extractall(extract_path)


def download(url: str, filepath: str, sha256: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    content = requests.get(url, stream=True).content
    h = hashlib.sha256()
    h.update(content)
    actual_sha256 = h.hexdigest()
    assert (
        sha256 == actual_sha256
    ), f"Failed to check sha256 of {url}, expected {sha256}, got {actual_sha256}."

    with open(filepath, "wb") as f:
        f.write(content)


def get_dataset(dataset: _Dataset, cache_dir: str = None):
    if not cache_dir:
        cache_dir = _CACHE_DIR

    filepath = f"{cache_dir}/{dataset.filename}"
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    import filelock

    with filelock.FileLock(f"{filepath}.lock"):
        needdownload = not Path(filepath).exists()
        if not needdownload:
            sha256 = sha256sum(filepath)
            if sha256 != dataset.sha256:
                os.remove(filepath)
                needdownload = True

        if needdownload:
            assert (
                dataset.url
            ), f"{dataset.filename} does not exist locally, please give a download url."

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
    assert name and isinstance(name, str), "Name shall be a valid string."
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
    filepath = get_dataset(_DATASETS["iris"])
    return create_df(
        source=filepath,
        parts=parts,
        axis=axis,
        shuffle=False,
        aggregator=aggregator,
        comparator=comparator,
    )


def load_iris_unpartitioned():
    filepath = get_dataset(_DATASETS["iris"])
    return pd.read_csv(filepath)


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
    df = load_dermatology_unpartitioned(class_starts_from_zero)
    return create_df(
        source=df,
        parts=parts,
        axis=axis,
        shuffle=False,
        aggregator=aggregator,
        comparator=comparator,
    )


def load_dermatology_unpartitioned(class_starts_from_zero: bool = True):
    filepath = get_dataset(_DATASETS["dermatology"])
    df = pd.read_csv(filepath)
    if class_starts_from_zero:
        df["class"] = df["class"] - 1
    return df


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
        filepath = get_dataset(_DATASETS["bank_marketing_full"])
    else:
        filepath = get_dataset(_DATASETS["bank_marketing"])
    return create_df(
        lambda: pd.read_csv(filepath, sep=";"),
        parts=parts,
        axis=axis,
        shuffle=False,
        aggregator=aggregator,
        comparator=comparator,
    )


def load_bank_marketing_unpartitioned(full=False):
    if full:
        filepath = get_dataset(_DATASETS["bank_marketing_full"])
    else:
        filepath = get_dataset(_DATASETS["bank_marketing"])
    return pd.read_csv(filepath, sep=";")


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
    filepath = get_dataset(_DATASETS["linear"])
    return create_vdf(source=filepath, parts=parts, shuffle=False)


def load_linear_unpartitioned():
    filepath = get_dataset(_DATASETS["linear"])
    return pd.read_csv(filepath)


def load_creditcard_unpartitioned(dataset_name: str = "creditcard"):
    filepath = get_dataset(_DATASETS[dataset_name])
    raw_df = pd.read_csv(filepath)
    raw_df_neg = raw_df[raw_df["Class"] == 0]
    raw_df_pos = raw_df[raw_df["Class"] == 1]
    down_df_neg = raw_df_neg  # .sample(40000)
    down_df = pd.concat([down_df_neg, raw_df_pos])
    cleaned_df = down_df.copy()
    # You don't want the `Time` column.
    cleaned_df.pop("Time")
    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001  # 0 => 0.1
    cleaned_df["Log Ammount"] = np.log(cleaned_df.pop("Amount") + eps)
    return cleaned_df


def load_creditcard(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    axis=1,
    num_sample: int = 284160,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
    dataset_name: str = "creditcard",
):
    if isinstance(parts, List):
        assert len(parts) == 2
        parts = {parts[0]: (0, 25), parts[1]: (25, 30)}

    cleaned_df = load_creditcard_unpartitioned()
    alice_data_index = [
        col
        for col in cleaned_df.columns
        if col != "Class"
        and col != "V1"
        and col != "V2"
        and col != "V3"
        and col != "V4"
    ]
    alice_data = cleaned_df[alice_data_index]
    bob_data = cleaned_df[["V1", "V2", "V3", "V4", "Class"]]
    df = pd.concat([alice_data, bob_data], axis=1)
    df = df[-num_sample:]
    return create_df(
        source=df,
        parts=parts,
        axis=axis,
        aggregator=aggregator,
        comparator=comparator,
        shuffle=False,
    )


def load_creditcard_small(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    axis=1,
    num_sample: int = 50000,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
):
    return load_creditcard(
        parts, axis, num_sample, aggregator, comparator, "creditcard_small"
    )
