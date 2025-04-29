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

import random
from typing import Callable, Dict, List, Union

import pandas as pd

from secretflow.data import partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU
from secretflow.security.aggregation.aggregator import Aggregator
from secretflow.security.compare.comparator import Comparator
from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.simulation.data._utils import (
    SPLIT_METHOD,
    dirichlet_partition,
    iid_partition,
    label_skew_partition,
)


def create_df(
    source: Union[str, pd.DataFrame, Callable],
    parts: Union[List[PYU], Dict[PYU, Union[float, tuple]]],
    axis: int = 0,
    shuffle: bool = False,
    random_state: int = None,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
    split_method: SPLIT_METHOD = SPLIT_METHOD.IID,
    label_column: str = None,
    **kwargs,
) -> Union[HDataFrame, VDataFrame]:
    """Create a federated dataframe from a single data source.
    TODO: support other backends.
    Args:
        source: the dataset source, shall be a file path or pandas.DataFrame or
            callable (shall returns a pandas.DataFrame).
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict of pyu with value, the value shall be one of the followings:
            1. a float
            2. an interval in tuple closed on the left-side and open on the right-side.
        axis: optional, the value is 0 or 1. 0 means split by row returning a
            horizontal partitioning federated DataFrame. 1 means split by
            column returning a vertical partitioning federated DataFrame.
        shuffle: optional, if suffule the dataset before split.
        random_state: optional, the random state for shuffle.
        aggregator: optional, shall be provided only when axis is 0. For details,
            please refer to `secretflow.data.horizontal.HDataFrame`.
        comparator:  optional, shall be provided only when axis is 0. For details,
            please refer to `secretflow.data.horizontal.HDataFrame`.
        kwargs: optional, will accept params for other split method, such as dirichlet_partition and laebl_skew.

    Returns:
        Union[HDataFrame, VDataFrame]: return a HDataFrame if axis is 0 else
        VDataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'f1': [1, 2, 3, 4], 'f3': [11, 12, 13, 14]})

    >>> # Create a HDataFrame evenly.
    >>> hdf = create_df(df, [alice, bob], axis=0)

    >>> # Create a VDataFrame with a given percentage.
    >>> vdf = create_df(df, {alice: 0.3, bob: 0.7}, axis=1)

    >>> # Create a HDataFrame with a given index.
    >>> hdf = create_df(df, {alice: (0, 1), bob: (1, 4)})

    >>> # Create a HDataFrame with DIRICHLET partition method.
    >>> hdf = create_df(df, [alice,bob], axis=0, split_method=SPLIT_METHOD.DIRICHLET, num_classes=2, alpha=10000)

    >>> # Create a HDataFrame with LABEL_SKEW partition method.
    >>> hdf = create_df(df, [alice,bob], axis=0, split_method=SPLIT_METHOD.LABEL_SKEW, label_column='f3', skew_ratio=0.5)
    """

    assert parts, 'Parts should not be none or empty!'
    if isinstance(source, str):
        # engin="pyarrow" will lead to stuck in production tests.
        df = pd.read_csv(source)
    elif isinstance(source, pd.DataFrame):
        df = source
    elif isinstance(source, Callable):
        df = source()
        assert isinstance(
            df, pd.DataFrame
        ), f'Callable source must return a pandas DataFrame but got {type(df)}'
    else:
        raise InvalidArgumentError(
            f'Unknown source type, expect a file or dataframe or callable but got {type(source)}'
        )
    if not random_state:
        random_state = random.randint(0, 100000)
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)

    total_num = len(df) if axis == 0 else len(df.columns)
    if split_method == SPLIT_METHOD.IID:
        indexes = iid_partition(
            parts=parts,
            total_num=total_num,
            shuffle=shuffle,
            random_seed=random_state,
        )

    elif split_method == SPLIT_METHOD.DIRICHLET:
        assert axis == 0, "dirichlet only support horizontal partition"
        num_classes = kwargs.pop("num_classes", 0)
        alpha = kwargs.pop("alpha", 10000)
        target = df[label_column].values
        assert num_classes > 0, "dirichlet partition must supply num_classes"

        indexes = dirichlet_partition(
            parts=parts,
            targets=target,
            num_classes=num_classes,
            alpha=alpha,
            random_seed=random_state,
        )
    elif split_method == SPLIT_METHOD.LABEL_SCREW:
        assert axis == 0, "label screw only support horizontal partition"
        num_classes = kwargs.pop('num_classes', 0)
        max_class_nums = kwargs.pop('max_class_nums', num_classes)

        assert num_classes > 0, "dirichlet partition must supply num_classes"
        target = df[label_column].values
        indexes = label_skew_partition(
            parts=parts,
            targets=target,
            num_classes=num_classes,
            max_class_nums=max_class_nums,
            random_seed=random_state,
        )
    if axis == 0:
        return HDataFrame(
            partitions={
                device: partition(
                    lambda df, index: df.loc[index, :],
                    device=device,
                    df=df,
                    index=index,
                )
                for device, index in indexes.items()
            },
            aggregator=aggregator,
            comparator=comparator,
        )
    else:
        columns = df.columns
        return VDataFrame(
            partitions={
                device: partition(
                    lambda df, index: df.loc[:, [columns[idx] for idx in index]],
                    device=device,
                    df=df,
                    index=index,
                )
                for device, index in indexes.items()
            }
        )


def create_hdf(
    source: Union[str, pd.DataFrame, Callable],
    parts: Union[List[PYU], Dict[PYU, Union[float, tuple]]],
    shuffle: bool = False,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
    random_seed: int = None,
) -> HDataFrame:
    """Create a HDataFrame from a single dataset source.

    Refer to :py:func:`~.create_df` for full documentation.
    """
    return create_df(
        source=source,
        parts=parts,
        axis=0,
        shuffle=shuffle,
        aggregator=aggregator,
        comparator=comparator,
        random_state=random_seed,
    )


def create_vdf(
    source: Union[str, pd.DataFrame, Callable],
    parts: Union[List[PYU], Dict[PYU, Union[float, tuple]]],
    shuffle: bool = False,
    random_seed: int = None,
) -> VDataFrame:
    """Create a VDataFrame from a single dataset source.

    Refer to :py:func:`~.create_df` for full documentation.
    """
    return create_df(
        source=source,
        parts=parts,
        axis=1,
        shuffle=shuffle,
        random_state=random_seed,
    )
