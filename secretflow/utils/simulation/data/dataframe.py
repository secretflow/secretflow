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

from typing import Callable, Dict, List, Union

import pandas as pd

from secretflow.data.base import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU
from secretflow.security.aggregation.aggregator import Aggregator
from secretflow.security.compare.comparator import Comparator
from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.simulation.data._utils import cal_indexes


def create_df(
    source: Union[str, pd.DataFrame, Callable],
    parts: Union[List[PYU], Dict[PYU, Union[float, tuple]]],
    axis: int = 0,
    shuffle: bool = False,
    random_state: int = None,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
) -> Union[HDataFrame, VDataFrame]:
    """Create a federated dataframe from a single data source.

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
    """

    assert parts, 'Parts should not be none or empty!'

    if isinstance(source, str):
        df = pd.read_csv(source, engine="pyarrow")
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

    if shuffle:
        df = df.sample(frac=1, random_state=random_state)

    total_num = len(df) if axis == 0 else len(df.columns)
    indexes = cal_indexes(parts, total_num)
    if axis == 0:
        return HDataFrame(
            partitions={
                device: Partition(
                    device(lambda df: df.iloc[index[0] : index[1], :])(df)
                )
                for device, index in indexes.items()
            },
            aggregator=aggregator,
            comparator=comparator,
        )
    else:
        return VDataFrame(
            partitions={
                device: Partition(
                    device(lambda df: df.iloc[:, index[0] : index[1]])(df)
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
    )


def create_vdf(
    source: Union[str, pd.DataFrame, Callable],
    parts: Union[List[PYU], Dict[PYU, Union[float, tuple]]],
    shuffle: bool = False,
) -> VDataFrame:
    """Create a VDataFrame from a single dataset source.

    Refer to :py:func:`~.create_df` for full documentation.
    """
    return create_df(source=source, parts=parts, axis=1, shuffle=shuffle)
