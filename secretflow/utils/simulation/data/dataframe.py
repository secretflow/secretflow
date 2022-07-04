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


def create_df(
    source: Union[str, pd.DataFrame, Callable],
    parts: Union[List[PYU], Dict[PYU, Union[float, tuple]]],
    axis: int = 0,
    shuffle: bool = False,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
) -> Union[HDataFrame, VDataFrame]:
    """Create a federated dataframe from a single data source.

    Args:
        source: the dataset source, shall be a file path or pandas.DataFrame or
            callable (shall returns a pandas.DataFrame).
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict {PYU: value}, the value shall be one of the followings.
            1) a float
            2) a interval in tuple closed on the left-side and open on the
               right-side.
        axis: optional, the value is 0 or 1. 0 means split by row returning a
            horizontal partitioning federated DataFrame. 1 means split by
            column returning a vertical partitioning federated DataFrame.
        shuffle: optional, if suffule the dataset before split.
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

    if shuffle:
        df = df.sample(frac=1)

    total_num = len(df) if axis == 0 else len(df.columns)
    assert total_num >= len(
        parts
    ), f'Total samples/columns {total_num} is less than parts number {len(parts)}.'

    indexes = []
    devices = None
    # Evenly divided when no pertentages are provided.
    if isinstance(parts, (list, tuple)):
        for part in parts:
            assert isinstance(
                part, PYU
            ), f'Parts shall be list like of PYUs but got {type(part)}.'
        devices = parts
        start, end = 0, 0
        step = total_num / float(len(parts))
        for i in range(len(parts)):
            if i == len(parts) - 1:
                end = total_num
            else:
                end = round((i + 1) * step)
            indexes.append((start, end))
            start = end
    elif isinstance(parts, dict):
        devices = parts.keys()
        for device in devices:
            assert isinstance(
                device, PYU
            ), f'Keys of parts shall be PYU but got {type(device)}.'
        is_percent = isinstance(list(parts.values())[0], float)
        if is_percent:
            for percent in parts.values():
                assert isinstance(
                    percent, float
                ), f'Not all dict values are percentages.'
            assert sum(parts.values()) == 1.0, f'Sum of percentages shall be 1.0.'
            start, end = 0, 0
            for i, percent in enumerate(parts.values()):
                if i == len(parts) - 1:
                    end = total_num
                else:
                    end = round(total_num * percent)
                indexes.append((start, end))
                start = end
        else:
            indexes = list(parts.values())

    if axis == 0:
        return HDataFrame(
            partitions={
                device: Partition(
                    device(lambda df: df.iloc[index[0] : index[1], :])(df)
                )
                for device, index in zip(devices, indexes)
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
                for device, index in zip(devices, indexes)
            }
        )


def create_hdf(
    source: Union[str, pd.DataFrame, Callable],
    parts: Union[List[PYU], Dict[PYU, Union[float, tuple]]],
    shuffle: bool = True,
    aggregator: Aggregator = None,
    comparator: Comparator = None,
) -> HDataFrame:
    """Create a HDataFrame from a single dataset source.

    Refer to `create_df` for full documentation.
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
    shuffle: bool = True,
) -> VDataFrame:
    """Create a VDataFrame from a single dataset source.

    Refer to `create_df` for full documentation.
    """
    return create_df(source=source, parts=parts, axis=1, shuffle=shuffle)
