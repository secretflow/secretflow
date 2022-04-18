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

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index

from secretflow.data.base import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.security.aggregation.aggregator import Aggregator
from secretflow.security.compare.comparator import Comparator
from secretflow.utils.errors import (InvalidArgumentError, NotFoundError,
                                     UnexpectedError)


@unique
class PartitionWay(Enum):
    """混合DataFrame的数据切分方式。

    HORIZONTAL表示数据被水平切分，VERTICAL表示被垂直切分，更详细参见MixDataFrame的注释。
    """
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'


@dataclass
class MixDataFrame:
    """由HDataFrame/VDataFrame组成的混合DataFrame。

    混合DataFrame根据数据切分方式提供两种视角。
    我们通过一个示例来说明，假设有以下Partition：
    alice_part0/alice_part1、bob、carol、dave_part0/dave_part1。
    其中，(alice_part0, bob, dave_part0)的数据是对齐的，
    (alice_part1, carol, dave_part1)的数据是对齐的。
    +------------+-------------——----------+
    |   col1     | col2, col3 | col4, col5 |
    +--------------------------——----------+
    | alice_part0|   bob,     | dave_part0 |
    +---------------------------——---------+
    | alice_part1|   carol    | dave_part1 |
    +------------+------------+-——---------+

    1） 若切分方式为水平（PartitionWay.HORIZONTAL），则混合DataFrame的视角如下：
    +-----------------------------------------+
    |      col1, col2, col3, col4, col5       |
    +-----------------------------------------+
    |     (alice_part0, bob, dave_part0)      |
    +-----------------------------------------+
    |     (alice_part1, carol, dave_part1)    |
    +-----------------------------------------+
    2. 若切分方式为垂直（PartitionWay.VERTICAL），则混合DataFrame的视角如下：
    +-------------+--------------+--------------+
    | col1        |  col2, col3  |  col4, col5  |
    +-------------------------------------------+
    |(alice_part0,|  (bob,       |(dave_part0,  |
    | alice_part1)|   carol)     | dave_part1)  |
    +-------------+--------------+--------------+

    混合DataFrame具有以下特点。
    1. 列对应的多个Partition可以由不同方提供，也可以是同一方提供。
    2. 每个列对应的Partition数量是相同的
    3. 对齐的多个Partition样本数是相同的。
    """

    partitions: Tuple[Union[HDataFrame, VDataFrame]] = None
    """组成混合DataFrame的分块，必须全部是HDataFrame或者VDataFrame，不应该出现混用。
    """

    def __post__init(self):
        self._check_partitions(self.partitions)
        if not isinstance(self.partitions, tuple):
            self.partitions = tuple(self.partitions)

    @staticmethod
    def _check_partitions(partitions):
        assert partitions, 'Partitions should not be None or empty.'
        assert isinstance(partitions, (list, tuple)), 'Partitions should be tuple or list.'
        first_part = partitions[0]
        assert isinstance(first_part, (HDataFrame, VDataFrame)), f'Not all partitions are hdatafrane or vdataframes.'
        part_type = type(first_part)

        for part in partitions[1:]:
            assert isinstance(
                part, part_type), f'All partitions should have same type but got {part_type} and {type(part)}.'
            assert len(
                part.partitions) == len(
                first_part.partitions), f'All partitions should have same partitions quantities.'
            if part_type == VDataFrame:
                assert (part.columns == first_part.columns).all(
                ), 'All partitions should have same columns when partitioned horizontally.'
            else:
                len(part) == len(first_part), 'All partitions should have same length when partitioned vertically.'

    def __setattr__(self, __name: str, __value: Any):
        if __name == 'partitions':
            self._check_partitions(__value)
            if not isinstance(__value, tuple):
                __value = tuple(__value)
        super().__setattr__(__name, __value)

    @property
    def partition_way(self) -> PartitionWay:
        if isinstance(self.partitions[0], HDataFrame):
            return PartitionWay.VERTICAL
        elif isinstance(self.partitions[0], VDataFrame):
            return PartitionWay.HORIZONTAL
        else:
            raise UnexpectedError(f'Unknown partition type: {type(self.partitions[0])}')

    def mean(self, *args, **kwargs) -> pd.Series:
        means = [part.mean(*args, **kwargs) for part in self.partitions]
        if self.partition_way == PartitionWay.HORIZONTAL:
            if 'numeric_only' in kwargs:
                numeric_only = kwargs['numeric_only']
            cnts = [part.count(numeric_only=numeric_only) for part in self.partitions]
            return pd.Series(np.average(means, weights=cnts, axis=0), index=means[0].index)
        else:
            return pd.concat(means)

    def min(self, *args, **kwargs) -> pd.Series:
        mins = [part.min(*args, **kwargs) for part in self.partitions]
        if self.partition_way == PartitionWay.HORIZONTAL:
            return pd.Series(np.min(mins, axis=0), index=mins[0].index)
        else:
            return pd.concat(mins)

    def max(self, *args, **kwargs) -> pd.Series:
        maxs = [part.max(*args, **kwargs) for part in self.partitions]
        if self.partition_way == PartitionWay.HORIZONTAL:
            return pd.Series(np.max(maxs, axis=0), index=maxs[0].index)
        else:
            return pd.concat(maxs)

    def count(self, *args, **kwargs) -> pd.Series:
        cnts = [part.count(*args, **kwargs) for part in self.partitions]
        if self.partition_way == PartitionWay.HORIZONTAL:
            return sum(cnts)
        else:
            return pd.concat(cnts)

    @property
    def values(self):
        # TODO
        pass

    @property
    def dtypes(self) -> pd.Series:
        return self.partitions[0].dtypes

    @property
    def columns(self):
        cols = self.partitions[0].columns
        if self.partition_way == PartitionWay.VERTICAL:
            for part in self.partitions[1:]:
                cols.append(part.columns)
        return cols

    def copy(self) -> 'MixDataFrame':
        return MixDataFrame(partitions=[part.copy() for part in self.partitions])

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise') -> Union[
            'MixDataFrame', None]:
        new_partitions = [part.drop(labels=labels, axis=axis, index=index, columns=columns,
                                    level=level, inplace=inplace, errors=errors) for part in self.partitions]
        if not inplace:
            return MixDataFrame(partitions=new_partitions)

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None) -> Union['MixDataFrame', None]:
        new_partitions = [
            part.fillna(
                value=value, method=method, axis=axis, inplace=inplace, limit=limit, downcast=downcast)
            for part in self.partitions]
        if not inplace:
            return HDataFrame(partitions=new_partitions)

    def __len__(self):
        if self.partition_way == PartitionWay.HORIZONTAL:
            return sum([len(part) for part in self.partitions])
        else:
            return max([len(part) for part in self.partitions])

    def _col_index(self, col) -> Dict[int, Union[str, List[str]]]:
        assert col.tolist() if isinstance(col, Index) else col, f'Column to index is None or empty!'
        part_col = {}
        listed_col = col.tolist() if isinstance(col, Index) else col
        if not isinstance(listed_col, (list, tuple)):
            listed_col = [listed_col]
        for key in listed_col:
            found = False
            for i, part in enumerate(self.partitions):
                if key not in part.dtypes:
                    continue

                found = True
                if i not in part_col:
                    part_col[i] = key
                else:
                    if not isinstance(part_col[i], list):
                        # 有多个列，则转为列表。
                        part_col[i] = [part_col[i]]
                    part_col[i].append(key)

                break

            if not found:
                raise NotFoundError(f'Item {key} does not exist.')
        return part_col

    def __getitem__(self, item) -> 'MixDataFrame':
        if self.partition_way == PartitionWay.HORIZONTAL:
            return MixDataFrame(partitions=[part[item] for part in self.partitions])
        else:
            part_col = self._col_index(item)
            return MixDataFrame(partitions=[self.partitions[idx][col] for idx, col in part_col.items()])

    def __setitem__(self, key, value):
        if isinstance(value, (HDataFrame, VDataFrame, Partition)):
            raise InvalidArgumentError('Can not assgin a HDataFrame/VDataFrame/Partition to MixDataFrame.')
        elif isinstance(value, MixDataFrame):
            assert len(value.partitions) == len(
                self.partitions), f'Partitions length of the MixDataFrame to assign not equals to this dataframe: {len(value.partitions)} != {len(self.partitions)}'
            assert type(
                value.partitions[0]) == type(
                self.partitions[0]), f'Can not assgin a MixDataFrame with partition type {type(value.partitions[0])} differs with {type(self.partitions[0])}.'
            if self.partition_way == PartitionWay.HORIZONTAL:
                for i, part in enumerate(value.partitions):
                    self.partitions[i][key] = part
            else:
                part_key = self._col_index(key)
                for idx, key in part_key.items():
                    self.partitions[idx][key] = value.partitions[idx]
        else:
            for part in self.partitions:
                part[key] = value
