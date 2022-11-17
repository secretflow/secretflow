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
from secretflow.utils.errors import InvalidArgumentError, NotFoundError, UnexpectedError


@unique
class PartitionWay(Enum):
    """The partitioning.
    HORIZONTAL: horizontal partitioning.
    VERATICAL: vertical partitioning.
    """

    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'


@dataclass
class MixDataFrame:
    """Mixed DataFrame consisting of HDataFrame/VDataFrame.

    MixDataFrame provides two perspectives based on how the data is partitioned.
    Let's illustrate with an example, assuming the following partitions:
    alice_part0, alice_part1, bob, carol, dave_part0/dave_part1.

    Among them, (alice_part0, bob, dave_part0) is aligned, (alice_part1, carol,
    dave_part1) is aligned.

    ============  ===========  ===========
    col1          col2, col3   col4, col5
    ============  ===========  ===========
    alice_part0   bob          dave_part0
    alice_part1   carol        dave_part1
    ============  ===========  ===========

    1. If horizontal partitioned(PartitionWay.HORIZONTAL), the perspective of
    the mixed DataFrame is as follows:

    +-----------------------------------------+
    |      col1, col2, col3, col4, col5       |
    +-----------------------------------------+
    |     alice_part0, bob, dave_part0        |
    +-----------------------------------------+
    |     alice_part1, carol, dave_part1      |
    +-----------------------------------------+

    2. If vertical partitioned(PartitionWay.VERTICAL), the perspective of the
    mixed DataFrame is as follows:

    +-------------+--------------+--------------+
    | col1        |  col2, col3  |  col4, col5  |
    +-------------+--------------+--------------+
    | alice_part0 |   bob        | dave_part0   |
    | alice_part1 |   carol      | dave_part1   |
    +-------------+--------------+--------------+

    MixDataFrame has the following characteristics.

    1. Multiple Partitions corresponding to a column can be provided by
    different parties or by the same party.

    2. The number of Partitions corresponding to each column is the same

    3. The number of aligned Partition samples is the same.
    """

    partitions: Tuple[Union[HDataFrame, VDataFrame]] = None
    """The blocks that make up a mixed DataFrame. Shall all be HDataFrame
       or VDataFrame, and shall not be mixed.
    """

    def __post__init(self):
        self._check_partitions(self.partitions)
        if not isinstance(self.partitions, tuple):
            self.partitions = tuple(self.partitions)

    @staticmethod
    def _check_partitions(partitions):
        assert partitions, 'Partitions should not be None or empty.'
        assert isinstance(
            partitions, (list, tuple)
        ), 'Partitions should be tuple or list.'
        first_part = partitions[0]
        assert isinstance(
            first_part, (HDataFrame, VDataFrame)
        ), f'Not all partitions are HDataFrames or VDataFrames.'
        part_type = type(first_part)

        for part in partitions[1:]:
            assert isinstance(
                part, part_type
            ), f'All partitions should have same type but got {part_type} and {type(part)}.'
            assert len(part.partitions) == len(
                first_part.partitions
            ), f'All partitions should have same partitions quantities.'
            if part_type == VDataFrame:
                assert (
                    part.columns == first_part.columns
                ).all(), 'All partitions should have same columns when partitioned horizontally.'
            else:
                len(part) == len(
                    first_part
                ), 'All partitions should have same length when partitioned vertically.'

    def __setattr__(self, __name: str, __value: Any):
        if __name == 'partitions':
            self._check_partitions(__value)
            if not isinstance(__value, tuple):
                __value = tuple(__value)
        super().__setattr__(__name, __value)

    @property
    def partition_way(self) -> PartitionWay:
        """Data partitioning."""
        if isinstance(self.partitions[0], HDataFrame):
            return PartitionWay.VERTICAL
        elif isinstance(self.partitions[0], VDataFrame):
            return PartitionWay.HORIZONTAL
        else:
            raise UnexpectedError(f'Unknown partition type: {type(self.partitions[0])}')

    def mean(self, *args, **kwargs) -> pd.Series:
        """
        Returns the mean of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.mean`.

        Returns:
            pd.Series
        """
        means = [part.mean(*args, **kwargs) for part in self.partitions]
        if self.partition_way == PartitionWay.HORIZONTAL:
            if 'numeric_only' in kwargs:
                numeric_only = kwargs['numeric_only']
            cnts = [part.count(numeric_only=numeric_only) for part in self.partitions]
            return pd.Series(
                np.average(means, weights=cnts, axis=0), index=means[0].index
            )
        else:
            return pd.concat(means)

    def min(self, *args, **kwargs) -> pd.Series:
        """
        Returns the min of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.min`.

        Returns:
            pd.Series
        """
        mins = [part.min(*args, **kwargs) for part in self.partitions]
        if self.partition_way == PartitionWay.HORIZONTAL:
            return pd.Series(np.min(mins, axis=0), index=mins[0].index)
        else:
            return pd.concat(mins)

    def max(self, *args, **kwargs) -> pd.Series:
        """
        Returns the max of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.max`.

        Returns:
            pd.Series
        """
        maxs = [part.max(*args, **kwargs) for part in self.partitions]
        if self.partition_way == PartitionWay.HORIZONTAL:
            return pd.Series(np.max(maxs, axis=0), index=maxs[0].index)
        else:
            return pd.concat(maxs)

    def count(self, *args, **kwargs) -> pd.Series:
        """Count non-NA cells for each column or row.

        All arguments are same with :py:meth:`pandas.DataFrame.count`.

        Returns:
            pd.Series
        """
        cnts = [part.count(*args, **kwargs) for part in self.partitions]
        if self.partition_way == PartitionWay.HORIZONTAL:
            return sum(cnts)
        else:
            return pd.concat(cnts)

    # TODO(zoupeicheng.zpc): Schedule to implement horizontal and mix case functionality.
    def isna(self):
        raise NotImplementedError

    def quantile(self, q=0.5, axis=0):
        raise NotImplementedError

    def kurtosis(self, *args, **kwargs):
        raise NotImplementedError

    def skew(self, *args, **kwargs):
        raise NotImplementedError

    def sem(self, *args, **kwargs):
        raise NotImplementedError

    def std(self, *args, **kwargs):
        raise NotImplementedError

    def var(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def values(self):
        # TODO
        pass

    @property
    def dtypes(self) -> pd.Series:
        """
        Returns the dtypes in the DataFrame.

        Returns:
            pd.Series: the data type of each column.
        """
        return self.partitions[0].dtypes

    def astype(self, dtype, copy: bool = True, errors: str = "raise"):
        """
        Cast object to a specified dtype ``dtype``.

        All args are same as :py:meth:`pandas.DataFrame.astype`.
        """
        if isinstance(dtype, dict) and self.partition_way == PartitionWay.VERTICAL:
            col_idx = self._col_index(list(dtype.keys()))
            new_parts = []
            for i, part in enumerate(self.partitions):
                if i not in col_idx:
                    new_parts.append(part.copy())
                else:
                    cols = col_idx[i]
                    if not isinstance(cols, list):
                        cols = [cols]
                    new_parts.append(
                        part.astype(
                            dtype={col: dtype[col] for col in cols},
                            copy=copy,
                            errors=errors,
                        )
                    )

            return MixDataFrame(partitions=new_parts)

        return MixDataFrame(
            partitions=[
                vdf.astype(dtype, copy=copy, errors=errors) for vdf in self.partitions
            ]
        )

    @property
    def columns(self):
        """
        The column labels of the DataFrame.
        """
        cols = self.partitions[0].columns
        if self.partition_way == PartitionWay.VERTICAL:
            for part in self.partitions[1:]:
                cols = cols.append(part.columns)
        return cols

    @property
    def shape(self) -> Tuple:
        """Returns a tuple representing the dimensionality of the DataFrame."""
        shapes = [part.shape for part in self.partitions]
        if self.partition_way == PartitionWay.HORIZONTAL:
            return sum([shape[0] for shape in shapes]), shapes[0][1]
        else:
            return shapes[0][0], sum([shape[1] for shape in shapes])

    def copy(self) -> 'MixDataFrame':
        """
        Shallow copy of this dataframe.

        Returns:
            MixDataFrame.
        """
        return MixDataFrame(partitions=[part.copy() for part in self.partitions])

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors='raise',
    ) -> Union['MixDataFrame', None]:
        """Drop specified labels from rows or columns.

        All arguments are same with :py:meth:`pandas.DataFrame.drop`.

        Returns:
            MixDataFrame without the removed index or column labels
            or None if inplace=True.
        """
        new_partitions = [
            part.drop(
                labels=labels,
                axis=axis,
                index=index,
                columns=columns,
                level=level,
                inplace=inplace,
                errors=errors,
            )
            for part in self.partitions
        ]
        if not inplace:
            return MixDataFrame(partitions=new_partitions)

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union['MixDataFrame', None]:
        """Fill NA/NaN values using the specified method.

        All arguments are same with :py:meth:`pandas.DataFrame.fillna`.

        Returns:
            MixDataFrame with missing values filled or None if inplace=True.
        """
        new_partitions = [
            part.fillna(
                value=value,
                method=method,
                axis=axis,
                inplace=inplace,
                limit=limit,
                downcast=downcast,
            )
            for part in self.partitions
        ]
        if not inplace:
            return MixDataFrame(partitions=new_partitions)

    def __len__(self):
        if self.partition_way == PartitionWay.HORIZONTAL:
            return sum([len(part) for part in self.partitions])
        else:
            return max([len(part) for part in self.partitions])

    def _col_index(self, col) -> Dict[int, Union[str, List[str]]]:
        assert (
            col.tolist() if isinstance(col, Index) else col
        ), f'Column to index is None or empty!'
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
                        # Convert to list if more than one column.
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
            return MixDataFrame(
                partitions=[self.partitions[idx][col] for idx, col in part_col.items()]
            )

    def __setitem__(self, key, value):
        if isinstance(value, (HDataFrame, VDataFrame, Partition)):
            raise InvalidArgumentError(
                'Can not assgin a HDataFrame/VDataFrame/Partition to MixDataFrame.'
            )
        elif isinstance(value, MixDataFrame):
            assert len(value.partitions) == len(self.partitions), (
                'Partitions length of the MixDataFrame to assign not equals to this dataframe: '
                f'{len(value.partitions)} != {len(self.partitions)}'
            )
            assert type(value.partitions[0]) == type(self.partitions[0]), (
                'Can not assgin a MixDataFrame with partition type '
                f'{type(value.partitions[0])} differs with {type(self.partitions[0])}.'
            )
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
