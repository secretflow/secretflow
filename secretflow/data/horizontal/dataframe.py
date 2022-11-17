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

from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
import pandas as pd

from secretflow.data.base import DataFrameBase, Partition
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.device import PYU, reveal
from secretflow.security.aggregation.aggregator import Aggregator
from secretflow.security.compare.comparator import Comparator
from secretflow.utils.errors import InvalidArgumentError


@dataclass
class HDataFrame(DataFrameBase):
    """Federated dataframe holds `horizontal` partitioned data.

    This dataframe is design to provide a federated pandas dataframe
    and just same as using pandas. The original data is still stored
    locally in the data holder and is not transmitted out of the domain
    during all the methods execution.

    In some methods we need to compute the global statistics, e.g. global
    maximum is needed when call max method. A aggregator or comparator is
    expected here for global sum or extreme value respectively.

    Attributes:
        partitions: a dict of pyu and partition.
        aggregator: the aggagator for computing global values such as mean.
        comparator: the comparator for computing global values such as
            maximum/minimum.

    Examples:
        >>> from secretflow.data.horizontal import read_csv
        >>> from secretflow.security.aggregation import PlainAggregator, PlainComparator
        >>> from secretflow import PYU
        >>> alice = PYU('alice')
        >>> bob = PYU('bob')
        >>> h_df = read_csv({alice: 'alice.csv', bob: 'bob.csv'},
                            aggregator=PlainAggregagor(alice),
                            comparator=PlainComparator(alice))
        >>> h_df.columns
        Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'], dtype='object')
        >>> h_df.mean(numeric_only=True)
        sepal_length    5.827693
        sepal_width     3.054000
        petal_length    3.730000
        petal_width     1.198667
        dtype: float64
        >>> h_df.min(numeric_only=True)
        sepal_length    4.3
        sepal_width     2.0
        petal_length    1.0
        petal_width     0.1
        dtype: float64
        >>> h_df.max(numeric_only=True)
        sepal_length    7.9
        sepal_width     4.4
        petal_length    6.9
        petal_width     2.5
        dtype: float64
        >>> h_df.count()
        sepal_length    130
        sepal_width     150
        petal_length    120
        petal_width     150
        class           150
        dtype: int64
        >>> h_df.fillna({'sepal_length': 2})
    """

    partitions: Dict[PYU, Partition] = field(default_factory=dict)
    aggregator: Aggregator = None
    comparator: Comparator = None

    def _check_parts(self):
        assert self.partitions, 'Partitions in the dataframe is None or empty.'

    def mean(self, *args, **kwargs) -> pd.Series:
        """
        Return the mean of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.mean`.

        Returns:
            pd.Series
        """
        assert self.aggregator is not None, 'Aggregator should be provided for mean.'
        means = [part.mean(*args, **kwargs) for part in self.partitions.values()]
        if 'numeric_only' in kwargs:
            numeric_only = kwargs['numeric_only']
        cnts = [
            part.count(numeric_only=numeric_only) for part in self.partitions.values()
        ]
        return pd.Series(
            reveal(
                self.aggregator.average(
                    [m.values for m in means],
                    axis=0,
                    weights=[cnt.values for cnt in cnts],
                )
            ),
            index=means[0].index,
            dtype=np.float64,
        )

    # TODO(@zhouaihui)：Whether min/max, etc. shall be publicly revealed needs further discussion.
    # If not reveal, then the result shall be stored in somewhere.
    @reveal
    def min(self, *args, **kwargs) -> pd.Series:
        """
        Return the min of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.min`.

        Returns:
            pd.Series
        """
        assert self.comparator is not None, 'Compartor should be provided for min.'
        mins = [part.min(*args, **kwargs) for part in self.partitions.values()]
        return pd.Series(
            reveal(self.comparator.min([m.values for m in mins], axis=0)),
            index=mins[0].index,
            dtype=np.float64,
        )

    @reveal
    def max(self, *args, **kwargs) -> pd.Series:
        """
        Return the max of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.max`.

        Returns:
            pd.Series
        """
        assert self.comparator is not None, 'Compartor should be provided for min.'
        maxs = [part.max(*args, **kwargs) for part in self.partitions.values()]
        return pd.Series(
            reveal(self.comparator.max([m.values for m in maxs], axis=0)),
            index=maxs[0].index,
            dtype=np.float64,
        )

    def sum(self, *args, **kwargs) -> pd.Series:
        """
        Return the sum of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.sum`.

        Returns:
            pd.Series
        """
        assert self.aggregator is not None, 'Aggregator should be provided for sum.'
        sums = [part.sum(*args, **kwargs) for part in self.partitions.values()]
        return pd.Series(
            reveal(self.aggregator.sum([s.values for s in sums], axis=0)),
            index=sums[0].index,
        )

    def count(self, *args, **kwargs) -> pd.Series:
        """Count non-NA cells for each column or row.

        All arguments are same with :py:meth:`pandas.DataFrame.count`.

        Returns:
            pd.Series
        """
        assert self.aggregator is not None, 'Aggregator should be provided for count.'
        cnts = [part.count(*args, **kwargs) for part in self.partitions.values()]
        return pd.Series(
            reveal(self.aggregator.sum([cnt.values for cnt in cnts], axis=0)),
            index=cnts[0].index,
            dtype=np.int64,
        )

    def isna(self) -> 'HDataFrame':
        """Detects missing values for an array-like object.
        Same as pandas.DataFrame.isna
        Returns
            DataFrame: Mask of bool values for each element in DataFrame
                 that indicates whether an element is an NA value.

        Returns:
            VDataFrame

        Reference:
            pd.DataFrame.isna
        """
        return HDataFrame(
            {pyu: part.isna() for pyu, part in self.partitions.items()},
            aggregator=self.aggregator,
            comparator=self.comparator,
        )

    # TODO(zoupeicheng.zpc): Schedule to implement horizontal and mix case functionality.
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

    def replace(self, *args, **kwargs):
        raise NotImplementedError

    def mode(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def values(self) -> FedNdarray:
        """
        Return a federated Numpy representation of the DataFrame.

        Returns:
            FedNdarray.
        """
        return FedNdarray(
            partitions={pyu: part.values for pyu, part in self.partitions.items()},
            partition_way=PartitionWay.HORIZONTAL,
        )

    @property
    def dtypes(self) -> pd.Series:
        """
        Return the dtypes in the DataFrame.

        Returns:
            pd.Series: the data type of each column.
        """
        self._check_parts()
        return list(self.partitions.values())[0].dtypes

    def astype(self, dtype, copy: bool = True, errors: str = "raise"):
        """
        Cast object to a specified dtype ``dtype``.

        All args are same as :py:meth:`pandas.DataFrame.astype`.
        """
        return HDataFrame(
            partitions={
                pyu: part.astype(
                    dtype=dtype,
                    copy=copy,
                    errors=errors,
                )
                for pyu, part in self.partitions.items()
            },
            aggregator=self.aggregator,
            comparator=self.comparator,
        )

    @property
    def columns(self):
        """
        The column labels of the DataFrame.
        """
        self._check_parts()
        return list(self.partitions.values())[0].columns

    @property
    def shape(self):
        """Return a tuple representing the dimensionality of the DataFrame."""
        self._check_parts()
        shapes = [part.shape for part in self.partitions.values()]
        return (sum([shape[0] for shape in shapes]), shapes[0][1])

    @reveal
    def partition_shape(self):
        """Return shapes of each partition.

        Returns:
            a dict of {pyu: shape}
        """
        return {
            device: partition.shape for device, partition in self.partitions.items()
        }

    def copy(self) -> 'HDataFrame':
        """
        Shallow copy of this dataframe.

        Returns:
            HDataFrame.
        """
        return HDataFrame(
            partitions={pyu: part.copy() for pyu, part in self.partitions.items()},
            aggregator=self.aggregator,
            comparator=self.comparator,
        )

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors='raise',
    ) -> Union['HDataFrame', None]:
        """Drop specified labels from rows or columns.

        All arguments are same with :py:meth:`pandas.DataFrame.drop`.

        Returns:
            HDataFrame without the removed index or column labels
            or None if inplace=True.
        """
        if inplace:
            for part in self.partitions.values():
                part.drop(
                    labels=labels,
                    axis=axis,
                    index=index,
                    columns=columns,
                    level=level,
                    inplace=inplace,
                    errors=errors,
                )
        else:
            return HDataFrame(
                partitions={
                    pyu: part.drop(
                        labels=labels,
                        axis=axis,
                        index=index,
                        columns=columns,
                        level=level,
                        inplace=inplace,
                        errors=errors,
                    )
                    for pyu, part in self.partitions.items()
                },
                aggregator=self.aggregator,
                comparator=self.comparator,
            )

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union['HDataFrame', None]:
        """Fill NA/NaN values using the specified method.

        All arguments are same with :py:meth:`pandas.DataFrame.fillna`.

        Returns:
            HDataFrame with missing values filled or None if inplace=True.
        """
        if inplace:
            for part in self.partitions.values():
                part.fillna(
                    value=value,
                    method=method,
                    axis=axis,
                    inplace=inplace,
                    limit=limit,
                    downcast=downcast,
                )
        else:
            return HDataFrame(
                partitions={
                    pyu: part.fillna(
                        value=value,
                        method=method,
                        axis=axis,
                        inplace=inplace,
                        limit=limit,
                        downcast=downcast,
                    )
                    for pyu, part in self.partitions.items()
                },
                aggregator=self.aggregator,
                comparator=self.comparator,
            )

    def to_csv(self, fileuris: Dict[PYU, str], **kwargs):
        """Write object to a comma-separated values (csv) file.

        Args:
            fileuris: a dict of file uris specifying file for each PYU.
            kwargs: other arguments are same with :py:meth:`pandas.DataFrame.to_csv`.

        Returns:
            Returns a list of PYUObjects whose value is none. You can use
            `secretflow.wait` to wait for the save to complete.
        """
        for device, uri in fileuris.items():
            if device not in self.partitions:
                raise InvalidArgumentError(f'PYU {device} is not in this dataframe.')

        return [
            self.partitions[device].to_csv(uri, **kwargs)
            for device, uri in fileuris.items()
        ]

    def __len__(self):
        return sum([len(part) for part in self.partitions.values()])

    def __getitem__(self, item) -> 'HDataFrame':
        return HDataFrame(
            aggregator=self.aggregator,
            comparator=self.comparator,
            partitions={pyu: part[item] for pyu, part in self.partitions.items()},
        )

    def __setitem__(self, key, value):
        if isinstance(value, HDataFrame):
            assert len(value.partitions) == len(self.partitions), (
                'Length of HDataFrame to assign not equals to this dataframe: '
                f'{len(value.partitions)} != {len(self.partitions)}'
            )
            for pyu, part in value.partitions.items():
                self.partitions[pyu][key] = part
        elif isinstance(value, Partition):
            assert (
                value.data.device in self.partitions
            ), f'Partition to assgin is not in this dataframe pyu list.'
            self.partitions[value.data.device][key] = value
        else:
            for part in self.partitions.values():
                part[key] = value
