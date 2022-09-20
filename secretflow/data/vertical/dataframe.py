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
from typing import Dict, List, Union

import pandas as pd
from pandas.core.indexes.base import Index
from secretflow.data.base import DataFrameBase, Partition
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.device import PYU, Device, reveal
from secretflow.utils.errors import InvalidArgumentError, NotFoundError


@dataclass
class VDataFrame(DataFrameBase):
    """Federated dataframe holds `vertical` partitioned data.

    This dataframe is design to provide a federated pandas dataframe
    and just same as using pandas. The original data is still stored
    locally in the data holder and is not transmitted out of the domain
    during all the methods execution.

    The method with a prefix `partition_` will return a dict
    {pyu of partition: result of partition}.

    Attributes:
        partitions: a dict of pyu and partition.
        aligned: a boolean indicating whether the data is

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

    partitions: Dict[PYU, Partition]
    aligned: bool = True

    def _check_parts(self):
        assert self.partitions, 'Partitions in the dataframe is None or empty.'

    def min(self, *args, **kwargs) -> pd.Series:
        """
        Return the min of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.min`.

        Returns:
            pd.Series
        """
        return pd.concat(
            reveal(
                [part.min(*args, **kwargs).data for part in self.partitions.values()]
            )
        )

    def max(self, *args, **kwargs) -> pd.Series:
        """
        Return the max of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.max`.

        Returns:
            pd.Series
        """
        return pd.concat(
            reveal(
                [part.max(*args, **kwargs).data for part in self.partitions.values()]
            )
        )

    @property
    def dtypes(self) -> pd.Series:
        """
        Return the dtypes in the DataFrame.

        Returns:
            pd.Series: the data type of each column.
        """
        return pd.concat([part.dtypes for part in self.partitions.values()])

    def astype(self, dtype, copy: bool = True, errors: str = "raise"):
        """
        Cast object to a specified dtype ``dtype``.

        All args are same as :py:meth:`pandas.DataFrame.astype`.
        """
        if isinstance(dtype, dict):
            item_index = self._col_index(list(dtype.keys()))
            new_parts = {}
            for pyu, part in self.partitions.items():
                if pyu not in item_index:
                    new_parts[pyu] = part.copy()
                else:
                    cols = item_index[pyu]
                    if not isinstance(cols, list):
                        cols = [cols]
                    new_parts[pyu] = part.astype(
                        dtype={col: dtype[col] for col in cols},
                        copy=copy,
                        errors=errors,
                    )
            return VDataFrame(partitions=new_parts, aligned=self.aligned)

        return VDataFrame(
            partitions={
                pyu: part.astype(dtype, copy, errors)
                for pyu, part in self.partitions.items()
            },
            aligned=self.aligned,
        )

    @property
    def columns(self):
        """
        The column labels of the DataFrame.
        """
        self._check_parts()
        cols = None
        for part in self.partitions.values():
            if cols is None:
                cols = part.columns
            else:
                cols = cols.append(part.columns)
        return cols

    @property
    def shape(self):
        """Return a tuple representing the dimensionality of the DataFrame."""
        self._check_parts()
        shapes = [part.shape for part in self.partitions.values()]
        return (shapes[0][0], sum([shape[1] for shape in shapes]))

    def mean(self, *args, **kwargs) -> pd.Series:
        """
        Return the mean of the values over the requested axis.

        All arguments are same with :py:meth:`pandas.DataFrame.mean`.

        Returns:
            pd.Series
        """
        return pd.concat(
            reveal(
                [part.mean(*args, **kwargs).data for part in self.partitions.values()]
            )
        )

    def count(self, *args, **kwargs) -> pd.Series:
        """Count non-NA cells for each column or row.

        All arguments are same with :py:meth:`pandas.DataFrame.count`.

        Returns:
            pd.Series
        """
        return pd.concat(
            reveal(
                [part.count(*args, **kwargs).data for part in self.partitions.values()]
            )
        )

    @property
    def values(self):
        """
        Return a federated Numpy representation of the DataFrame.

        Returns:
            FedNdarray.
        """
        return FedNdarray(
            partitions={pyu: part.values for pyu, part in self.partitions.items()},
            partition_way=PartitionWay.VERTICAL,
        )

    def copy(self) -> 'VDataFrame':
        """
        Shallow copy of this dataframe.

        Returns:
            VDataFrame.
        """
        return VDataFrame(
            partitions={pyu: part.copy() for pyu, part in self.partitions.items()},
            aligned=self.aligned,
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
    ) -> Union['VDataFrame', None]:
        """Drop specified labels from rows or columns.

        All arguments are same with :py:meth:`pandas.DataFrame.drop`.

        Returns:
            VDataFrame without the removed index or column labels
            or None if inplace=True.
        """
        if columns:
            col_index = self._col_index(columns)
            if inplace:
                for pyu, col in col_index.items():
                    self.partitions[pyu].drop(
                        labels=labels,
                        axis=axis,
                        index=index,
                        columns=col,
                        level=level,
                        inplace=inplace,
                        errors=errors,
                    )
            else:
                new_parts = self.partitions.copy()
                for pyu, col in col_index.items():
                    new_parts[pyu] = self.partitions[pyu].drop(
                        labels=labels,
                        axis=axis,
                        index=index,
                        columns=col,
                        level=level,
                        inplace=inplace,
                        errors=errors,
                    )
                return VDataFrame(partitions=new_parts, aligned=self.aligned)

        else:
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
                return VDataFrame(
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
                    aligned=self.aligned,
                )

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union['VDataFrame', None]:
        """Fill NA/NaN values using the specified method.

        All arguments are same with :py:meth:`pandas.DataFrame.fillna`.

        Returns:
            VDataFrame with missing values filled or None if inplace=True.
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
            return VDataFrame(
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
                aligned=self.aligned,
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
        """Return the max length if not aligned."""
        parts = list(self.partitions.values())
        assert parts, 'No partitions in vdataframe.'
        return max([len(part) for part in parts])

    def _col_index(self, col) -> Dict[Device, Union[str, List[str]]]:
        assert (
            col.tolist() if isinstance(col, Index) else col
        ), f'Column to index is None or empty!'
        pyu_col = {}
        listed_col = col.tolist() if isinstance(col, Index) else col
        if not isinstance(listed_col, (list, tuple)):
            listed_col = [listed_col]
        for key in listed_col:
            found = False
            for pyu, part in self.partitions.items():
                if key not in part.dtypes:
                    continue

                found = True
                if pyu not in pyu_col:
                    pyu_col[pyu] = key
                else:
                    if not isinstance(pyu_col[pyu], list):
                        # Convert to list if more than one column.
                        pyu_col[pyu] = [pyu_col[pyu]]
                    pyu_col[pyu].append(key)

                break

            if not found:
                raise NotFoundError(f'Item {key} does not exist.')
        return pyu_col

    def __getitem__(self, item) -> 'VDataFrame':
        item_index = self._col_index(item)
        return VDataFrame(
            partitions={
                pyu: self.partitions[pyu][keys] for pyu, keys in item_index.items()
            }
        )

    def __setitem__(self, key, value):
        if isinstance(value, Partition):
            assert (
                value.data.device in self.partitions
            ), 'Device of the partition to assgin is not in this dataframe devices.'
            self.partitions[value.data.device][key] = value
            return
        elif isinstance(value, VDataFrame):
            for pyu in value.partitions.keys():
                assert (
                    pyu in self.partitions
                ), 'Partitions to assgin is not same with this dataframe partitions.'
            try:
                key_index = self._col_index(key)
                for pyu, col in key_index.items():
                    self.partitions[pyu][col] = (
                        value.partitions[pyu]
                        if isinstance(value, VDataFrame)
                        else value
                    )
            except NotFoundError:
                # Insert as a new key if not seen.
                for pyu, part in value.partitions.items():
                    self.partitions[pyu][part.dtypes.index] = part
        else:
            key_index = self._col_index(key)
            for pyu, col in key_index.items():
                self.partitions[pyu][col] = (
                    value.partitions[pyu] if isinstance(value, VDataFrame) else value
                )

    @reveal
    def partition_shape(self):
        """Return shapes of each partition.

        Returns:
            a dict of {pyu: shape}
        """
        return {
            device: partition.shape for device, partition in self.partitions.items()
        }

    @property
    def partition_columns(self):
        """Returns columns of each partition.

        Returns:
            a dict of {pyu: columns}
        """
        assert len(self.partitions) > 0, 'Partitions in the dataframe is None or empty.'
        return {
            device: partition.columns for device, partition in self.partitions.items()
        }
