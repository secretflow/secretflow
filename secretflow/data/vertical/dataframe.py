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
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Union

import numpy as np

import pandas as pd

from pandas import Index

from secretflow.data.core import Partition
from secretflow.data.base import DataFrameBase
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.device import PYU, reveal, SPU, PYUObject
from secretflow.utils.errors import InvalidArgumentError, NotFoundError
from secretflow.data.groupby import DataFrameGroupBy


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
        >>> from secretflow.data.vertical import read_csv
        >>> from secretflow import PYU
        >>> alice = PYU('alice')
        >>> bob = PYU('bob')
        >>> v_df = read_csv({alice: 'alice.csv', bob: 'bob.csv'})
        >>> v_df.columns
        Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'], dtype='object')
        >>> v_df.mean(numeric_only=True)
        sepal_length    5.827693
        sepal_width     3.054000
        petal_length    3.730000
        petal_width     1.198667
        dtype: float64
        >>> v_df.min(numeric_only=True)
        sepal_length    4.3
        sepal_width     2.0
        petal_length    1.0
        petal_width     0.1
        dtype: float64
        >>> v_df.max(numeric_only=True)
        sepal_length    7.9
        sepal_width     4.4
        petal_length    6.9
        petal_width     2.5
        dtype: float64
        >>> v_df.count()
        sepal_length    130
        sepal_width     150
        petal_length    120
        petal_width     150
        class           150
        dtype: int64
        >>> v_df.fillna({'sepal_length': 2})
    """

    partitions: Dict[PYU, Partition]
    aligned: bool = True

    def _check_parts(self):
        assert self.partitions, 'Partitions in the VDataFrame is None or empty.'

    def __concat_apply_reveal(self, fn: Callable, *args, **kwargs) -> pd.Series:
        """Helper function to concatenate the revealed results of fn applied on each partition.

        Args:
            fn: a reflection of a Callable fucntion.
                A function that
                    takes partition with additional args and kwargs,
                    and returns a Partition object
        """
        return pd.concat(
            reveal(
                [fn(part, *args, **kwargs).data for part in self.partitions.values()]
            )
        )

    def __part_apply(self, fn, *args, **kwargs) -> 'VDataFrame':
        """Helper function to generate a new VDataFrame by applying fn on each partition.
        Args:
            fn: a reflection of a Callable fucntion.
                A function that
                takes partition with additional args and kwargs,
                    and returns a Partition object
        """

        # Note the [par.columns] is here to make sure alice does not see bob's columns.
        # and it is only effective for subtract two VDataFrames with the same partitions roles and shapes.
        return VDataFrame(
            {
                pyu: fn(part, *args, **kwargs)[part.columns]
                for pyu, part in self.partitions.items()
            },
            self.aligned,
        )

    def _col_index(self, col) -> Dict[PYU, Union[str, List[str]]]:
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
                    pyu_col[pyu] = []  # ensure the output is []
                    pyu_col[pyu].append(key)
                else:
                    if not isinstance(pyu_col[pyu], list):
                        # Convert to list if more than one column.
                        pyu_col[pyu] = [pyu_col[pyu]]
                    pyu_col[pyu].append(key)

                break

            if not found:
                raise NotFoundError(f'Item {key} does not exist.')
        return pyu_col

    def __getitem__(self, item) -> "VDataFrame":
        item_index = self._col_index(item)
        return VDataFrame(
            partitions={
                pyu: self.partitions[pyu][keys] for pyu, keys in item_index.items()
            }
        )

    def __setitem__(self, key, value):
        if isinstance(value, Partition):
            assert (
                value.device in self.partitions
            ), 'Device of the partition to assgin is not in this dataframe devices.'
            self.partitions[value.device][key] = value
            return
        elif isinstance(value, VDataFrame):
            for pyu in value.partitions.keys():
                assert (
                    pyu in self.partitions
                ), 'Partitions to assgin is not same with this dataframe partitions.'
            try:
                key_index = self._col_index(key)
                for pyu, col in key_index.items():
                    self.partitions[pyu][col] = value.partitions[pyu]
            except NotFoundError:
                # Insert as a new key if not seen.
                for pyu, part in value.partitions.items():
                    self.partitions[pyu][list(part.dtypes.keys())] = part
        else:
            key_index = self._col_index(key)
            for pyu, col in key_index.items():
                self.partitions[pyu][col] = value

    @property
    def columns(self) -> list:
        """
        The column labels of the DataFrame.
        """
        self._check_parts()
        cols = None
        for part in self.partitions.values():
            if cols is None:
                cols = part.columns
            elif isinstance(cols, list):
                cols.extend(part.columns)
            else:
                cols = cols.append(part.columns)
        return cols

    @property
    def dtypes(self) -> dict:
        """
        Return the dtypes in the DataFrame.

        Returns:
            dict: the data type of each column.
        """
        my_dtypes = {}
        for part in self.partitions.values():
            my_dtypes.update(part.dtypes)
        return my_dtypes

    @property
    def shape(self) -> tuple:
        """Return a tuple representing the dimensionality of the DataFrame."""
        self._check_parts()
        shapes = [part.shape for part in self.partitions.values()]
        return shapes[0][0], sum([shape[1] for shape in shapes])

    def count(self, numeric_only=False) -> pd.Series:
        """Count non-NA cells for each column.

        Restrict count on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.count, numeric_only=numeric_only)

    def index(self) -> list:
        raise NotImplementedError()

    def sum(self, numeric_only=False) -> pd.Series:
        """
        Return the sum of the values over the axis 0.

        Restrict sum on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.sum, numeric_only=numeric_only)

    def min(self, numeric_only=False) -> pd.Series:
        """
        Return the min of the values over the axis 0.

        Note columns containing None values are ignored. Fill before proceed.

        Restrict min on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.min, numeric_only=numeric_only)

    def max(self, numeric_only=False):
        """
        Return the max of the values over the axis 0.

        Note columns containing None values are ignored. Fill before proceed.

        Restrict max on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.max, numeric_only=numeric_only)

    def mean(self, numeric_only=False) -> pd.Series:
        """
        Return the mean of the values over the axis 0.

        Note columns containing None values are ignored. Fill before proceed.

        Restrict mean on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.mean, numeric_only=numeric_only)

    # TODO(zoupeicheng.zpc): support in HDataFrame is not scheduled yet
    # TODO(zoupeicheng.zpc): DataFrame variance currently ignore None columns.
    # However, original pandas ignore None entries!
    def var(self, numeric_only=False) -> pd.Series:
        """
        Return the var of the values over the axis 0.

        Note columns containing None values are ignored. Fill before proceed.

        Restrict var on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.var, numeric_only=numeric_only)

    # TODO(zoupeicheng.zpc): support in HDataFrame is not scheduled yet
    # TODO(zoupeicheng.zpc): DataFrame std currently ignore None columns.
    # However, original pandas ignore None entries!
    def std(self, numeric_only=False) -> pd.Series:
        """
        Return the std of the values over the axis 0.

        Note columns containing None values are ignored. Fill before proceed.

        Restrict std on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.std, numeric_only=numeric_only)

    # TODO(zoupeicheng.zpc): support in HDataFrame is not scheduled yet
    # TODO(zoupeicheng.zpc): DataFrame sem currently ignore None columns.
    # However, original pandas ignore None entries!
    def sem(self, numeric_only=False) -> pd.Series:
        """
        Return the standard error of the mean over the axis 0.

        Note columns containing None values are ignored. Fill before proceed.

        Restrict sem on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.sem, numeric_only=numeric_only)

    # TODO(zoupeicheng.zpc): support in HDataFrame is not scheduled yet
    # TODO(zoupeicheng.zpc): DataFrame skew currently ignore None columns.
    # However, original pandas ignore None entries!
    def skew(self, numeric_only=False) -> pd.Series:
        """
        Return the skewness over the axis 0.

        Note columns containing None values are ignored. Fill before proceed.

        Restrict skew on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.skew, numeric_only=numeric_only)

    # TODO(zoupeicheng.zpc): support in HDataFrame is not scheduled yet
    # TODO(zoupeicheng.zpc): DataFrame kurtosis currently ignore None columns.
    # However, original pandas ignore None entries!
    def kurtosis(self, numeric_only=False) -> pd.Series:
        """
        Return the kurtosis over the requested axis.

        Note columns containing None values are ignored. Fill before proceed.

        Restrict kurtosis on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(Partition.kurtosis, numeric_only=numeric_only)

    def quantile(self, q=0.5, numeric_only=True, **kwargs) -> pd.Series:
        """Returns values at the given quantile over axis 0.

        Note columns containing None values are ignored. Fill before proceed.

        Restrict quantile on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(
            Partition.quantile, q=q, axis=0, numeric_only=numeric_only, **kwargs
        )

    def mode(self, numeric_only=False, dropna=True) -> pd.Series:
        """
        Return the mode of the values over the axis 0.
        The mode of a set of values is the value that appears most often.
        Restrict mode on axis 0 in VDataFrame for data protection reasons.

        Returns:
            pd.Series
        """
        return self.__concat_apply_reveal(
            Partition.mode,
            numeric_only=numeric_only,
            dropna=dropna,
            axis=0,
        )

    def value_counts(self, *args, **kwargs) -> pd.Series:
        return self.__concat_apply_reveal(Partition.value_counts, *args, **kwargs)

    @property
    def values(self) -> FedNdarray:
        """
        Return a federated Numpy representation of the DataFrame.

        Returns:
            FedNdarray.
        """
        return FedNdarray(
            partitions={pyu: part.values for pyu, part in self.partitions.items()},
            partition_way=PartitionWay.VERTICAL,
        )

    def isna(self) -> "DataFrameBase":
        """ "Detects missing values for an array-like object.
        Same as pandas.DataFrame.isna
        Returns
            DataFrame: Mask of bool values for each element in DataFrame
                 that indicates whether an element is an NA value.

        Returns:
            A VDataFrame

        Reference:
            pd.DataFrame.isna
        """
        return self.__part_apply(Partition.isna)

    def replace(self, *args, **kwargs) -> "VDataFrame":
        """Replace values given in to_replace with value.
        Same as pandas.DataFrame.replace
        Values of the DataFrame are replaced with other values dynamically.

        Returns:
            VDataFrame: same shape except value replaced
        """
        return self.__part_apply(Partition.replace, *args, **kwargs)

    def astype(self, dtype, copy: bool = True, errors: str = "raise") -> "VDataFrame":
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

    def copy(self) -> 'VDataFrame':
        """
        Shallow copy of this dataframe.

        Returns:
            VDataFrame.
        """
        return self.__part_apply(Partition.copy)

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
                return self.__part_apply(
                    Partition.drop,
                    labels=labels,
                    axis=axis,
                    index=index,
                    columns=columns,
                    level=level,
                    inplace=inplace,
                    errors=errors,
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
            return self.__part_apply(
                Partition.fillna,
                value=value,
                method=method,
                axis=axis,
                inplace=inplace,
                limit=limit,
                downcast=downcast,
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

    def iloc(self, index: Union[int, slice, List[int]]) -> 'DataFrameBase':
        raise NotImplementedError()

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
        errors='ignore',
    ) -> Union['DataFrameBase', None]:
        raise NotImplementedError()

    def pow(self, *args, **kwargs) -> 'VDataFrame':
        """Gets Exponential power of dataframe and other, element-wise (binary operator pow).
        Equivalent to dataframe ** other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rpow.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.

        Returns:
            VDataFrame

        Reference:
            pd.DataFrame.pow
        """
        return self.__part_apply(Partition.pow, *args, **kwargs)

    def round(self, decimals: Union[int, dict]) -> 'VDataFrame':
        """Round the DataFrame to a variable number of decimal places.

        Returns:
            VDataFrame: same shape except value rounded

        Reference:
            pd.DataFrame.round
        """
        if isinstance(decimals, dict):
            col_idx = self._col_index(list(decimals.keys()))
            ret_part = {}
            for pyu, part in self.partitions.items():
                if pyu in col_idx:
                    cols = col_idx[pyu]
                    if not isinstance(cols, list):
                        cols = [cols]
                    ret_part[pyu] = part.round({col: decimals[col] for col in cols})
                else:
                    ret_part[pyu] = part
            return VDataFrame(
                ret_part,
                self.aligned,
            )
        else:
            return self.__part_apply(Partition.round, decimals)

    def select_dtypes(self, *args, **kwargs) -> 'VDataFrame':
        """Returns a subset of the DataFrame's columns based on the column dtypes.

        Reference:
            pandas.DataFrame.select_dtypes
        """
        return VDataFrame(
            {
                pyu: part.select_dtypes(*args, **kwargs)
                for pyu, part in self.partitions.items()
            },
            self.aligned,
        )

    def subtract(self, *args, **kwargs) -> 'VDataFrame':
        """Gets Subtraction of dataframe and other, element-wise (binary operator sub).
        Equivalent to dataframe - other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rsub.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.

        Note each part only will contain its own columns.

        Reference:
            pd.DataFrame.subtract
        """
        return self.__part_apply(Partition.subtract, *args, **kwargs)

    def apply_func(self, func, *, nums_return: int = 1, **kwargs) -> 'VDataFrame':
        return self.__part_apply(
            Partition.apply_func, nums_return=nums_return, func=func, **kwargs
        )

    @reveal
    def partition_shape(self):
        """Return shapes of each core.

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

    def __len__(self):
        """Return the max length if not aligned."""
        parts = list(self.partitions.values())
        assert parts, 'No partitions in VDataFrame.'
        return max([len(part) for part in parts])

    def to_pandas(self) -> "VDataFrame":
        return VDataFrame(
            {pyu: part.to_pandas() for pyu, part in self.partitions.items()},
            self.aligned,
        )

    def groupby(self, spu: SPU, by: List[str]) -> DataFrameGroupBy:
        """Group the VDataFrame by the specified columns.
        To groupby with string columns, use encode the string columns first.

        Args:
            spu: SPU device used to perform secure groupby.
            by: a list of column names to group by.

        Returns:
            A DataFrameGroupBy object.
        """
        _, key_cols = self.get_numeric_cols_from_df(by)
        assert len(key_cols) < len(
            self.columns
        ), "by should have length < column number"
        value_col_names_prev = [col for col in self.columns if col not in by]
        value_col_names, value_cols = self.get_numeric_cols_from_df(
            value_col_names_prev
        )

        if len(value_col_names) <= len(value_col_names_prev):
            logging.warning(
                "There are some target columns that are of non-numeric dtypes. Encode those columns if we want to include them."
            )

        # float types are not recommended to be used as key for numerical considerations.

        key_cols_spu = [key_col.to(spu) for key_col in key_cols]
        value_cols_spu = [value_col.to(spu) for value_col in value_cols]

        return DataFrameGroupBy(
            spu,
            parties=[*self.partitions.keys()],
            key_cols=key_cols_spu,
            target_cols=value_cols_spu,
            target_col_names=value_col_names,
            n_samples=len(self),
        )

    def get_numeric_cols_from_df(self, cols: List[str]) -> List[PYUObject]:
        """Get columns from VDataFrame.

        Args:
            df: a VDataFrame object.
            cols: a list of column names.

        Returns:
            first: a list of str, each of which is a name of numeric column.
            second: a list of PYUObject, each of which is a np array corresponds to col_names.
        """
        col_dtype_map = self.dtypes
        numeric_col_names = [
            col_name
            for col_name in cols
            if np.issubdtype(col_dtype_map[col_name], np.number)
        ]
        return numeric_col_names, [
            self.get_flat_column(col_name) for col_name in numeric_col_names
        ]

    def get_flat_column(self, col_name: str) -> PYUObject:
        """From the VDataFrame, extract a single column in flattened numpy array"""
        # note there will be only one party since one column belongs to only one party
        for device, column_df in self[col_name].partitions.items():
            return device(lambda x: x.to_numpy().reshape(-1))(column_df.data)
