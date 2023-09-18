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
from pathlib import Path
from typing import Union, List, Callable

import pandas as pd
from jax.tree_util import tree_map
from pandas.core.indexes.base import Index

from secretflow.device import PYUObject, reveal
from ...base import PartitionBase
from ...io.util import is_local_file


class PdPartition(PartitionBase):
    """Slice of data that makes up horizontal, vertical and mixed partitioned DataFrame.

    Attributes:
        data (PYUObject): Reference to pandas.DataFrame located in local node.
    """

    def __init__(self, data: PYUObject = None):
        super().__init__(data, backend="pandas")

    def __partition_wrapper(self, fn: Callable, *args, **kwargs) -> 'PartitionBase':
        return PdPartition(self.data.device(fn)(self.data, *args, **kwargs))

    def mean(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the mean of the values over the requested axis.

        Returns:
            PartitionBase: mean values series.
        """
        return self.__partition_wrapper(pd.DataFrame.mean, *args, **kwargs)

    def var(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the variance of the values over the requested axis.

        Returns:
            PartitionBase: variance values series.
        """
        return self.__partition_wrapper(pd.DataFrame.var, *args, **kwargs)

    def std(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the standard deviation of the values over the requested axis.

        Returns:
            PartitionBase: standard deviation values series.

        """
        return self.__partition_wrapper(pd.DataFrame.std, *args, **kwargs)

    def sem(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the standard error of the mean over the requested axis.

        Returns:
            PartitionBase: standard error of the mean series.

        """
        return self.__partition_wrapper(pd.DataFrame.sem, *args, **kwargs)

    def skew(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the skewness over the requested axis.

        Returns:
            PartitionBase: skewness series.

        """
        return self.__partition_wrapper(pd.DataFrame.skew, *args, **kwargs)

    def kurtosis(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the kurtosis over the requested axis.

        Returns:
            PartitionBase: kurtosis series.

        """
        return self.__partition_wrapper(pd.DataFrame.kurtosis, *args, **kwargs)

    def sum(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the sum of the values over the requested axis.

        Returns:
            PartitionBase: sum values series.
        """
        return self.__partition_wrapper(pd.DataFrame.sum, *args, **kwargs)

    def replace(self, *args, **kwargs) -> 'PartitionBase':
        """Replace values given in to_replace with value.
        Same as pandas.DataFrame.replace
        Values of the DataFrame are replaced with other values dynamically.

        Returns:
            PartitionBase: same shape except value replaced
        """
        return self.__partition_wrapper(pd.DataFrame.replace, *args, **kwargs)

    def quantile(self, q=0.5, axis=0) -> 'PartitionBase':
        """Returns values at the given quantile over requested axis.

        Returns:
            PartitionBase: quantile values series.
        """
        # q is between 0 and 1
        q = max(min(q, 1), 0)
        # limit q to one of 0, 0.25, 0.5, 0.75 and 1
        q = round(4 * q) / 4
        return self.__partition_wrapper(pd.DataFrame.quantile, q=q, axis=axis)

    def min(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the minimum of the values over the requested axis.

        Returns:
            PartitionBase: minimum values series.
        """
        return self.__partition_wrapper(pd.DataFrame.min, *args, **kwargs)

    def mode(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the mode of the values over the requested axis.

        For data protection reasons, only one mode will be returned.

        Returns:
            PartitionBase: mode values series.
        """

        def _mode(*_args, **_kwargs):
            return pd.DataFrame.mode(*_args, **_kwargs).iloc[0, :]

        return self.__partition_wrapper(_mode, *args, **kwargs)

    def max(self, *args, **kwargs) -> 'PartitionBase':
        """Returns the maximum of the values over the requested axis.

        Returns:
            PartitionBase: maximum values series.
        """
        return self.__partition_wrapper(pd.DataFrame.max, *args, **kwargs)

    def count(self, *args, **kwargs) -> 'PartitionBase':
        """Counts non-NA cells for each column or row.

        Returns:
            PartitionBase: count values series.
        """
        return self.__partition_wrapper(pd.DataFrame.count, *args, **kwargs)

    def isna(self) -> 'PartitionBase':
        """Detects missing values for an array-like object.
        Same as pandas.DataFrame.isna
        Returns
            DataFrame: Mask of bool values for each element in DataFrame
                 that indicates whether an element is an NA value.
        """
        return self.__partition_wrapper(pd.DataFrame.isna)

    def __unwrap(self, args, kwargs):
        new_args = tree_map(lambda x: x.data if (type(x) == type(self)) else x, args)
        new_kwargs = tree_map(
            lambda x: x.data if (type(x) == type(self)) else x, kwargs
        )
        return new_args, new_kwargs

    def pow(self, *args, **kwargs) -> 'PartitionBase':
        """Gets Exponential power of (partition of) dataframe and other, element-wise (binary operator pow).
        Equivalent to dataframe ** other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rpow.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.

        Reference:
            pd.DataFrame.pow
        """
        new_args, new_kwargs = self.__unwrap(args, kwargs)

        return self.__partition_wrapper(pd.DataFrame.pow, *new_args, **new_kwargs)

    def subtract(self, *args, **kwargs) -> 'PartitionBase':
        """Gets Subtraction of (partition of) dataframe and other, element-wise (binary operator sub).
        Equivalent to dataframe - other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rsub.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.

        Reference:
            pd.DataFrame.subtract
        """
        new_args, new_kwargs = self.__unwrap(args, kwargs)

        return self.__partition_wrapper(pd.DataFrame.subtract, *new_args, **new_kwargs)

    def round(self, *args, **kwargs) -> 'PartitionBase':
        """Round the (partition of) DataFrame to a variable number of decimal places.

        Reference:
            pd.DataFrame.round
        """
        new_args, new_kwargs = self.__unwrap(args, kwargs)

        return self.__partition_wrapper(pd.DataFrame.round, *new_args, **new_kwargs)

    def select_dtypes(self, *args, **kwargs) -> 'PartitionBase':
        """Returns a subset of the DataFrame's columns based on the column dtypes.

        Reference:
            pandas.DataFrame.select_dtypes
        """
        return self.__partition_wrapper(pd.DataFrame.select_dtypes, *args, **kwargs)

    @property
    def values(self):
        """Returns the underlying ndarray."""
        return self.data.device(lambda df: df.values)(self.data)

    @property
    @reveal
    def index(self):
        """Returns the index (row labels) of the DataFrame."""
        return self.data.device(lambda df: df.index)(self.data)

    @property
    @reveal
    def dtypes(self):
        """Returns the dtypes in the DataFrame."""
        # return dict always.
        return self.data.device(
            lambda df: df.dtypes.to_dict()
            if isinstance(df, pd.DataFrame)
            else {df.name: df.types}
        )(self.data)

    def astype(self, dtype, copy: bool = True, errors: str = "raise"):
        """
        Cast a pandas object to a specified dtype ``dtype``.

        All args are same as :py:meth:`pandas.DataFrame.astype`.
        """
        return PdPartition(
            self.data.device(pd.DataFrame.astype)(self.data, dtype, copy, errors)
        )

    @property
    @reveal
    def columns(self):
        """Returns the column labels of the DataFrame."""
        return self.data.device(lambda df: df.columns.to_list())(self.data)

    @property
    @reveal
    def shape(self):
        """Returns a tuple representing the dimensionality of the DataFrame."""
        return self.data.device(lambda df: df.shape)(self.data)

    def iloc(self, index: Union[int, slice, List[int]]) -> 'PartitionBase':
        """Integer-location based indexing for selection by position.

        Args:
            index (Union[int, slice, List[int]]): rows index.

        Returns:
            PartitionBase: Selected DataFrame.
        """
        return PdPartition(
            self.data.device(lambda df, idx: df.iloc[idx])(self.data, index)
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
    ) -> Union['PartitionBase', None]:
        """See `pandas.DataFrame.drop`"""

        def _drop(df: pd.DataFrame, **kwargs):
            if inplace:
                new_df = df.copy(deep=True)
                new_df.drop(**kwargs)
                return new_df
            else:
                return df.drop(**kwargs)

        new_data = self.data.device(_drop)(
            self.data,
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )
        if inplace:
            self.data = new_data
        else:
            return PdPartition(new_data)

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union['PartitionBase', None]:
        """See :py:meth:`pandas.DataFrame.fillna`"""

        def _fillna(df: pd.DataFrame, **kwargs):
            if inplace:
                new_df = df.copy(deep=True)
                new_df.fillna(**kwargs)
                return new_df
            else:
                return df.fillna(**kwargs)

        new_data = self.data.device(_fillna)(
            self.data,
            value=value,
            method=method,
            axis=axis,
            inplace=inplace,
            limit=limit,
            downcast=downcast,
        )
        if inplace:
            self.data = new_data
            return self
        else:
            return PdPartition(new_data)

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
    ) -> Union['PartitionBase', None]:
        """See :py:meth:`pandas.DataFrame.rename`"""

        def _rename(df: pd.DataFrame, **kwargs):
            if inplace:
                new_df = df.copy(deep=True)
                new_df.rename(**kwargs)
                return new_df
            else:
                return df.rename(**kwargs)

        new_data = self.data.device(_rename)(
            self.data,
            mapper=mapper,
            index=index,
            columns=columns,
            axis=axis,
            copy=copy,
            inplace=inplace,
            level=level,
            errors=errors,
        )
        if inplace:
            self.data = new_data
        else:
            return PdPartition(new_data)

    def value_counts(self, *args, **kwargs) -> 'PartitionBase':
        """Return a Series containing counts of unique values."""
        return self.__partition_wrapper(pd.DataFrame.value_counts, *args, **kwargs)

    def to_csv(self, filepath, **kwargs):
        """Save DataFrame to csv file."""

        def _to_csv_wrapper(df: pd.DataFrame, path, **_kwargs):
            if is_local_file(path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(open(path, 'wb'), **_kwargs)

        return self.data.device(_to_csv_wrapper)(self.data, filepath, **kwargs)

    @reveal
    def __len__(self):
        """Returns the number of rows."""
        return self.data.device(lambda df: len(df))(self.data)

    def __getitem__(self, item: Union[str, List[str]]) -> 'PartitionBase':
        """Get columns from DataFrame.

        NOTE: Always return DataFrame even if specify single column.

        Args:
            item (Union[str, List[str]]): Columns to get.

        Returns:
            PartitionBase: DataFrame.
        """
        item_list = item
        if not isinstance(item, (list, tuple, Index)):
            item_list = [item_list]
        return self.__partition_wrapper(pd.DataFrame.__getitem__, item_list)

    def __setitem__(self, key, value):
        """Assign values to columns.

        Args:
            key (Union[str, List[str]]): columns to be assigned.
            value (PartitionBase): assigned values.
        """
        if isinstance(value, PartitionBase):
            assert (
                self.data.device == value.data.device
            ), f'Can not assign a partition with different device.'

        def _setitem(df: pd.DataFrame, k, v):
            # Deep copy DataFrame since ray object store is immutable.
            df = df.copy(deep=True)
            df[k] = v
            return df

        self.data = self.data.device(_setitem)(
            self.data, key, value if not isinstance(value, PdPartition) else value.data
        )

    def copy(self):
        """Shallow copy."""
        return PdPartition(self.data)

    def to_pandas(self):
        return self
