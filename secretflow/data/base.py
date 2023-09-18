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
from abc import ABC, abstractmethod
from typing import List, Union

from jax.tree_util import tree_map

from secretflow.device import PYUObject, reveal


class DataFrameBase(ABC):
    """Abstract base class for horizontal, vertical and mixed partitioned DataFrame"""

    @abstractmethod
    def __getitem__(self, item):
        """Get columns"""
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        """Set columns"""
        pass

    @abstractmethod
    def sum(self, *args, **kwargs):
        """Returns the sum of the values over the requested axis."""
        pass

    @abstractmethod
    def min(self, *args, **kwargs):
        """Gets minimum value of all columns"""
        pass

    @abstractmethod
    def max(self, *args, **kwargs):
        """Gets maximum value of all columns"""
        pass

    @abstractmethod
    def count(self, *args, **kwargs):
        """Gets number of rows"""
        pass

    @abstractmethod
    def values(self):
        """Get underlying ndarray"""
        pass

    @abstractmethod
    def mean(self, *args, **kwargs):
        """Returns the mean of the values over the requested axis."""
        pass

    @abstractmethod
    def var(self, *args, **kwargs):
        """Returns the variance of the values over the requested axis."""
        pass

    @abstractmethod
    def std(self, *args, **kwargs):
        """Returns the standard deviation of the values over the requested axis."""
        pass

    @abstractmethod
    def sem(self, *args, **kwargs):
        """Returns the standard error of the mean over the requested axis."""
        pass

    @abstractmethod
    def skew(self, *args, **kwargs):
        """Returns the skewness over the requested axis."""
        pass

    @abstractmethod
    def kurtosis(self, *args, **kwargs):
        """Returns the kurtosis over the requested axis."""
        pass

    @abstractmethod
    def replace(self, *args, **kwargs):
        """Replace values given in to_replace with value.
        Same as pandas.DataFrame.replace
        Values of the DataFrame are replaced with other values dynamically.
        """
        pass

    @abstractmethod
    def quantile(self, q=0.5, axis=0):
        """Returns values at the given quantile over requested axis."""
        pass

    @abstractmethod
    def mode(self, *args, **kwargs):
        """Returns the mode of the values over the requested axis.

        For data protection reasons, only one mode will be returned.
        """

        pass

    @abstractmethod
    def isna(self):
        """Detects missing values for an array-like object.
        Same as pandas.DataFrame.isna
        Returns
             Mask of bool values for each element in DataFrame
                 that indicates whether an element is an NA value.
        """
        pass

    @abstractmethod
    def dtypes(self) -> dict:
        """Returns the dtypes in the DataFrame."""
        # return series always.
        pass

    @abstractmethod
    def astype(self, dtype, copy: bool = True, errors: str = "raise"):
        """
        Cast a pandas object to a specified dtype ``dtype``.

        All args are same as :py:meth:`pandas.DataFrame.astype`.
        """
        pass

    @abstractmethod
    def columns(self):
        """Returns the column labels of the DataFrame."""
        pass

    @abstractmethod
    def shape(self):
        """Returns a tuple representing the dimensionality of the DataFrame."""
        pass

    @abstractmethod
    def copy(self):
        """Shallow copy."""
        pass

    @abstractmethod
    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors='raise',
    ):
        pass

    @abstractmethod
    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union['PartitionBase', None]:
        pass

    @abstractmethod
    def to_csv(self, filepath, **kwargs):
        """Save DataFrame to csv file."""
        pass


class PartitionBase(DataFrameBase):
    """Slice of data that makes up horizontal, vertical and mixed partitioned DataFrame.

    Attributes:
        data (PYUObject): Reference to pandas.DataFrame located in local node.
    """

    def __init__(self, data: PYUObject = None, backend="pandas"):
        self.data = data
        self.backend = backend

    def __unwrap(self, args, kwargs):
        new_args = tree_map(lambda x: x.data if (type(x) == type(self)) else x, args)
        new_kwargs = tree_map(
            lambda x: x.data if (type(x) == type(self)) else x, kwargs
        )
        return new_args, new_kwargs

    @reveal
    def __len__(self):
        """Returns the number of rows."""
        return self.data.device(lambda df: len(df))(self.data)

    @abstractmethod
    def index(self):
        """Returns the index (row labels) of the DataFrame."""
        pass

    @abstractmethod
    def iloc(self, index: Union[int, slice, List[int]]) -> 'PartitionBase':
        """Integer-location based indexing for selection by position.

        Args:
            index (Union[int, slice, List[int]]): rows index.

        Returns:
            PartitionBase: Selected DataFrame.
        """
        pass

    @abstractmethod
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

        pass

    @abstractmethod
    def value_counts(self, *args, **kwargs) -> 'PartitionBase':
        """Return a Series containing counts of unique values."""
        pass

    @abstractmethod
    def pow(self, *args, **kwargs) -> 'PartitionBase':
        """Gets Exponential power of (partition of) dataframe and other, element-wise (binary operator pow).
        Equivalent to dataframe ** other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rpow.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        pass

    @abstractmethod
    def round(self, *args, **kwargs) -> 'PartitionBase':
        """Round the (partition of) DataFrame to a variable number of decimal places."""
        pass

    @abstractmethod
    def select_dtypes(self, *args, **kwargs) -> 'PartitionBase':
        """Returns a subset of the DataFrame's columns based on the column dtypes."""
        pass

    @abstractmethod
    def subtract(self, *args, **kwargs) -> 'PartitionBase':
        """Gets Subtraction of (partition of) dataframe and other, element-wise (binary operator sub).
        Equivalent to dataframe - other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rsub.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        pass

    @abstractmethod
    def to_pandas(self):
        pass


def partition(data, backend="pandas") -> PartitionBase:
    """Construct a Parititon with input data and backend.
    Default backend is Pandas, support Polars as well.
    """
    if backend == "pandas":
        from secretflow.data.partition.pandas.partition import PdPartition

        return PdPartition(data)
    elif backend == "polars":
        from secretflow.data.partition.polars.partition import PolarsPartition

        logging.warning(
            "Currently, polars is still an experimental version. Please use it with caution."
        )
        return PolarsPartition(data)
    else:
        raise RuntimeError(f"Unknown backend {backend}")


def Partition(data):
    """For compatibility of some legacy cases, which should be replaced by function partition()."""
    logging.warning(
        "Constructor 'Partition' is deperated, please use partition(data) instead."
    )
    return partition(data)
