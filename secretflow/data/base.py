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
from typing import Callable, List, Union

import numpy as np
import pandas as pd


class DataFrameBase(ABC):
    """Abstract base class for:
    - API Level: horizontal, vertical and mixed partitioned DataFrame;
    - Partition Level: Partition and Partition agent;
    - Partitin implement Level: PdPartDataFrame or other implements.
    """

    @abstractmethod
    def __getitem__(self, item) -> "DataFrameBase":
        """Get items from indicate columns"""
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        """Set item from indicate columns"""
        pass

    @abstractmethod
    def __len__(self):
        """Get len of the dataframe"""
        pass

    @abstractmethod
    def columns(self) -> list:
        """Returns the column names of the DataFrame."""
        pass

    @abstractmethod
    def dtypes(self) -> dict:
        """Returns the dtypes in the DataFrame.
        Returns dict always.
        """
        pass

    @abstractmethod
    def shape(self) -> tuple:
        """Returns a tuple representing the dimensionality of the DataFrame."""
        pass

    @abstractmethod
    def index(self) -> list:
        """Returns the index (row labels) of the DataFrame."""
        pass

    @abstractmethod
    def count(self, *args, **kwargs) -> pd.Series:
        """Count non-NA cells for each column."""
        pass

    @abstractmethod
    def sum(self, *args, **kwargs) -> pd.Series:
        """Returns the sum of the values over the requested axis."""
        pass

    @abstractmethod
    def min(self, *args, **kwargs) -> pd.Series:
        """Gets minimum value of all columns"""
        pass

    @abstractmethod
    def max(self, *args, **kwargs) -> pd.Series:
        """Gets maximum value of all columns"""
        pass

    @abstractmethod
    def mean(self, *args, **kwargs) -> pd.Series:
        """Returns the mean of the values over the requested axis."""
        pass

    @abstractmethod
    def var(self, *args, **kwargs) -> pd.Series:
        """Returns the variance of the values over the requested axis."""
        pass

    @abstractmethod
    def std(self, *args, **kwargs) -> pd.Series:
        """Returns the standard deviation of the values over the requested axis."""
        pass

    @abstractmethod
    def sem(self, *args, **kwargs) -> pd.Series:
        """Returns the standard error of the mean over the requested axis."""
        pass

    @abstractmethod
    def skew(self, *args, **kwargs) -> pd.Series:
        """Returns the skewness over the requested axis."""
        pass

    @abstractmethod
    def kurtosis(self, *args, **kwargs) -> pd.Series:
        """Returns the kurtosis over the requested axis."""
        pass

    @abstractmethod
    def quantile(self, *args, **kwargs) -> pd.Series:
        """Returns values at the given quantile over requested axis."""
        pass

    @abstractmethod
    def mode(self, *args, **kwargs) -> pd.Series:
        """Returns the mode of the values over the requested axis.

        For data protection reasons, only one mode will be returned.
        """

        pass

    @abstractmethod
    def value_counts(self, *args, **kwargs) -> pd.Series:
        """Return a Series containing counts of unique values."""
        pass

    @abstractmethod
    def values(self) -> np.ndarray:
        """Get underlying ndarray"""
        pass

    @abstractmethod
    def isna(self) -> "DataFrameBase":
        """Detects missing values for an array-like object.
        Same as pandas.DataFrame.isna
        Returns
             Mask of bool values for each element in DataFrame
                 that indicates whether an element is an NA value.
        """
        pass

    @abstractmethod
    def replace(self, *args, **kwargs) -> "DataFrameBase":
        """Replace values given in to_replace with value.
        Same as pandas.DataFrame.replace
        Values of the DataFrame are replaced with other values dynamically.
        """
        pass

    @abstractmethod
    def astype(
        self, dtype, copy: bool = True, errors: str = "raise"
    ) -> "DataFrameBase":
        """
        Cast a pandas object to a specified dtype ``dtype``.

        All args are same as :py:meth:`pandas.DataFrame.astype`.
        """
        pass

    @abstractmethod
    def copy(self) -> "DataFrameBase":
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
    ) -> "DataFrameBase":
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
    ) -> Union['DataFrameBase', None]:
        pass

    @abstractmethod
    def to_csv(self, filepath, **kwargs):
        """Save DataFrame to csv file."""
        pass

    @abstractmethod
    def iloc(self, index: Union[int, slice, List[int]]) -> 'DataFrameBase':
        """Integer-location based indexing for selection by position.

        Args:
            index (Union[int, slice, List[int]]): rows index.

        Returns:
            DataFrameBase: Selected DataFrame.
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
    ) -> Union['DataFrameBase', None]:
        """See :py:meth:`pandas.DataFrame.rename`"""

        pass

    @abstractmethod
    def pow(self, *args, **kwargs) -> 'DataFrameBase':
        """Gets Exponential power of (partition of) dataframe and other, element-wise (binary operator pow).
        Equivalent to dataframe ** other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rpow.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        pass

    @abstractmethod
    def round(self, *args, **kwargs) -> 'DataFrameBase':
        """Round the (partition of) DataFrame to a variable number of decimal places."""
        pass

    @abstractmethod
    def select_dtypes(self, *args, **kwargs) -> 'DataFrameBase':
        """Returns a subset of the DataFrame's columns based on the column dtypes."""
        pass

    @abstractmethod
    def subtract(self, *args, **kwargs) -> 'DataFrameBase':
        """Gets Subtraction of (partition of) dataframe and other, element-wise (binary operator sub).
        Equivalent to dataframe - other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rsub.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        pass

    @abstractmethod
    def apply_func(
        self, func: Callable, *, nums_return: int = 1, **kwargs
    ) -> 'DataFrameBase':
        """
        Apply any function inside the dataframe actor.
        Please make sure the function retures a dataframe type same as the type of the self.data
        Args:
            func: any function, with first argument must be dataframe itself.
            nums_return: the return nums, defualt to 1.
            kwargs: contains the dataframe executed by func, the dataframe should be real df like pandas.
        Returns:
            A Partition with data applyed by this function.
        """
        pass

    @abstractmethod
    def to_pandas(self) -> 'DataFrameBase':
        """
        Convert myself to pandas type.
        Returns:
            A DataFrameBase within pandas type.
        """
        pass


def Partition(data):
    """For compatibility of some legacy cases, which should be replaced by function partition()."""
    logging.warning(
        "Constructor 'Partition' is deperated, please use 'partition(data)' instead and import like 'from "
        "secretflow.data.partition import partition'."
    )
    from .core import partition

    return partition(data)
