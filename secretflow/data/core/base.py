# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np
import pandas as pd

from ...device import PYUObject
from ..base import DataFrameBase


class AgentIndex:
    def __init__(self, idx: int):
        self.idx = idx

    def __eq__(self, other: 'AgentIndex'):
        return self.idx == other.idx

    def __hash__(self):
        return self.idx.__hash__()

    def __str__(self):
        return f"AgentIndex: {self.idx}"

    def __repr__(self):
        return f"AgentIndex: {self.idx}"


class PartitionAgentBase:
    """
    Abstract PartDataFrame Base class.
    The implement of this class should add @proxy(PYUObject) as an actor.
    Therefore, all function who returns something will return PYUObject instead.
    """

    @abstractmethod
    def append_data(
        self, source: Union[Callable, 'AnyDataFrame'], backend="pandas", **kwargs
    ) -> AgentIndex:
        """
        Append data or construct data into this remote agent.
        Args:
            source: the source to construct part dataframe
                It can be a Callable func type which takes kwargs as parameters and retures a real dataframe.
                or, it can be any dataframe (like pd.DataFrame).
            backend: The partition backend, default to pandas.
            **kwargs: if the source is a function, use kwargs as its parameters.

        Returns:
            AgentIndex of the generated dataframe.
        """
        pass

    @abstractmethod
    def get_data(self, idx: AgentIndex) -> PYUObject:
        """
        Get the real data of the idx inside the partition agent.
        Args:
            idx: the data's index

        Returns:
            the real dataframe such as pd.DataFrame.
        """
        pass

    @abstractmethod
    def del_object(self, idx: PYUObject):
        pass

    @abstractmethod
    def get_backend(self, idx: AgentIndex) -> PYUObject:
        pass

    @abstractmethod
    def __getitem__(self, idx: AgentIndex, item) -> PYUObject:
        """Get columns"""
        pass

    @abstractmethod
    def __setitem__(self, idx: AgentIndex, key, value: Union[AgentIndex, PYUObject]):
        """Set columns"""
        pass

    @abstractmethod
    def __len__(self, idx: AgentIndex) -> PYUObject:
        pass

    @abstractmethod
    def columns(self, idx: AgentIndex) -> PYUObject:
        """Returns the column labels of the DataFrame."""
        pass

    @abstractmethod
    def dtypes(self, idx: AgentIndex) -> PYUObject:
        """Returns the dtypes in the DataFrame."""
        # return series always.
        pass

    @abstractmethod
    def shape(self, idx: AgentIndex) -> PYUObject:
        """Returns a tuple representing the dimensionality of the DataFrame."""
        pass

    @abstractmethod
    def index(self, idx: AgentIndex) -> PYUObject:
        """Returns the index (row labels) of the DataFrame."""
        pass

    @abstractmethod
    def count(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Count non-NA cells for each column."""
        pass

    @abstractmethod
    def sum(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns the sum of the values over the requested axis."""
        pass

    @abstractmethod
    def min(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Gets minimum value of all columns"""
        pass

    @abstractmethod
    def max(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Gets maximum value of all columns"""
        pass

    @abstractmethod
    def mean(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns the mean of the values over the requested axis."""
        pass

    @abstractmethod
    def var(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns the variance of the values over the requested axis."""
        pass

    @abstractmethod
    def std(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns the standard deviation of the values over the requested axis."""
        pass

    @abstractmethod
    def sem(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns the standard error of the mean over the requested axis."""
        pass

    @abstractmethod
    def skew(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns the skewness over the requested axis."""
        pass

    @abstractmethod
    def kurtosis(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns the kurtosis over the requested axis."""
        pass

    @abstractmethod
    def quantile(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns values at the given quantile over requested axis."""
        pass

    @abstractmethod
    def mode(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns the mode of the values over the requested axis.

        For data protection reasons, only one mode will be returned.
        """

        pass

    @abstractmethod
    def value_counts(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Return a Series containing counts of unique values."""
        pass

    @abstractmethod
    def values(self, idx: AgentIndex) -> PYUObject:
        """Get underlying ndarray"""
        pass

    @abstractmethod
    def isna(self, idx: AgentIndex) -> PYUObject:
        """Detects missing values for an array-like object.
        Same as pandas.DataFrame.isna
        Returns
             Mask of bool values for each element in DataFrame
                 that indicates whether an element is an NA value.
        """
        pass

    @abstractmethod
    def replace(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Replace values given in to_replace with value.
        Same as pandas.DataFrame.replace
        Values of the DataFrame are replaced with other values dynamically.
        """
        pass

    @abstractmethod
    def astype(
        self, idx: AgentIndex, dtype, copy: bool = True, errors: str = "raise"
    ) -> PYUObject:
        """
        Cast a pandas object to a specified dtype ``dtype``.

        All args are same as :py:meth:`pandas.DataFrame.astype`.
        """
        pass

    @abstractmethod
    def copy(self, idx: AgentIndex) -> PYUObject:
        """Shallow copy."""
        pass

    @abstractmethod
    def drop(
        self,
        idx: AgentIndex,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors='raise',
    ) -> PYUObject:
        pass

    @abstractmethod
    def fillna(
        self,
        idx: AgentIndex,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union[PYUObject, None]:
        pass

    @abstractmethod
    def to_csv(self, idx: AgentIndex, filepath, **kwargs):
        """Save DataFrame to csv file."""
        pass

    @abstractmethod
    def iloc(self, idx: AgentIndex, index: Union[int, slice, List[int]]) -> PYUObject:
        """Integer-location based indexing for selection by position.

        Args:
            idx: data index in agent.
            index (Union[int, slice, List[int]]): rows index.

        Returns:
            Selected DataFrame.
        """
        pass

    @abstractmethod
    def rename(
        self,
        idx: AgentIndex,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
        errors='ignore',
    ) -> Union[PYUObject, None]:
        """See :py:meth:`pandas.DataFrame.rename`"""

        pass

    @abstractmethod
    def pow(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Gets Exponential power of (partition of) dataframe and other, element-wise (binary operator pow).
        Equivalent to dataframe ** other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rpow.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        pass

    @abstractmethod
    def round(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Round the (partition of) DataFrame to a variable number of decimal places."""
        pass

    @abstractmethod
    def select_dtypes(self, idx: AgentIndex, *args, **kwargs) -> PYUObject:
        """Returns a subset of the DataFrame's columns based on the column dtypes."""
        pass

    @abstractmethod
    def subtract(self, idx: AgentIndex, other, *args, **kwargs) -> PYUObject:
        """Gets Subtraction of (partition of) dataframe and other, element-wise (binary operator sub).
        Equivalent to dataframe - other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rsub.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        pass

    @abstractmethod
    def apply_func(
        self, idx: AgentIndex, func: Callable, *, nums_return: int = 1, **kwargs
    ) -> PYUObject:
        """
        Apply any function inside the dataframe actor.
        Please make sure the function retures a dataframe type same as the type of the self.data
        Args:
            idx: the agent index to indicate which data to apply func.
            func: any function, with first argument must be dataframe itself.
            nums_return: nums to return, default to 1.
            kwargs: contains the dataframe executed by func, the dataframe should be real df like pandas.
        Returns:
            A Partition with data applyed by this function.
        """
        pass

    @abstractmethod
    def to_pandas(self, idx: AgentIndex) -> PYUObject:
        """
        Convert myself to pandas type.
        Returns:
            A DataFrameBase within pandas type.
        """
        pass


class PartDataFrameBase(DataFrameBase, ABC):
    @abstractmethod
    def get_data(self) -> Union["pd.DataFrame", "pl.DataFrame"]:
        pass

    @abstractmethod
    def __getitem__(self, item) -> "PartDataFrameBase":
        """Get columns"""
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        """Set columns"""
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def columns(self) -> list:
        """Returns the column labels of the DataFrame."""
        pass

    @abstractmethod
    def dtypes(self) -> dict:
        """Returns the dtypes in the DataFrame."""
        # return series always.
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
    def isna(self) -> "PartDataFrameBase":
        """Detects missing values for an array-like object.
        Same as pandas.DataFrame.isna
        Returns
             Mask of bool values for each element in DataFrame
                 that indicates whether an element is an NA value.
        """
        pass

    @abstractmethod
    def replace(self, *args, **kwargs) -> "PartDataFrameBase":
        """Replace values given in to_replace with value.
        Same as pandas.DataFrame.replace
        Values of the DataFrame are replaced with other values dynamically.
        """
        pass

    @abstractmethod
    def astype(
        self, dtype, copy: bool = True, errors: str = "raise"
    ) -> "PartDataFrameBase":
        """
        Cast a pandas object to a specified dtype ``dtype``.

        All args are same as :py:meth:`pandas.DataFrame.astype`.
        """
        pass

    @abstractmethod
    def copy(self) -> "PartDataFrameBase":
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
    ) -> "PartDataFrameBase":
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
    ) -> Union['PartDataFrameBase', None]:
        pass

    @abstractmethod
    def to_csv(self, filepath, **kwargs):
        """Save DataFrame to csv file."""
        pass

    @abstractmethod
    def iloc(self, index: Union[int, slice, List[int]]) -> 'PartDataFrameBase':
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
    ) -> Union['PartDataFrameBase', None]:
        """See :py:meth:`pandas.DataFrame.rename`"""

        pass

    @abstractmethod
    def pow(self, *args, **kwargs) -> 'PartDataFrameBase':
        """Gets Exponential power of (partition of) dataframe and other, element-wise (binary operator pow).
        Equivalent to dataframe ** other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rpow.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        pass

    @abstractmethod
    def round(self, *args, **kwargs) -> 'PartDataFrameBase':
        """Round the (partition of) DataFrame to a variable number of decimal places."""
        pass

    @abstractmethod
    def select_dtypes(self, *args, **kwargs) -> 'PartDataFrameBase':
        """Returns a subset of the DataFrame's columns based on the column dtypes."""
        pass

    @abstractmethod
    def subtract(self, *args, **kwargs) -> 'PartDataFrameBase':
        """Gets Subtraction of (partition of) dataframe and other, element-wise (binary operator sub).
        Equivalent to dataframe - other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rsub.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        pass

    @abstractmethod
    def apply_func(
        self, func: Callable, *, nums_return: int = 1, **kwargs
    ) -> 'PartDataFrameBase':
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
    def to_pandas(self) -> 'PartDataFrameBase':
        """
        Convert myself to pandas type.
        Returns:
            A DataFrameBase within pandas type.
        """
        pass
