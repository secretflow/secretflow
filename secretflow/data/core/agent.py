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

from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd

from secretflow.device import PYUObject, proxy

from .base import AgentIndex, PartDataFrameBase, PartitionAgentBase
from .pandas import PdPartDataFrame


def partition_data(source, backend="pandas") -> "PartDataFrameBase":
    if backend == "pandas":
        return PdPartDataFrame(source)
    elif backend == "polars":
        from secretflow.data.core.polars import PlPartDataFrame

        return PlPartDataFrame(source)
    else:
        raise RuntimeError(f"Unknown backend {backend}")


@proxy(PYUObject)
class PartitionAgent(PartitionAgentBase):
    """A partition agent actor.
    In order to achieve the best performance and prevent copying PYUObject,
    Each party resides with a partition agent actor during the data process phase.
    The agent actor create in initialization (such as read_csv) and is shared among multiple partitions.
    The relation between partition and partition agent like this:
    Partition0  ...                       ...  pd.DataFrame
                     .                 .
    Partition1  ....   PartitionAgent     ...  pd.DataFrame
                     .    (actor)      .
    Partition2  ...                       ...  pd.DataFrame
    """

    working_objects: Dict[AgentIndex, PartDataFrameBase]

    def __init__(self):
        super().__init__()
        self.cur_id = 0
        self.working_objects = {}

    def append_data(
        self, source: Union[Callable, Any], backend="pandas", **kwargs
    ) -> AgentIndex:
        cur_idx = self.__next_agent_index()
        if callable(source):
            source = source(**kwargs)
        data = partition_data(source, backend)
        self.working_objects[cur_idx] = data
        return cur_idx

    def __next_agent_index(self):
        self.cur_id += 1
        return AgentIndex(self.cur_id - 1)

    def get_data(self, idx: AgentIndex):
        # returns pd.DataFrame or any dataframe backend.
        return self.working_objects[idx].get_data()

    def del_object(self, idx: AgentIndex):
        if idx in self.working_objects:
            del self.working_objects[idx]

    def get_backend(self, idx: AgentIndex) -> str:
        working_object = self.working_objects[idx]
        if isinstance(working_object, PdPartDataFrame):
            return "pandas"
        else:
            try:
                from secretflow.data.core.polars import PlPartDataFrame

                if isinstance(working_object, PlPartDataFrame):
                    return "polars"
            except ImportError:
                pass
        return "unknown"

    def __getitem__(self, idx: AgentIndex, item) -> AgentIndex:
        working_object = self.working_objects[idx]
        data = working_object.__getitem__(item)
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = data
        return cur_idx

    def __setitem__(self, idx: AgentIndex, key, value: Union[AgentIndex, PYUObject]):
        working_object = self.working_objects[idx]
        if isinstance(value, AgentIndex):
            value = self.working_objects[value]

        working_object.__setitem__(key, value)

    def __len__(self, idx: AgentIndex) -> int:
        working_object = self.working_objects[idx]
        return working_object.__len__()

    def columns(self, idx: AgentIndex) -> list:
        working_object = self.working_objects[idx]
        return working_object.columns()

    def dtypes(self, idx: AgentIndex) -> dict:
        working_object = self.working_objects[idx]

        return working_object.dtypes()

    def shape(self, idx: AgentIndex) -> tuple:
        working_object = self.working_objects[idx]
        return working_object.shape()

    def index(self, idx: AgentIndex) -> list:
        working_object = self.working_objects[idx]
        return working_object.index()

    def count(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Count non-NA cells for each column."""
        working_object = self.working_objects[idx]
        return working_object.count(*args, **kwargs)

    def sum(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Returns the sum of the values over the requested axis."""
        working_object = self.working_objects[idx]
        return working_object.sum(*args, **kwargs)

    def min(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Gets minimum value of all columns"""
        working_object = self.working_objects[idx]
        return working_object.min(*args, **kwargs)

    def max(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Gets maximum value of all columns"""
        working_object = self.working_objects[idx]
        return working_object.max(*args, **kwargs)

    def mean(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Returns the mean of the values over the requested axis."""
        working_object = self.working_objects[idx]
        return working_object.mean(*args, **kwargs)

    def var(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Returns the variance of the values over the requested axis."""
        working_object = self.working_objects[idx]
        return working_object.var(*args, **kwargs)

    def std(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Returns the standard deviation of the values over the requested axis."""
        working_object = self.working_objects[idx]
        return working_object.std(*args, **kwargs)

    def sem(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Returns the standard error of the mean over the requested axis."""
        working_object = self.working_objects[idx]
        return working_object.sem(*args, **kwargs)

    def skew(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Returns the skewness over the requested axis."""
        working_object = self.working_objects[idx]
        return working_object.skew(*args, **kwargs)

    def kurtosis(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Returns the kurtosis over the requested axis."""
        working_object = self.working_objects[idx]
        return working_object.kurtosis(*args, **kwargs)

    def quantile(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Returns values at the given quantile over requested axis."""
        working_object = self.working_objects[idx]
        return working_object.quantile(*args, **kwargs)

    def mode(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Returns the mode of the values over the requested axis.

        For data protection reasons, only one mode will be returned.
        """

        working_object = self.working_objects[idx]
        return working_object.mode(*args, **kwargs)

    def value_counts(self, idx: AgentIndex, *args, **kwargs) -> pd.Series:
        """Return a Series containing counts of unique values."""
        working_object = self.working_objects[idx]
        return working_object.value_counts(*args, **kwargs)

    def values(self, idx: AgentIndex) -> np.ndarray:
        """Get underlying ndarray"""
        working_object = self.working_objects[idx]
        return working_object.values()

    def isna(self, idx: AgentIndex) -> AgentIndex:
        """Detects missing values for an array-like object.
        Same as pandas.DataFrame.isna
        Returns
             Mask of bool values for each element in DataFrame
                 that indicates whether an element is an NA value.
        """
        working_object = self.working_objects[idx]
        data = working_object.isna()
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = data
        return cur_idx

    def replace(self, idx: AgentIndex, *args, **kwargs) -> AgentIndex:
        """Replace values given in to_replace with value.
        Same as pandas.DataFrame.replace
        Values of the DataFrame are replaced with other values dynamically.
        """
        working_object = self.working_objects[idx]
        data = working_object.replace(*args, **kwargs)
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = data
        return cur_idx

    def astype(
        self, idx: AgentIndex, dtype, copy: bool = True, errors: str = "raise"
    ) -> AgentIndex:
        """
        Cast a pandas object to a specified dtype ``dtype``.

        All args are same as :py:meth:`pandas.DataFrame.astype`.
        """
        working_object = self.working_objects[idx]
        data = working_object.astype(dtype, copy, errors)
        if copy:
            cur_idx = self.__next_agent_index()
            self.working_objects[cur_idx] = data
            return cur_idx
        else:
            return idx

    def copy(self, idx: AgentIndex) -> AgentIndex:
        """Shallow copy."""
        working_object = self.working_objects[idx]
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = working_object.copy()
        return cur_idx

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
    ) -> AgentIndex:
        working_object = self.working_objects[idx]
        data = working_object.drop(labels, axis, index, columns, level, inplace, errors)
        if not inplace:
            cur_idx = self.__next_agent_index()
            self.working_objects[cur_idx] = data
            return cur_idx
        else:
            return idx

    def fillna(
        self,
        idx: AgentIndex,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union[AgentIndex, None]:
        working_object = self.working_objects[idx]
        data = working_object.fillna(value, method, axis, inplace, limit, downcast)
        if not inplace:
            cur_idx = self.__next_agent_index()
            self.working_objects[cur_idx] = data
            return cur_idx
        else:
            return idx

    def to_csv(self, idx: AgentIndex, filepath, **kwargs):
        """Save DataFrame to csv file."""
        working_object = self.working_objects[idx]
        working_object.to_csv(filepath, **kwargs)

    def iloc(self, idx: AgentIndex, index: Union[int, slice, List[int]]) -> AgentIndex:
        working_object = self.working_objects[idx]
        data = working_object.iloc(index)
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = data
        return cur_idx

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
    ) -> Union[AgentIndex, None]:
        """See :py:meth:`pandas.DataFrame.rename`"""

        working_object = self.working_objects[idx]
        data = working_object.rename(
            mapper, index, columns, axis, copy, inplace, level, errors
        )
        if not inplace:
            cur_idx = self.__next_agent_index()
            self.working_objects[cur_idx] = data
            return cur_idx
        else:
            return idx

    def pow(self, idx: AgentIndex, *args, **kwargs) -> AgentIndex:
        """Gets Exponential power of (partition of) dataframe and other, element-wise (binary operator pow).
        Equivalent to dataframe ** other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rpow.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        working_object = self.working_objects[idx]
        data = working_object.pow(*args, **kwargs)
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = data
        return cur_idx

    def round(self, idx: AgentIndex, *args, **kwargs) -> AgentIndex:
        """Round the (partition of) DataFrame to a variable number of decimal places."""
        working_object = self.working_objects[idx]
        data = working_object.round(*args, **kwargs)
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = data
        return cur_idx

    def select_dtypes(self, idx: AgentIndex, *args, **kwargs) -> AgentIndex:
        """Returns a subset of the DataFrame's columns based on the column dtypes."""
        working_object = self.working_objects[idx]
        data = working_object.select_dtypes(*args, **kwargs)
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = data
        return cur_idx

    def subtract(self, idx: AgentIndex, other, *args, **kwargs) -> AgentIndex:
        """Gets Subtraction of (partition of) dataframe and other, element-wise (binary operator sub).
        Equivalent to dataframe - other, but with support to substitute a fill_value for missing data in one of the inputs.
        With reverse version, rsub.
        Among flexible wrappers (add, sub, mul, div, mod, pow) to arithmetic operators: +, -, *, /, //, %, **.
        """
        if isinstance(other, AgentIndex):
            other = self.working_objects[other]
        working_object = self.working_objects[idx]
        data = working_object.subtract(other, *args, **kwargs)
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = data
        return cur_idx

    def apply_func(
        self, idx: AgentIndex, func: Callable, *, nums_return: int = 1, **kwargs
    ) -> Union[AgentIndex, List[AgentIndex]]:
        working_object = self.working_objects[idx]
        data = working_object.apply_func(func, nums_return=nums_return, **kwargs)
        if nums_return != 1:
            assert isinstance(data, list) and len(data) == nums_return
            ret = []
            for d in data:
                cur_idx = self.__next_agent_index()
                self.working_objects[cur_idx] = d
                ret.append(cur_idx)
            return ret
        else:
            cur_idx = self.__next_agent_index()
            self.working_objects[cur_idx] = data
            return cur_idx

    def to_pandas(self, idx: AgentIndex) -> AgentIndex:
        """
        Convert myself to pandas type.
        Returns:
            A DataFrameBase within pandas type.
        """
        working_object = self.working_objects[idx]
        data = working_object.to_pandas()
        cur_idx = self.__next_agent_index()
        self.working_objects[cur_idx] = data
        return cur_idx
