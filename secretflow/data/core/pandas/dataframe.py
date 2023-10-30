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

from pathlib import Path
from typing import Callable, List, Union

import pandas as pd
from jax import tree_map
from pandas import Index
from pandas._typing import IgnoreRaise

from ...io.util import is_local_file
from ..base import PartDataFrameBase


class PdPartDataFrame(PartDataFrameBase):
    data: pd.DataFrame

    def __init__(self, data: pd.DataFrame):
        super().__init__()
        assert isinstance(
            data, (pd.DataFrame, pd.Series)
        ), f"need pd.DataFrame/pd.Series, got {type(data)}"
        self.data = data.copy(deep=True)

    def get_data(self):
        return self.data

    def __unwrap(self, args, kwargs):
        new_args = tree_map(lambda x: x.data if (type(x) == type(self)) else x, args)
        new_kwargs = tree_map(
            lambda x: x.data if (type(x) == type(self)) else x, kwargs
        )
        return new_args, new_kwargs

    def __getitem__(self, item) -> "PdPartDataFrame":
        item_list = item

        if not isinstance(item, (list, tuple, Index)):
            item_list = [item_list]
        return PdPartDataFrame(self.data.__getitem__(item_list))

    def __setitem__(self, key, value):
        if isinstance(value, PdPartDataFrame):
            value = value.get_data()

        self.data.__setitem__(key, value)

    def __len__(self):
        return self.data.__len__()

    def columns(self) -> list:
        return self.data.columns.tolist()

    def dtypes(self) -> dict:
        return self.data.dtypes.to_dict()

    def shape(self) -> tuple:
        return self.data.shape

    def index(self) -> list:
        return self.data.index

    def count(self, *args, **kwargs) -> pd.Series:
        return self.data.count(*args, **kwargs)

    def sum(self, *args, **kwargs) -> pd.Series:
        return self.data.sum(*args, **kwargs)

    def min(self, *args, **kwargs) -> pd.Series:
        return self.data.min(*args, **kwargs)

    def max(self, *args, **kwargs) -> pd.Series:
        return self.data.max(*args, **kwargs)

    def mean(self, *args, **kwargs) -> pd.Series:
        return self.data.mean(*args, **kwargs)

    def var(self, *args, **kwargs) -> pd.Series:
        return self.data.var(*args, **kwargs)

    def std(self, *args, **kwargs) -> pd.Series:
        return self.data.std(*args, **kwargs)

    def sem(self, *args, **kwargs) -> pd.Series:
        return self.data.sem(*args, **kwargs)

    def skew(self, *args, **kwargs) -> pd.Series:
        return self.data.skew(*args, **kwargs)

    def kurtosis(self, *args, **kwargs) -> pd.Series:
        return self.data.kurtosis(*args, **kwargs)

    def quantile(self, q=0.5, axis=0, **kwargs) -> pd.Series:
        return self.data.quantile(q=q, axis=axis, **kwargs)

    def mode(self, *args, **kwargs) -> pd.Series:
        return self.data.mode(*args, **kwargs).iloc[0, :]

    def value_counts(self, *args, **kwargs) -> pd.Series:
        return self.data.value_counts(*args, **kwargs)

    def values(self):
        return self.data.values

    def isna(self) -> "PdPartDataFrame":
        return PdPartDataFrame(self.data.isna())

    def replace(self, *args, **kwargs) -> "PdPartDataFrame":
        return PdPartDataFrame(self.data.replace(*args, **kwargs))

    def astype(
        self, dtype, copy: bool = True, errors: str = "raise"
    ) -> "PdPartDataFrame":
        return PdPartDataFrame(self.data.astype(dtype, copy, errors))

    def copy(self) -> "PdPartDataFrame":
        return PdPartDataFrame(self.data.copy(False))

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors: IgnoreRaise = 'raise',
    ) -> "PdPartDataFrame":
        data = self.data.drop(
            labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )
        if not inplace:
            return PdPartDataFrame(data)

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union['PdPartDataFrame', None]:
        data = self.data.fillna(
            value,
            method=method,
            axis=axis,
            inplace=inplace,
            limit=limit,
            downcast=downcast,
        )
        if not inplace:
            return PdPartDataFrame(data)

    def to_csv(self, filepath, **kwargs):
        if is_local_file(filepath):
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(filepath, **kwargs)

    def iloc(self, index: Union[int, slice, List[int]]) -> Union['PdPartDataFrame']:
        return PdPartDataFrame(self.data.iloc[index])

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
        errors: IgnoreRaise = 'ignore',
    ) -> Union['PdPartDataFrame', None]:
        data = self.data.rename(
            mapper,
            index=index,
            columns=columns,
            axis=axis,
            copy=copy,
            inplace=inplace,
            level=level,
            errors=errors,
        )
        if not inplace:
            return PdPartDataFrame(data)

    def pow(self, *args, **kwargs) -> 'PdPartDataFrame':
        new_args, new_kwargs = self.__unwrap(args, kwargs)

        return PdPartDataFrame(self.data.__pow__(*new_args, **new_kwargs))

    def round(self, *args, **kwargs) -> 'PdPartDataFrame':
        new_args, new_kwargs = self.__unwrap(args, kwargs)

        return PdPartDataFrame(self.data.round(*new_args, **new_kwargs))

    def select_dtypes(self, *args, **kwargs) -> 'PdPartDataFrame':
        return PdPartDataFrame(self.data.select_dtypes(*args, **kwargs))

    def subtract(self, *args, **kwargs) -> 'PdPartDataFrame':
        new_args, new_kwargs = self.__unwrap(args, kwargs)

        return PdPartDataFrame(self.data.__sub__(*new_args, **new_kwargs))

    def apply_func(
        self, func: Callable, *, nums_return: int = 1, **kwargs
    ) -> Union['PdPartDataFrame', 'List[PdPartDataFrame]']:
        dfs = func(self.data, **kwargs)
        if nums_return != 1:
            assert isinstance(dfs, tuple) and len(dfs) == nums_return
            return [PdPartDataFrame(df) for df in dfs]
        else:
            return PdPartDataFrame(dfs)

    def to_pandas(self) -> 'PdPartDataFrame':
        raise RuntimeError(
            "Should not got here, since it will loss performance, try to_pandas outside."
        )
