# Copyright 2023 Ant Group Co., Ltd.
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
from pathlib import Path
from typing import Callable, List, Union

import polars as pl
import polars.selectors as cs
from pandas.core.indexes.base import Index

from secretflow.data.io.util import is_local_file
from secretflow.device import PYUObject
from secretflow.device import reveal
from .util import infer_pl_dtype
from ..pandas.partition import PdPartition
from ...base import PartitionBase


class PolarsPartition(PartitionBase):
    def __init__(self, data: PYUObject = None):
        super().__init__(data, backend="polars")

    def __partition_wrapper(self, fn: Callable, *args, **kwargs) -> 'PartitionBase':
        return PolarsPartition(self.data.device(fn)(self.data, *args, **kwargs))

    def __pd_stat_wrapper(self, fn: Callable, *args, **kwargs) -> 'PartitionBase':
        return PdPartition(self.data.device(fn)(self.data, *args, **kwargs))

    def mean(self, *args, **kwargs) -> 'PartitionBase':
        def _mean(df, *_args, **_kwargs):
            mean_df = df.mean(*_args, **_kwargs)
            return mean_df.to_pandas().iloc[0, :]

        return self.__pd_stat_wrapper(_mean, *args, **kwargs)

    def var(self, *args, **kwargs) -> 'PartitionBase':
        def _var(df, *_args, **_kwargs):
            var_df = df.var(*_args, **_kwargs)
            return var_df.to_pandas().iloc[0, :]

        return self.__pd_stat_wrapper(_var, *args, **kwargs)

    def std(self, *args, **kwargs) -> 'PartitionBase':
        def _std(df, *_args, **_kwargs):
            std_df = df.std(*_args, **_kwargs)
            return std_df.to_pandas().iloc[0, :]

        return self.__pd_stat_wrapper(_std, *args, **kwargs)

    def sem(self, *args, **kwargs) -> 'PartitionBase':
        raise NotImplementedError()

    def skew(self, *args, **kwargs) -> 'PartitionBase':
        raise NotImplementedError()

    def kurtosis(self, *args, **kwargs) -> 'PartitionBase':
        raise NotImplementedError()

    def sum(self, *args, **kwargs) -> 'PartitionBase':
        def _sum(df, *_args, **_kwargs):
            sum_df = df.sum(*_args, **_kwargs)
            return sum_df.to_pandas().iloc[0, :]

        return self.__pd_stat_wrapper(_sum, *args, **kwargs)

    def replace(self, *args, **kwargs) -> 'PartitionBase':
        return self.__partition_wrapper(pl.DataFrame.replace, *args, **kwargs)

    @reveal
    def quantile(self, q=0.5, axis=0) -> 'PartitionBase':
        def _quantile(df, _q, _axis):
            quantile_df = df.quantile(q=_q, axis=_axis)
            return quantile_df.to_pandas().iloc[0, :]

        return self.__pd_stat_wrapper(_quantile, q, axis)

    def min(self, *args, **kwargs) -> 'PartitionBase':
        def _min(df, *_args, **_kwargs):
            axis = 0
            if "axis" in _kwargs:
                axis = _kwargs["axis"]
            min_df = df.min(axis)
            return min_df.to_pandas().iloc[0, :]

        return self.__pd_stat_wrapper(_min, *args, **kwargs)

    def mode(self, *args, **kwargs) -> 'PartitionBase':
        raise NotImplementedError()

    def max(self, *args, **kwargs) -> 'PartitionBase':
        def _max(df, *_args, **_kwargs):
            axis = 0
            if "axis" in _kwargs:
                axis = _kwargs["axis"]
            max_df = df.max(axis)
            return max_df.to_pandas().iloc[0, :]

        return self.__pd_stat_wrapper(_max, *args, **kwargs)

    def count(self, *args, **kwargs) -> 'PartitionBase':
        raise NotImplementedError()

    def isna(self) -> 'PartitionBase':
        raise NotImplementedError()

    def pow(self, *args, **kwargs) -> 'PartitionBase':
        raise NotImplementedError()

    def subtract(self, *args, **kwargs) -> 'PartitionBase':
        raise NotImplementedError()

    def round(self, *args, **kwargs) -> 'PartitionBase':
        return PolarsPartition(
            self.data.device(pl.DataFrame.with_columns)(
                self.data,
                cs.by_dtype(pl.FLOAT_DTYPES).round(*args, **kwargs),
            )
        )

    def select_dtypes(self, *args, **kwargs) -> 'PartitionBase':
        raise NotImplementedError()

    @property
    def values(self):
        return self.to_pandas().values

    @property
    @reveal
    def index(self):
        raise NotImplementedError()

    @property
    @reveal
    def dtypes(self):
        def get_dict_dtypes(_df: pl.DataFrame):
            return {_df.columns[i]: _df.dtypes[i] for i in range(len(_df.columns))}

        return self.data.device(get_dict_dtypes)(self.data)

    def astype(self, dtype, copy: bool = True, errors: str = "raise"):
        def _cast_type(df: pl.DataFrame, _dtype, _copy):
            if _copy:
                new_df = df.clone()
            else:
                new_df = df
            exprs = []
            if isinstance(_dtype, dict):
                for col in _dtype:
                    exprs.append(pl.col(col).cast(infer_pl_dtype(_dtype[col])))
            else:
                exprs.append(pl.col("*").cast(infer_pl_dtype(_dtype)))
            new_df = new_df.with_columns(*exprs)
            return new_df

        if copy:
            return PolarsPartition(self.data.device(_cast_type)(self.data, dtype, copy))
        else:
            self.data = self.data.device(_cast_type)(self.data, dtype, copy)
            return self

    @property
    @reveal
    def columns(self):
        return self.data.device(lambda df: df.columns)(self.data)

    @property
    @reveal
    def shape(self):
        return self.data.device(lambda df: df.shape)(self.data)

    def iloc(self, index: Union[int, slice, List[int]]) -> 'PartitionBase':
        raise NotImplementedError()

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
        def _drop(df: pl.DataFrame, **kwargs):
            if inplace:
                new_df = df.clone()
                new_df.drop(**kwargs)
                return new_df
            else:
                return df.drop(**kwargs)

        new_data = self.data.device(_drop)(
            self.data,
            columns=columns,
        )
        if inplace:
            self.data = new_data
        else:
            return PolarsPartition(new_data)

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union['PartitionBase', None]:
        def _fillna(df: pl.DataFrame, **kwargs):
            if inplace:
                new_df = df.clone()
                new_df.fill_null(**kwargs)  # TODO: 明确fill_null和fill_na的区别
                return new_df
            else:
                return df.fill_null(**kwargs)

        new_data = self.data.device(_fillna)(self.data, value=value)
        if inplace:
            self.data = new_data
            return self
        else:
            return PolarsPartition(new_data)

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
        if mapper is not None:
            logging.warning("Polars dataframe does not support mapper in reaname.")
        if index is not None:
            logging.warning("Polars dataframe does not support index in reaname.")
        if axis is not None:
            logging.warning("Polars dataframe does not support index in reaname.")

        def _rename(df: pl.DataFrame, rename_col_dict):
            if inplace:
                new_df = df.clone()
                new_df.rename(rename_col_dict)
                return new_df
            else:
                return df.rename(rename_col_dict)

        new_data = self.data.device(_rename)(self.data, columns)
        if inplace:
            self.data = new_data
        else:
            return PolarsPartition(new_data)

    def value_counts(self, *args, **kwargs) -> 'PartitionBase':
        raise NotImplementedError()

    def to_csv(self, filepath, **kwargs):
        def _to_csv_wrapper(df: pl.DataFrame, path):
            if is_local_file(path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.write_csv(path)

        return self.data.device(_to_csv_wrapper)(self.data, filepath)

    @reveal
    def __len__(self):
        """Returns the number of rows."""
        return self.data.device(lambda df: len(df))(self.data)

    def __getitem__(self, item):
        item_list = item
        if not isinstance(item, (list, tuple, Index)):
            item_list = [item_list]
        return self.__partition_wrapper(pl.DataFrame.__getitem__, item_list)

    def __setitem__(self, key, value):
        if isinstance(value, PolarsPartition):
            assert (
                self.data.device == value.data.device
            ), f'Can not assign a partition with different device.'

        def _setitem(df: pl.DataFrame, ks, v):
            # Deep copy DataFrame since ray object store is immutable.
            df = df.clone()
            if isinstance(v, pl.DataFrame):
                """set a DataFrame"""
                assert ks == v.columns
                return df.with_columns(v)
            df[ks] = v
            return df

        self.data = self.data.device(_setitem)(
            self.data,
            key,
            value if not isinstance(value, PolarsPartition) else value.data,
        )

    def copy(self):
        """shallow copy"""
        return PolarsPartition(self.data)

    def to_pandas(self):
        return PdPartition(self.data.device(pl.DataFrame.to_pandas)(self.data))
