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

import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
import polars as pl
import polars.selectors as cs
from pandas.core.dtypes.inference import is_list_like

from ...io.util import is_local_file
from ..base import PartDataFrameBase
from ..pandas import PdPartDataFrame
from .util import infer_pd_dtype, infer_pl_dtype


class PlPartDataFrame(PartDataFrameBase):
    df: pl.DataFrame
    use_item: Optional[List[str]]

    def __init__(self, df: pl.DataFrame, use_item=None):
        super().__init__()
        self.df = df.clone()
        self.use_item = use_item

    def set_data(self, data: pl.DataFrame):
        assert isinstance(data, pl.DataFrame)
        self.df = data

    def get_data(self):
        self._collect()
        return self.df

    def _collect(self):
        """compute result if it is lazy mode"""
        if self.use_item is not None:
            self.df = self.df.select(pl.col(self.use_item))
            self.use_item = None

    def __getitem__(self, item) -> "PlPartDataFrame":
        item_list = item
        if not isinstance(item, (list, tuple)):
            item_list = [item_list]
        lazy_df = PlPartDataFrame(self.df)
        if self.use_item is not None:
            assert all([it in self.use_item for it in item_list])
        lazy_df.use_item = item_list

        return lazy_df

    def __setitem__(self, key, value):
        self._collect()
        if isinstance(value, PlPartDataFrame):
            if value.use_item is not None:
                if isinstance(key, str):
                    key = [key]
                assert len(key) == len(
                    value.use_item
                ), f"expect key len({len(key)}) == value.expr len({len(value.use_item)})"
                if value.df is self.df:
                    self.df = self.df.with_columns(
                        [
                            pl.col(value.use_item[i]).alias(key[i])
                            for i in range(len(value.use_item))
                        ]
                    )
                    return
            value = value.get_data()
        if isinstance(key, str):
            if isinstance(value, pl.DataFrame):
                self.__setitem__([key], value)
            elif isinstance(value, pl.Series):
                self.df = self.df.with_columns(value.alias(key))
            elif is_list_like(value):
                self.df = self.df.with_columns(
                    pl.Series(name=key, value=value).alias(key)
                )
            else:
                # is a str or int to fullfill all values in this columns
                self.df = self.df.with_columns(
                    pl.Series(
                        name=key, values=[value for i in range(len(self.df))]
                    ).alias(key)
                )
        elif isinstance(key, list):
            if isinstance(value, pl.DataFrame):
                assert (
                    isinstance(key, list) and len(key) == value.shape[1]
                ), f"input value len({value.shape[1]}) does not equal with len of key({len(key)})"
                self.df = self.df.with_columns(
                    [value_series.alias(key[i]) for i, value_series in enumerate(value)]
                )
            elif isinstance(value, pl.Series):
                assert (
                    len(key) == 1
                ), f"to set a pl.Series, key must be str or ['xxx'] with len = 1"
                self.df = self.df.with_columns(value.alias(key[0]))
            elif not is_list_like(value):
                # is a str or int to fullfill all values in this columns
                self.df = self.df.with_columns(
                    [
                        pl.Series(
                            name=k, values=[value for i in range(len(self.df))]
                        ).alias(k)
                        for k in key
                    ]
                )
            else:
                self.df.__setitem__(key, value)
        else:
            self.df.__setitem__(key, value)

    def __len__(self):
        self._collect()
        return self.df.__len__()

    def columns(self) -> list:
        self._collect()
        return self.df.columns

    def dtypes(self) -> dict:
        self._collect()
        schema = self.df.schema
        return {k: infer_pd_dtype(v) for k, v in schema.items()}

    def shape(self) -> tuple:
        self._collect()
        return self.df.shape

    def index(self) -> list:
        raise NotImplementedError("polars does not support index.")

    @staticmethod
    def __pd_series_convert(df: pl.DataFrame) -> pd.Series:
        """
        For compatibility with Pandas when execute statistic function.
        Take 'max()' as an example, in pandas, it returns a pd.Series.
        However, in polars, 'max()' returns a pl.DataFrame, which has 1 row for each colum's result.
        We need to convert pl.DataFrame to pd.DataFrame, and then convert a row to pd.Series.
        In addition, pd.Series converted from polars has a default name, while the result in Pandas do not.
        So we set it to None insted.
        Args:
            df: the statistic result polars DataFrame with 1 row.

        Returns:
            a pd.Series type result.
        """
        series = df.to_pandas().iloc[0, :]
        series.name = None
        return series

    def count(self, *args, **kwargs) -> pd.Series:
        raise NotImplementedError()

    def sum(self, *args, **kwargs) -> pd.Series:
        self._collect()
        axis = kwargs.get('axis', 0)
        null_strategy = kwargs.get('null_strategy', 'ignore')
        numeric_only = kwargs.get('numeric_only', False)
        if numeric_only:
            return self.__pd_series_convert(
                self.df.select(pl.col(pl.NUMERIC_DTYPES)).sum(
                    axis=axis, null_strategy=null_strategy
                )
            )
        else:
            return self.__pd_series_convert(
                self.df.sum(axis=axis, null_strategy=null_strategy)
            )

    def min(self, *args, **kwargs) -> pd.Series:
        self._collect()
        axis = kwargs.get('axis', 0)
        numeric_only = kwargs.get('numeric_only', False)
        if numeric_only:
            return self.__pd_series_convert(
                self.df.select(pl.col(pl.NUMERIC_DTYPES)).min(axis=axis)
            )
        else:
            return self.__pd_series_convert(self.df.min(axis=axis))

    def max(self, *args, **kwargs) -> pd.Series:
        self._collect()
        axis = kwargs.get('axis', 0)
        numeric_only = kwargs.get('numeric_only', False)
        if numeric_only:
            return self.__pd_series_convert(
                self.df.select(pl.col(pl.NUMERIC_DTYPES)).max(axis=axis)
            )
        else:
            return self.__pd_series_convert(self.df.max(axis=axis))

    def mean(self, *args, **kwargs) -> pd.Series:
        self._collect()
        axis = kwargs.get('axis', 0)
        null_strategy = kwargs.get('null_strategy', 'ignore')
        numeric_only = kwargs.get('numeric_only', False)
        if numeric_only:
            return self.__pd_series_convert(
                self.df.select(pl.col(pl.NUMERIC_DTYPES)).mean(
                    axis=axis, null_strategy=null_strategy
                )
            )
        else:
            return self.__pd_series_convert(
                self.df.mean(axis=axis, null_strategy=null_strategy)
            )

    def var(self, *args, **kwargs) -> pd.Series:
        self._collect()
        ddof = kwargs.get('ddof', 1)
        numeric_only = kwargs.get('numeric_only', False)
        if numeric_only:
            return self.__pd_series_convert(
                self.df.select(pl.col(pl.NUMERIC_DTYPES)).var(ddof=ddof)
            )
        else:
            return self.__pd_series_convert(self.df.var(ddof=ddof))

    def std(self, *args, **kwargs) -> pd.Series:
        self._collect()
        ddof = kwargs.get('ddof', 1)

        numeric_only = kwargs.get('numeric_only', False)
        if numeric_only:
            return self.__pd_series_convert(
                self.df.select(pl.col(pl.NUMERIC_DTYPES)).std(ddof=ddof)
            )
        else:
            return self.__pd_series_convert(self.df.std(ddof=ddof))

    def sem(self, *args, **kwargs) -> pd.Series:
        raise NotImplementedError()

    def skew(self, *args, **kwargs) -> pd.Series:
        raise NotImplementedError()

    def kurtosis(self, *args, **kwargs) -> pd.Series:
        raise NotImplementedError()

    def quantile(self, *args, **kwargs) -> pd.Series:
        self._collect()
        assert len(args) == 0, f"please use keyword arguments to input q and axis"
        q = kwargs.get('q', 0.5)
        interpolation = kwargs.get('interpolation', 'linear')
        numeric_only = kwargs.get('numeric_only', False)
        if numeric_only:
            ret = self.__pd_series_convert(
                self.df.select(pl.col(pl.NUMERIC_DTYPES)).quantile(
                    quantile=q, interpolation=interpolation
                )
            )
        else:
            ret = self.__pd_series_convert(
                self.df.quantile(quantile=q, interpolation=interpolation)
            )
        ret.name = q
        return ret

    def mode(self, *args, **kwargs) -> pd.Series:
        raise NotImplementedError()

    def value_counts(self, *args, **kwargs) -> pd.Series:
        raise NotImplementedError()

    def values(self):
        raise NotImplementedError()

    def isna(self) -> "PlPartDataFrame":
        raise NotImplementedError()

    def replace(self, *args, **kwargs) -> "PlPartDataFrame":
        raise NotImplementedError("'replace' in polars is different with it in pandas.")

    def astype(
        self, dtype, copy: bool = True, errors: str = "raise"
    ) -> "PlPartDataFrame":
        self._collect()
        exprs = []
        if isinstance(dtype, dict):
            for col in dtype:
                exprs.append(pl.col(col).cast(infer_pl_dtype(dtype[col])))
        else:
            exprs.append(pl.col("*").cast(infer_pl_dtype(dtype)))
        new_data = self.df.with_columns(*exprs)
        if copy:
            return PlPartDataFrame(new_data)
        else:
            # In this actor coly = false since not work.
            return self

    def copy(self) -> "PlPartDataFrame":
        self._collect()
        cp = self.df.__copy__()
        return PlPartDataFrame(cp)

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors='raise',
    ) -> "PlPartDataFrame":
        self._collect()
        new_data = self.df.drop(columns=columns)
        if not inplace:
            return PlPartDataFrame(new_data)
        else:
            self.df = new_data

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ) -> Union['PlPartDataFrame', None]:
        self._collect()
        logging.warning(
            "polars' fillna (actually pl.DataFrame.fill_null) does not act same as pandas, col whose tpye diffrent "
            "with value will not be filled."
        )
        new_data = self.df.fill_null(value=value)
        if not inplace:
            return PlPartDataFrame(new_data)
        else:
            self.df = new_data

    def to_csv(self, filepath, **kwargs):
        self._collect()
        if is_local_file(filepath):
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.df.write_csv(filepath)

    def iloc(self, index: Union[int, slice, List[int]]) -> 'PlPartDataFrame':
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
    ) -> Union['PlPartDataFrame', None]:
        self._collect()
        if index is not None:
            logging.warning(
                "Polars dataframe only support rename column names, index parameters will be ignored."
            )
        if axis is not None and axis != 1:
            logging.warning(
                f"Polars dataframe only support axis = 0 and rename columns names. Got axis = {axis}"
            )
        if columns is None and mapper is not None:
            columns = mapper
        new_data = self.df.rename(columns)
        if not inplace:
            return PlPartDataFrame(new_data)

    def pow(self, *args, **kwargs) -> 'PlPartDataFrame':
        raise NotImplementedError()

    def round(self, decimals) -> 'PlPartDataFrame':
        """

        Args:
            decimals: An int or dict decimals.

        Returns:
            round value.

        Note:
            Polars round's working principle is differnt from Pandas.
            It works more like round (2.4-> 2, 2.5 -> 3, 2.6 -> 3).
            But in Pandas, it will be rounded to Even Numbers (2.4 -> 2, 2.5 -> 2, 2.6 -> 3).
        """
        self._collect()
        df = None
        if isinstance(decimals, dict):
            for col, decimal in decimals.items():
                df = self.df.with_columns(pl.col(col).round(decimals=decimal))
        else:
            df = self.df.with_columns(
                pl.col(pl.FLOAT_DTYPES).round(decimals=decimals),
            )
        return PlPartDataFrame(df)

    def select_dtypes(self, include=None, exclude=None) -> 'PlPartDataFrame':
        self._collect()
        assert exclude is None, f"does not support exclude dtypes in polars yet."
        assert include is not None, f"inxlude must indicate"
        if include == 'number':
            include = pl.NUMERIC_DTYPES
        else:
            if not isinstance(include, (list, tuple)):
                include = [include]
            include = [infer_pl_dtype(icld) for icld in include]
        return PlPartDataFrame(self.df.select(cs.by_dtype(include)))

    def subtract(self, *args, **kwargs) -> 'PlPartDataFrame':
        raise NotImplementedError()

    def apply_func(
        self, func: Callable, *, nums_return: int = 1, **kwargs
    ) -> Union['PlPartDataFrame', 'List[PlPartDataFrame]']:
        self._collect()
        dfs = func(self.df, **kwargs)
        if nums_return != 1:
            assert isinstance(dfs, tuple) and len(dfs) == nums_return
            for df in dfs:
                assert isinstance(df, (pl.DataFrame, pd.DataFrame))
            return [
                PlPartDataFrame(df)
                if isinstance(df, pl.DataFrame)
                else PdPartDataFrame(df)
                for df in dfs
            ]
        else:
            assert isinstance(
                dfs, (pl.DataFrame, pd.DataFrame)
            ), f"need DataFrame, got {type(dfs)}"
            return (
                PlPartDataFrame(dfs)
                if isinstance(dfs, pl.DataFrame)
                else PdPartDataFrame(dfs)
            )

    def to_pandas(self) -> 'PdPartDataFrame':
        self._collect()
        return PdPartDataFrame(self.df.to_pandas())
