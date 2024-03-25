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

import numpy as np
import polars as pl
from polars.datatypes import DataTypeClass


def read_polars_csv(filepath, *args, **kwargs):
    if 'delimiter' in kwargs and kwargs['delimiter'] is not None:
        kwargs['separator'] = kwargs.pop('delimiter')
    if 'usecols' in kwargs and kwargs['usecols'] is not None:
        # polars only recognized list columns but not dictkeys.
        kwargs['columns'] = list(kwargs.pop('usecols'))
    if 'dtype' in kwargs and kwargs['dtype'] is not None:
        pl_dtypes = {}
        for col, dt in kwargs.pop('dtype').items():
            pl_dtypes[col] = infer_pl_dtype(dt)
        kwargs['dtypes'] = pl_dtypes
    if 'nrows' in kwargs:
        kwargs['n_rows'] = kwargs.pop('nrows')
    if 'header' in kwargs and kwargs['header'] is None:
        # no header from pandas
        kwargs['has_header'] = False

    kwargs.pop('delimiter', None)
    kwargs.pop('usecols', None)
    kwargs.pop('dtype', None)
    kwargs.pop('header', None)

    skiprows = kwargs.pop('skip_rows_after_header', None)
    if skiprows is not None:
        assert isinstance(skiprows, int)
        kwargs['skip_rows_after_header'] = skiprows
        try:
            df = pl.read_csv(filepath, *args, **kwargs)
        except pl.NoDataError:
            # skip ending with empty df, not exception
            df = pl.DataFrame()
    else:
        df = pl.read_csv(filepath, *args, **kwargs)

    if len(df.columns) == 1:
        # for compatibility of pandas, single columns will drop null when read.
        df = df.drop_nulls()
    if 'columns' in kwargs and kwargs['columns'] is not None:
        return df[kwargs['columns']]
    else:
        return df


def infer_pl_dtype(tp):
    if isinstance(tp, str):
        relation = {
            "float32": pl.Float32,
            "float64": pl.Float64,
            "int8": pl.Int8,
            "int16": pl.Int16,
            "int32": pl.Int32,
            "int64": pl.Int64,
            "bool": pl.Boolean,
            "str": str,
            "string": str,
        }
        if tp in relation:
            return relation[tp]
        raise RuntimeError(f"Cannot infer polars dtype with str {tp}")

    np_relation = {
        np.float32: pl.Float32,
        np.float64: pl.Float64,
        np.int8: pl.Int8,
        np.int16: pl.Int16,
        np.int32: pl.Int32,
        np.int64: pl.Int64,
        np.bool8: pl.Boolean,
        np.string_: str,
        np.str0: str,
        np.str: str,
    }
    if tp in np_relation:
        return np_relation[tp]
    return tp


def infer_pd_dtype(tp: DataTypeClass):
    assert isinstance(tp, DataTypeClass)
    np_relation = {
        pl.Float32: np.float32,
        pl.Float64: np.float64,
        pl.Int8: np.int8,
        pl.Int16: np.int16,
        pl.Int32: np.int32,
        pl.Int64: np.int64,
        pl.Boolean: np.bool8,
        pl.Utf8: np.string_,
        pl.Object: np.object_,
    }
    if tp in np_relation:
        return np_relation[tp]
    logging.warning(f"Cannot infer padas dtype with input tp: {tp}")
    return tp
