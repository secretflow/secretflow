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
import polars as pl
import logging
import numpy as np


def read_polars_csv(filepath, *args, **kwargs):
    if 'delimiter' in kwargs:
        kwargs['separator'] = kwargs['delimiter']
    if 'usecols' in kwargs:
        # polars only recognized list columns but not dictkeys.
        kwargs['columns'] = list(kwargs['usecols'])
    if 'dtype' in kwargs:
        logging.warning("dtypes with np.xxx in polars backend will go wrong.")
        pl_dtypes = {}
        for col, dt in kwargs['dtype'].items():
            pl_dtypes[col] = infer_pl_dtype(dt)
        kwargs['dtypes'] = pl_dtypes
    del kwargs['delimiter'], kwargs['dtype'], kwargs['usecols']
    df = pl.read_csv(filepath, *args, **kwargs)
    if len(df.columns) == 1:
        # for compatibility of pandas, single columns will drop null when read.
        df = df.drop_nulls()
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
        np.str_: str,
        np.str: str,
    }
    if tp in np_relation:
        return np_relation[tp]
    return tp
