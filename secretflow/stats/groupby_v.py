# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from secretflow.data import partition
from secretflow.data.groupby import DataFrameGroupBy
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, SPU, reveal
from secretflow.preprocessing.encoder import VOrdinalEncoder
from secretflow.utils.consistent_ops import unique_list


def ordinal_encoded_groupby(
    df: VDataFrame,
    by: List[str],
    values: List[str],
    compute_device: Union[SPU, PYU],
    max_group_size: int = None,
) -> DataFrameGroupBy:
    encoder = VOrdinalEncoder()
    df[by] = encoder.fit_transform(df[by])

    if max_group_size is not None:
        group_num = int(np.prod(df[by].max().values + 1))
        assert (
            group_num <= max_group_size
        ), f"num groups {group_num} is larger than limit {max_group_size}"
    selected_cols = by + values
    return df[selected_cols].groupby(compute_device, by), encoder


def ordinal_encoded_postprocess(
    df: VDataFrame,
    stats: Union[pd.DataFrame, pd.Series],
    encoder: VOrdinalEncoder,
    by: List[str],
    values: List[str],
):
    if isinstance(stats, pd.Series):
        stats = stats.to_frame().reset_index(names=by)
    elif isinstance(stats, pd.DataFrame):
        stats = stats.reset_index(names=by)

    v_dataframe_by = {}

    for device, cols in df[by].partition_columns.items():
        v_dataframe_by[device] = partition(data=device(lambda x: x)(stats[cols]))
    df = VDataFrame(v_dataframe_by)
    df = encoder.inverse_transform(df)
    for device, cols in df.partition_columns.items():
        stats[cols] = reveal(df.partitions[device].data)
    return stats.set_index(by)[values]


def ordinal_encoded_groupby_agg(
    df: VDataFrame,
    by: List[str],
    values: List[str],
    spu: SPU,
    agg: str,
    max_group_size: int = None,
):
    """apply ordinal encoder df before doing groupby
    df columns must be of unifrom type"""
    df_groupby, encoder = ordinal_encoded_groupby(df, by, values, spu, max_group_size)

    stat = getattr(
        df_groupby,
        agg,
    )()
    return ordinal_encoded_postprocess(df, stat, encoder, by, values)


def ordinal_encoded_groupby_value_agg_pairs(
    df: VDataFrame,
    by: List[str],
    value_agg_pairs: List[Tuple[str, str]],
    compute_device: Union[SPU, PYU],
    max_group_size: int = None,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """apply ordinal encoder df before doing groupby
    df columns must be of unifrom type"""
    values = unique_list([pair[0] for pair in value_agg_pairs])

    logging.info("ordinal_encoded_groupby begin")
    df_groupby, encoder = ordinal_encoded_groupby(
        df, by, values, compute_device, max_group_size
    )
    logging.info("ordinal_encoded_groupby complete")
    results = {}
    for value, agg in value_agg_pairs:
        stat = getattr(
            df_groupby,
            agg,
        )([value])
        logging.info(f" {value, agg} stats computed")
        results[(value, agg)] = ordinal_encoded_postprocess(
            df, stat, encoder, by, [value]
        )
        logging.info(f" {value, agg} postprocessed")
    return results


def ordinal_encoded_groupby_aggs(
    df: VDataFrame,
    by: List[str],
    values: List[str],
    compute_device: Union[SPU, PYU],
    aggs: List[str],
    max_group_size: int = None,
):
    df_groupby, encoder = ordinal_encoded_groupby(
        df, by, values, compute_device, max_group_size
    )
    results = {}
    for agg in aggs:
        stat = getattr(
            df_groupby,
            agg,
        )()

        stat = ordinal_encoded_postprocess(df, stat, encoder, by, values)
        results[agg] = stat

    return results
