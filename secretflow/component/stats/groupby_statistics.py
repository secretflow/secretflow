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

from typing import Dict

import pandas as pd

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import DistDataType, load_table
from secretflow.device.device.spu import SPU
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Div, Report, Tab, Table
from secretflow.stats.groupby_v import ordinal_encoded_groupby_aggs

groupby_statistics_comp = Component(
    name="groupby_statistics",
    domain="stats",
    version="0.0.2",
    desc="""Get a groupby of statistics, like pandas groupby statistics.
    Currently only support VDataframe.
    """,
)

groupby_statistics_comp.str_attr(
    name="aggs",
    desc="What kind of aggregation statistics we want to do?\
        Currently only supports min, max, mean, sum, var, count(number of elements in each group). \
        No repeatitions allowed",
    is_list=True,
    is_optional=False,
    allowed_values=["min", "max", "mean", "sum", "var", "count"],
    list_max_length_inclusive=6,
    list_min_length_inclusive=1,
)


groupby_statistics_comp.int_attr(
    name="max_group_size",
    desc="The maximum number of groups allowed",
    is_list=False,
    is_optional=True,
    default_value=10000,
    lower_bound=0,
    upper_bound=10001,
)

groupby_statistics_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input table.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="by",
            desc="by what columns should we group the values",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=4,
        ),
        TableColParam(
            name="values",
            desc="on which columns should we calculate the statistics, only numerical columns are allowed.",
            col_min_cnt_inclusive=1,
        ),
    ],
)
groupby_statistics_comp.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="Output groupby statistics report.",
    types=[DistDataType.REPORT],
    col_params=None,
)


def gen_groupby_statistic_reports(
    agg_df_dict: Dict[str, pd.DataFrame], input_data_by
) -> Report:
    r_tables = {
        agg: gen_groupby_statistic_report(df, agg, input_data_by)
        for agg, df in agg_df_dict.items()
    }
    return Report(
        name="groupby statistics",
        desc="",
        tabs=[
            Tab(
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="table",
                                table=r_table,
                            )
                        ],
                    )
                ],
                name=agg,
            )
            for agg, r_table in r_tables.items()
        ],
    )


def gen_groupby_statistic_report(df: pd.DataFrame, agg: str, input_data_by) -> Report:
    headers, rows = [], []
    for k in df.columns:
        headers.append(
            Table.HeaderItem(
                name=k, desc="key" if k in input_data_by else "value", type="str"
            )
        )

    for index, df_row in df.iterrows():
        rows.append(
            Table.Row(
                name=str(index), items=[Attribute(s=str(df_row[k])) for k in df.columns]
            )
        )

    r_table = Table(
        headers=headers,
        rows=rows,
        name=agg,
        desc=f"Groupby statistics table for {agg} operation",
    )
    return r_table


def dump_groupby_statistics(
    name, system_info, agg_df_dict: Dict[str, pd.DataFrame], input_data_by
) -> DistData:
    report_mate = gen_groupby_statistic_reports(agg_df_dict, input_data_by)
    res = DistData(
        name=name,
        system_info=system_info,
        type=str(DistDataType.REPORT),
        data_refs=[],
    )
    res.meta.Pack(report_mate)
    return res


@groupby_statistics_comp.eval_fn
def groupby_statistics_eval_fn(
    *, ctx, aggs, max_group_size, input_data, input_data_by, input_data_values, report
):
    assert len(set(aggs)) == len(aggs), "no repeat in aggs allowed"
    assert (
        len(set(input_data_by).intersection(input_data_values)) == 0
    ), "by columns and key columns should have no intersection"
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    input_df = load_table(
        ctx, input_data, load_features=True, load_labels=True, load_ids=True
    )
    with ctx.tracer.trace_running():
        result = ordinal_encoded_groupby_aggs(
            input_df, input_data_by, input_data_values, spu, aggs, max_group_size
        )
    result = {agg: df.reset_index() for agg, df in result.items()}
    return {
        "report": dump_groupby_statistics(
            report, input_data.system_info, result, input_data_by
        )
    }
