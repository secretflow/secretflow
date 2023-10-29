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

import pandas as pd
import numpy as np
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Div, Report, Tab, Table

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, load_table
from secretflow.component.component import CompEvalError
from secretflow.device.device.spu import SPU
from secretflow.device import reveal
from secretflow.preprocessing.encoder import LabelEncoder
from secretflow.data.vertical import VDataFrame
from secretflow.data.core import partition

groupby_statistics_comp = Component(
    name="groupby_statistics",
    domain="stats",
    version="0.0.1",
    desc="""Get a groupby of statistics, like pandas groupby statistics.
    Currently only support VDataframe.
    """,
)

groupby_statistics_comp.str_attr(
    name="agg",
    desc="What kind of aggregation statistics we want to do, currently only supports min, max, mean, sum, var, count(number of elements in each group)",
    is_list=False,
    is_optional=False,
    allowed_values=["min", "max", "mean", "sum", "var", "count"],
)


groupby_statistics_comp.int_attr(
    name="max_group_size",
    desc="The maximum number of groups allowed",
    is_list=False,
    is_optional=True,
    default_value=10000,
    lower_bound=0,
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
            desc="on which columns should we calculate the statistics",
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


def gen_groupby_statistic_report(df: pd.DataFrame) -> Report:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    headers, rows = [], []
    for k in df.columns:
        headers.append(Table.HeaderItem(name=k, desc="", type="str"))

    for index, df_row in df.iterrows():
        rows.append(
            Table.Row(
                name=str(index), items=[Attribute(s=str(df_row[k])) for k in df.columns]
            )
        )

    r_table = Table(headers=headers, rows=rows)

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
            )
        ],
    )


def dump_groupby_statistics(name, system_info, df: pd.DataFrame) -> DistData:
    report_mate = gen_groupby_statistic_report(df)
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
    *, ctx, agg, max_group_size, input_data, input_data_by, input_data_values, report
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    input_df = load_table(
        ctx, input_data, load_features=True, load_labels=True, load_ids=True
    )

    select_cols = input_data_values + input_data_by

    with ctx.tracer.trace_running():
        encoder = LabelEncoder()
        input_df[input_data_by] = encoder.fit_transform(input_df[input_data_by])
        group_num = np.prod(input_df[input_data_by].max().values + 1)
        assert (
            group_num <= max_group_size
        ), f"num groups {group_num} is larger than limit {max_group_size}"
        stat = getattr(
            input_df[select_cols].groupby(spu, input_data_by), agg
        )().reset_index(names=input_data_by)
        v_dataframe_by = {}

        for device, cols in input_df[input_data_by].partition_columns.items():
            v_dataframe_by[device] = partition(data=device(lambda x: x)(stat[cols]))
        df = VDataFrame(v_dataframe_by)
        df = encoder.inverse_transform(df)
        for device, cols in df.partition_columns.items():
            stat[cols] = reveal(df.partitions[device].data)

    return {"report": dump_groupby_statistics(report, input_data.system_info, stat)}
