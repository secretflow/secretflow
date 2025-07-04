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

from google.protobuf.json_format import MessageToJson

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    register,
    Reporter,
    VTable,
    VTableFieldKind,
)
from secretflow.spec.extend.groupby_aggregation_config_pb2 import (
    ColumnQuery,
    GroupbyAggregationConfig,
)
from secretflow.stats.groupby_v import ordinal_encoded_groupby_value_agg_pairs
from secretflow.utils.consistent_ops import unique_list
from secretflow.utils.errors import InvalidArgumentError

ENUM_TO_STR = {
    ColumnQuery.AggregationFunction.COUNT: "count",
    ColumnQuery.AggregationFunction.SUM: "sum",
    ColumnQuery.AggregationFunction.MEAN: "mean",
    ColumnQuery.AggregationFunction.MIN: "min",
    ColumnQuery.AggregationFunction.MAX: "max",
    ColumnQuery.AggregationFunction.VAR: "var",
}
STR_TO_ENUM = {v: k for k, v in ENUM_TO_STR.items()}


def map_enum_type_to_agg(enum_type: ColumnQuery.AggregationFunction):
    if enum_type in ENUM_TO_STR:
        return ENUM_TO_STR[enum_type]
    else:
        raise ValueError("unknown aggregation function")


@register(domain="stats", version="1.0.0")
class GroupbyStatistics(Component):
    '''
    Get a groupby of statistics, like pandas groupby statistics.
    Currently only support VDataframe.
    '''

    # it turns out that our implementation efficiency works bad in multiple columns
    # pandas style groupby is not practical to use, due to the above reason
    # so we change to sql style groupby instead
    aggregation_config: GroupbyAggregationConfig = Field.custom_attr(
        desc="input groupby aggregation config",
    )
    max_group_size: int = Field.attr(
        desc="The maximum number of groups allowed",
        default=10000,
        bound_limit=Interval.open(0, 10001),
    )
    by: list[str] = Field.table_column_attr(
        "input_ds",
        desc="by what columns should we group the values, encode values into int or str before groupby or else numeric errors may occur",
        limit=Interval.closed(1, 4),
    )
    input_ds: Input = Field.input(
        desc="Input table.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    report: Output = Field.output(
        desc="Output groupby statistics report.",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        value_columns = [
            col_config.column_name
            for col_config in self.aggregation_config.column_queries
        ]
        for col_config in self.aggregation_config.column_queries:
            if col_config.function == ColumnQuery.AggregationFunction.INVAL:
                raise InvalidArgumentError("aggregation function must be valid")

        if set(self.by).intersection(value_columns):
            raise InvalidArgumentError(
                "by columns and key columns should have no intersection",
                detail={"by": self.by, "value_columns": value_columns},
            )

        # FIXME: avoid to_pandas, use pa.Table
        input_vtable = VTable.from_distdata(
            self.input_ds,
            columns=self.by + unique_list(value_columns),
        )
        input_vtable.check_kinds(VTableFieldKind.FEATURE)

        input_df = ctx.load_table(input_vtable).to_pandas(check_null=False)
        value_agg_pair = [
            (col_query.column_name, map_enum_type_to_agg(col_query.function))
            for col_query in self.aggregation_config.column_queries
        ]

        if len(input_df.partitions) == 1:
            device = next(iter((input_df.partitions.keys())))
        else:
            device = ctx.make_spu()

        with ctx.tracer.trace_running():
            result = ordinal_encoded_groupby_value_agg_pairs(
                input_df,
                self.by,
                value_agg_pair,
                device,
                self.max_group_size,
            )
            result = {
                value_agg[0] + "_" + value_agg[1]: df.reset_index()
                for value_agg, df in result.items()
            }

        r = Reporter(name="groupby statistics", system_info=self.input_ds.system_info)
        for agg, df in result.items():
            df = df.astype(str)
            descriptions = {k: "key" if k in self.by else "value" for k in df.columns}
            desc = f"Groupby statistics table for {agg} operation"
            r_table = Reporter.build_table(
                df, name=agg, desc=desc, columns=descriptions
            )
            r.add_tab(r_table, name=agg)
        self.report.data = r.to_distdata()
        logging.info(f'\n--\n*report* \n\n{MessageToJson(r.report())}\n--\n')
