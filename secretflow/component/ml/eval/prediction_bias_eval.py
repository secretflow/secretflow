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

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, load_table
from secretflow.device.driver import reveal
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Div, Report, Tab, Table
from secretflow.stats.core.prediction_bias_core import PredictionBiasReport
from secretflow.stats.prediction_bias_eval import prediction_bias_eval

prediction_bias_comp = Component(
    "prediction_bias_eval",
    domain="ml.eval",
    version="0.0.1",
    desc="Calculate prediction bias, ie. average of predictions - average of labels.",
)

prediction_bias_comp.int_attr(
    name="bucket_num",
    desc="Num of bucket.",
    is_list=False,
    is_optional=True,
    default_value=10,
    lower_bound=1,
    lower_bound_inclusive=True,
)

prediction_bias_comp.int_attr(
    name="min_item_cnt_per_bucket",
    desc="Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 2.",
    is_list=False,
    is_optional=True,
    default_value=2,
    lower_bound=2,
    lower_bound_inclusive=True,
)

prediction_bias_comp.str_attr(
    name="bucket_method",
    desc="Bucket method.",
    is_list=False,
    is_optional=True,
    default_value="equal_width",
    allowed_values=["equal_width", "equal_frequency"],
)

prediction_bias_comp.io(
    io_type=IoType.INPUT,
    name="labels",
    desc="Input table with labels.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="col",
            desc="The column name to use in the dataset. If not provided, the label of dataset will be used by default.",
            col_max_cnt_inclusive=1,
        )
    ],
)

prediction_bias_comp.io(
    io_type=IoType.INPUT,
    name="predictions",
    desc="Input table with predictions.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="col",
            desc="The column name to use in the dataset. If not provided, the label of dataset will be used by default.",
            col_max_cnt_inclusive=1,
        )
    ],
)

prediction_bias_comp.io(
    io_type=IoType.OUTPUT,
    name="result",
    desc="Output report.",
    types=[DistDataType.REPORT],
)


def dump_report(name, system_info, report: PredictionBiasReport) -> DistData:
    table = Table(
        name="Prediction Bias Table",
        desc="Calculate prediction bias, ie. average of predictions - average of labels.",
        headers=[
            Table.HeaderItem(name="interval", desc="prediction interval", type="str"),
            Table.HeaderItem(
                name="left_endpoint",
                desc="left endpoint of interval",
                type="float",
            ),
            Table.HeaderItem(
                name="left_closed",
                desc="indicate if left endpoint of interval is closed",
                type="bool",
            ),
            Table.HeaderItem(
                name="right_endpoint",
                desc="right endpoint of interval",
                type="float",
            ),
            Table.HeaderItem(
                name="right_closed",
                desc="indicate if right endpoint of interval is closed",
                type="bool",
            ),
            Table.HeaderItem(
                name="avg_prediction",
                desc="average prediction of interval",
                type="float",
            ),
            Table.HeaderItem(
                name="avg_label",
                desc="average label of interval",
                type="float",
            ),
            Table.HeaderItem(
                name="bias",
                desc="prediction bias of interval",
                type="float",
            ),
        ],
    )

    def gen_interval_str(left_endpoint, left_closed, right_endpoint, right_closed):
        return f"{'[' if left_closed else '('}{left_endpoint}, {right_endpoint}{']' if right_closed else ')'}"

    for i, b in enumerate(report.buckets):
        table.rows.append(
            Table.Row(
                name=str(i),
                items=[
                    Attribute(
                        s=gen_interval_str(
                            b.left_endpoint,
                            b.left_closed,
                            b.right_endpoint,
                            b.right_closed,
                        )
                    ),
                    Attribute(f=b.left_endpoint),
                    Attribute(b=b.left_closed),
                    Attribute(f=b.right_endpoint),
                    Attribute(b=b.right_closed),
                    Attribute(f=b.avg_prediction),
                    Attribute(f=b.avg_label),
                    Attribute(f=b.bias),
                ],
            )
        )

    report_meta = Report(
        name="Prediction Bias Report",
        tabs=[
            Tab(
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="table",
                                table=table,
                            )
                        ],
                    )
                ],
            )
        ],
    )

    report_dd = DistData(
        name=name,
        type=str(DistDataType.REPORT),
        system_info=system_info,
    )
    report_dd.meta.Pack(report_meta)

    return report_dd


@prediction_bias_comp.eval_fn
def prediction_bias_eval_fn(
    *,
    ctx,
    bucket_num,
    min_item_cnt_per_bucket,
    bucket_method,
    labels,
    labels_col,
    predictions,
    predictions_col,
    result,
):
    labels_data = load_table(
        ctx,
        labels,
        load_labels=True,
        col_selects=labels_col if len(labels_col) else None,
    )
    predictions_data = load_table(
        ctx,
        predictions,
        load_labels=True,
        col_selects=predictions_col if len(predictions_col) else None,
    )

    with ctx.tracer.trace_running():
        res = reveal(
            prediction_bias_eval(
                prediction=predictions_data,
                label=labels_data,
                bucket_num=bucket_num,
                absolute=True,
                bucket_method=bucket_method,
                min_item_cnt_per_bucket=min_item_cnt_per_bucket,
            )
        )

    return {"result": dump_report(result, labels.system_info, res)}
