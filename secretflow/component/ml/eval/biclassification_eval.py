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

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, load_table
from secretflow.device.driver import reveal
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab, Table
from secretflow.stats.biclassification_eval import BiClassificationEval

biclassification_eval_comp = Component(
    name="biclassification_eval",
    domain="ml.eval",
    version="0.0.1",
    desc="""Statistics evaluation for a bi-classification model on a dataset.
        1. summary_report: SummaryReport

        2. group_reports: List[GroupReport]

        3. eq_frequent_bin_report: List[EqBinReport]

        4. eq_range_bin_report: List[EqBinReport]

        5. head_report: List[PrReport]
            reports for fpr = 0.001, 0.005, 0.01, 0.05, 0.1, 0.2
        """,
)

biclassification_eval_comp.int_attr(
    name="bucket_size",
    desc="Number of buckets.",
    is_list=False,
    is_optional=True,
    default_value=10,
    lower_bound=1,
    lower_bound_inclusive=True,
)

biclassification_eval_comp.int_attr(
    name="min_item_cnt_per_bucket",
    desc="Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 5.",
    is_list=False,
    is_optional=True,
    default_value=5,
    lower_bound=5,
    lower_bound_inclusive=True,
)

biclassification_eval_comp.io(
    io_type=IoType.INPUT,
    name="labels",
    desc="Input table with labels",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="col",
            desc="The column name to use in the dataset. If not provided, the label of dataset will be used by default.",
            col_max_cnt_inclusive=1,
        )
    ],
)

biclassification_eval_comp.io(
    io_type=IoType.INPUT,
    name="predictions",
    desc="Input table with predictions",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="col",
            desc="The column name to use in the dataset. If not provided, the label of dataset will be used by default.",
            col_max_cnt_inclusive=1,
        )
    ],
)

biclassification_eval_comp.io(
    io_type=IoType.OUTPUT,
    name="reports",
    desc="Output report.",
    types=[DistDataType.REPORT],
    col_params=None,
)


@biclassification_eval_comp.eval_fn
def biclassification_eval_fn(
    *,
    ctx,
    labels,
    labels_col,
    predictions,
    predictions_col,
    bucket_size,
    min_item_cnt_per_bucket,
    reports,
):
    labels_df = load_table(
        ctx,
        labels,
        load_labels=True,
        col_selects=labels_col if len(labels_col) else None,
    )

    predictions_df = load_table(
        ctx,
        predictions,
        load_labels=True,
        col_selects=predictions_col if len(predictions_col) else None,
    )

    with ctx.tracer.trace_running():
        result = reveal(
            BiClassificationEval(
                y_true=labels_df,
                y_score=predictions_df,
                bucket_size=bucket_size,
                min_item_cnt_per_bucket=min_item_cnt_per_bucket,
            ).get_all_reports()
        )

    logging.info(
        f"auc: {result.summary_report.auc}, ks: {result.summary_report.ks}, f1: {result.summary_report.f1_score}"
    )
    return {
        "reports": dump_biclassification_reports(reports, labels.system_info, result)
    }


def dump_biclassification_reports(name, system_info, reports):
    ret = DistData(
        name=name,
        system_info=system_info,
        type=str(DistDataType.REPORT),
    )

    def get_div_from_eq_bin_report(equal_bin_reports):
        headers, rows = [], []
        headers = [
            Table.HeaderItem(
                name="start_value",
                type="float",
            ),
            Table.HeaderItem(
                name="end_value",
                type="float",
            ),
            Table.HeaderItem(
                name="positive",
                type="int",
            ),
            Table.HeaderItem(
                name="negative",
                type="int",
            ),
            Table.HeaderItem(
                name="total",
                type="int",
            ),
            Table.HeaderItem(
                name="precision",
                type="float",
            ),
            Table.HeaderItem(
                name="recall",
                type="float",
            ),
            Table.HeaderItem(
                name="false_positive_rate",
                type="float",
            ),
            Table.HeaderItem(
                name="f1_score",
                type="float",
            ),
            Table.HeaderItem(
                name="lift",
                type="float",
            ),
            Table.HeaderItem(
                name="predicted_positive_ratio",
                type="float",
            ),
            Table.HeaderItem(
                name="predicted_negative_ratio",
                type="float",
            ),
            Table.HeaderItem(
                name="cumulative_percent_of_positive",
                type="float",
            ),
            Table.HeaderItem(
                name="cumulative_percent_of_negative",
                type="float",
            ),
            Table.HeaderItem(
                name="total_cumulative_percent",
                type="float",
            ),
            Table.HeaderItem(
                name="ks",
                type="float",
            ),
            Table.HeaderItem(
                name="avg_score",
                type="float",
            ),
        ]
        for idx, bin_report in enumerate(equal_bin_reports):
            rows.append(
                Table.Row(
                    name=f'bin_{idx}',
                    items=[
                        Attribute(f=bin_report.start_value),
                        Attribute(f=bin_report.end_value),
                        Attribute(i64=int(bin_report.positive)),
                        Attribute(i64=int(bin_report.negative)),
                        Attribute(i64=int(bin_report.total)),
                        Attribute(f=bin_report.precision),
                        Attribute(f=bin_report.recall),
                        Attribute(f=bin_report.false_positive_rate),
                        Attribute(f=bin_report.f1_score),
                        Attribute(f=bin_report.Lift),
                        Attribute(f=bin_report.predicted_positive_ratio),
                        Attribute(f=bin_report.predicted_negative_ratio),
                        Attribute(f=bin_report.cumulative_percent_of_positive),
                        Attribute(f=bin_report.cumulative_percent_of_negative),
                        Attribute(f=bin_report.total_cumulative_percent),
                        Attribute(f=bin_report.ks),
                        Attribute(f=bin_report.avg_score),
                    ],
                )
            )
        return Div(
            name="",
            desc="",
            children=[
                Div.Child(
                    type="table",
                    table=Table(
                        name="",
                        desc="",
                        headers=headers,
                        rows=rows,
                    ),
                ),
            ],
        )

    def make_head_report_div(head_report):
        headers = [
            Table.HeaderItem(
                name="threshold",
                type="float",
            ),
            Table.HeaderItem(
                name="FPR(False Positive Rate)",
                type="float",
            ),
            Table.HeaderItem(
                name="precision",
                type="float",
            ),
            Table.HeaderItem(
                name="recall",
                type="float",
            ),
        ]
        rows = []
        for idx, report in enumerate(head_report):
            rows.append(
                Table.Row(
                    name=f"case_{idx}",
                    items=[
                        Attribute(f=report.threshold),
                        Attribute(f=report.fpr),
                        Attribute(f=report.precision),
                        Attribute(f=report.recall),
                    ],
                )
            )
        return Div(
            name="",
            desc="",
            children=[
                Div.Child(
                    type="table",
                    table=Table(
                        name="",
                        desc="",
                        headers=headers,
                        rows=rows,
                    ),
                ),
            ],
        )

    meta = Report(
        name="reports",
        desc="",
        tabs=[
            Tab(
                name="SummaryReport",
                desc="Summary Report for bi-classification evaluation.",
                divs=[
                    Div(
                        name="",
                        desc="",
                        children=[
                            Div.Child(
                                type="descriptions",
                                descriptions=Descriptions(
                                    name="",
                                    desc="",
                                    items=[
                                        Descriptions.Item(
                                            name="total_samples",
                                            type="int",
                                            value=Attribute(
                                                i64=int(
                                                    reports.summary_report.total_samples
                                                )
                                            ),
                                        ),
                                        Descriptions.Item(
                                            name="positive_samples",
                                            type="int",
                                            value=Attribute(
                                                i64=int(
                                                    reports.summary_report.positive_samples
                                                )
                                            ),
                                        ),
                                        Descriptions.Item(
                                            name="negative_samples",
                                            type="int",
                                            value=Attribute(
                                                i64=int(
                                                    reports.summary_report.negative_samples
                                                )
                                            ),
                                        ),
                                        Descriptions.Item(
                                            name="auc",
                                            type="float",
                                            value=Attribute(
                                                f=reports.summary_report.auc
                                            ),
                                        ),
                                        Descriptions.Item(
                                            name="ks",
                                            type="float",
                                            value=Attribute(
                                                f=reports.summary_report.ks
                                            ),
                                        ),
                                        Descriptions.Item(
                                            name="f1_score",
                                            type="float",
                                            value=Attribute(
                                                f=reports.summary_report.f1_score
                                            ),
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            Tab(
                name="eq_frequent_bin_report",
                desc="Statistics Report for each bin.",
                divs=[get_div_from_eq_bin_report(reports.eq_frequent_bin_report)],
            ),
            Tab(
                name="eq_range_bin_report",
                desc="",
                divs=[get_div_from_eq_bin_report(reports.eq_range_bin_report)],
            ),
            Tab(
                name="head_report",
                desc="",
                divs=[make_head_report_div(reports.head_report)],
            ),
        ],
    )
    ret.meta.Pack(meta)
    return ret
