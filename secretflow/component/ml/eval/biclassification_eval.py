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

import numpy as np
import pandas as pd

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    Reporter,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.device.driver import reveal
from secretflow.stats.biclassification_eval import (
    BiClassificationEval as StatsBiClassificationEval,
)


@register(domain="ml.eval", version="1.0.0", name="biclassification_eval")
class BiClassificationEval(Component):
    '''
    Statistics evaluation for a bi-classification model on a dataset.
        1. summary_report: SummaryReport

        2. eq_frequent_bin_report: List[EqBinReport]

        3. eq_range_bin_report: List[EqBinReport]

        4. head_report: List[PrReport]
            reports for fpr = 0.001, 0.005, 0.01, 0.05, 0.1, 0.2
    '''

    bucket_size: int = Field.attr(
        desc="Number of buckets.",
        default=10,
        bound_limit=Interval.closed(1, None),
    )
    min_item_cnt_per_bucket: int = Field.attr(
        desc="Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 5.",
        default=5,
        bound_limit=Interval.closed(5, None),
    )
    label: str = Field.table_column_attr(
        "input_ds",
        desc="The label name to use in the dataset.",
    )
    prediction: str = Field.table_column_attr(
        "input_ds",
        desc="The prediction result column name to use in the dataset.",
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="Input table with prediction and label, usually is a result from a prediction component.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    report: Output = Field.output(
        desc="Output report.",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        tbl = VTable.from_distdata(self.input_ds, [self.label, self.prediction])
        tbl.check_kinds(VTableFieldKind.FEATURE_LABEL)
        label_prediction_df = ctx.load_table(tbl).to_pandas(
            check_null=False
        )  # FIXME: avoid to_pandas

        with ctx.trace_running():
            result = reveal(
                StatsBiClassificationEval(
                    y_true=label_prediction_df[[self.label]],
                    y_score=label_prediction_df[[self.prediction]],
                    bucket_size=self.bucket_size,
                    min_item_cnt_per_bucket=self.min_item_cnt_per_bucket,
                ).get_all_reports()
            )

        r = Reporter(name="reports")
        # build summary_report
        summary_report = result.summary_report
        summary_data = {
            "total_samples": int(summary_report.total_samples),
            "positive_samples": int(summary_report.positive_samples),
            "negative_samples": int(summary_report.negative_samples),
            "auc": float(summary_report.auc),
            "ks": float(summary_report.ks),
            "f1_score": float(summary_report.f1_score),
        }
        r.add_tab(
            summary_data,
            name="SummaryReport",
            desc="Summary Report for bi-classification evaluation.",
        )

        # eq_frequent_bin_report
        eq_frequent_bin_tbl = r.to_table(
            self.eq_bin_to_df(result.eq_frequent_bin_report), prefix="bin_"
        )
        r.add_tab(
            eq_frequent_bin_tbl,
            name="eq_frequent_bin_report",
            desc="Statistics Report for each bin.",
        )

        # eq_range_bin_report
        eq_range_bin_tbl = r.to_table(
            self.eq_bin_to_df(result.eq_range_bin_report), prefix="bin_"
        )
        r.add_tab(eq_range_bin_tbl, name="eq_range_bin_report")

        # head_report
        head_report_table = r.to_table(
            self.head_report_to_df(result.head_report), prefix="case_"
        )
        r.add_tab(head_report_table, name="head_report")

        r.dump_to(self.report, self.input_ds.system_info)

    @staticmethod
    def eq_bin_to_df(equal_bin_reports) -> pd.DataFrame:
        columns = [
            'start_value',
            'end_value',
            'positive',
            'negative',
            'total',
            'precision',
            'recall',
            'false_positive_rate',
            'f1_score',
            'lift',
            'predicted_positive_ratio',
            'predicted_negative_ratio',
            'cumulative_percent_of_positive',
            'cumulative_percent_of_negative',
            'total_cumulative_percent',
            'ks',
            'avg_score',
        ]
        rows = []
        for bin_report in equal_bin_reports:
            rows.append(
                [
                    float(bin_report.start_value),
                    float(bin_report.end_value),
                    int(bin_report.positive),
                    int(bin_report.negative),
                    int(bin_report.total),
                    float(bin_report.precision),
                    float(bin_report.recall),
                    float(bin_report.false_positive_rate),
                    float(bin_report.f1_score),
                    float(bin_report.Lift),
                    float(bin_report.predicted_positive_ratio),
                    float(bin_report.predicted_negative_ratio),
                    float(bin_report.cumulative_percent_of_positive),
                    float(bin_report.cumulative_percent_of_negative),
                    float(bin_report.total_cumulative_percent),
                    float(bin_report.ks),
                    float(bin_report.avg_score),
                ]
            )
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def head_report_to_df(head_report) -> pd.DataFrame:
        columns = ["threshold", "FPR(False Positive Rate)", "precision", "recall"]
        rows = []
        for r in head_report:
            rows.append(
                [float(r.threshold), float(r.fpr), float(r.precision), float(r.recall)]
            )

        return pd.DataFrame(rows, columns=columns)
