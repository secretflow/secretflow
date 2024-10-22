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
from secretflow.device import reveal
from secretflow.stats.core.prediction_bias_core import PredictionBiasReport
from secretflow.stats.prediction_bias_eval import prediction_bias_eval


@register(domain="ml.eval", version="1.0.0", name="prediction_bias_eval")
class PredictionBiasEval(Component):
    '''
    Calculate prediction bias, ie. average of predictions - average of labels.
    '''

    bucket_num: int = Field.attr(
        desc="Num of bucket.",
        default=10,
        bound_limit=Interval.closed(1, None),
    )
    min_item_cnt_per_bucket: int = Field.attr(
        desc="Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 2.",
        default=2,
        bound_limit=Interval.closed(2, None),
    )
    bucket_method: str = Field.attr(
        desc="Bucket method.",
        default="equal_width",
        choices=["equal_width", "equal_frequency"],
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
        )  # FIXME: avoid to_pandas, use pa.Table

        with ctx.trace_running():
            res: PredictionBiasReport = reveal(
                prediction_bias_eval(
                    prediction=label_prediction_df[[self.prediction]],
                    label=label_prediction_df[[self.label]],
                    bucket_num=self.bucket_num,
                    absolute=True,
                    bucket_method=self.bucket_method,
                    min_item_cnt_per_bucket=self.min_item_cnt_per_bucket,
                )
            )

        def gen_interval_str(left_endpoint, left_closed, right_endpoint, right_closed):
            return f"{'[' if left_closed else '('}{left_endpoint}, {right_endpoint}{']' if right_closed else ')'}"

        columns = {
            "interval": "prediction interval",
            "left_endpoint": "left endpoint of interval",
            "left_closed": "indicate if left endpoint of interval is closed",
            "right_endpoint": "right endpoint of interval",
            "right_closed": "indicate if right endpoint of interval is closed",
            "avg_prediction": "average prediction of interval",
            "avg_label": "average label of interval",
            "bias": "prediction bias of interval",
        }
        rows = []
        for b in res.buckets:
            interval_str = gen_interval_str(
                b.left_endpoint,
                b.left_closed,
                b.right_endpoint,
                b.right_closed,
            )
            rows.append(
                [
                    interval_str,
                    b.left_endpoint,
                    b.left_closed,
                    b.right_endpoint,
                    b.right_closed,
                    b.avg_prediction,
                    b.avg_label,
                    b.bias,
                ]
            )

        report_df = pd.DataFrame(rows, columns=columns.keys())
        for k in report_df.columns:
            Reporter.set_description(report_df[k], columns[k])

        r = Reporter(name="Prediction Bias Report")
        tbl_div = r.to_table(
            report_df,
            name="Prediction Bias Table",
            desc="Calculate prediction bias, ie. average of predictions - average of labels.",
        )
        r.add_tab(tbl_div)
        r.dump_to(self.report, self.input_ds.system_info)
