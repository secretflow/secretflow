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

import pandas as pd
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
from secretflow.device.driver import reveal
from secretflow.stats.regression_eval import RegressionEval as StatsRegressionEval

STATS_DESC = """
    R2 Score (r2_score): It is a statistical measure that represents the proportion of the variance in the dependent variable that can be predicted from the independent variables. It ranges from 0 to 1, where a higher value indicates a better fit.

    Mean Absolute Error (mean_abs_err): It calculates the average absolute difference between the predicted and actual values. It provides a measure of the average magnitude of the errors.

    Mean Absolute Percentage Error (mean_abs_percent_err): It calculates the average absolute percentage difference between the predicted and actual values. It measures the average magnitude of the errors in terms of percentages.

    Sum of Squared Errors (sum_squared_errors): It calculates the sum of the squared differences between the predicted and actual values. It provides an overall measure of the model's performance.

    Mean Squared Error (mean_squared_errors): It calculates the average of the squared differences between the predicted and actual values. It is widely used as a loss function in regression problems.

    Root Mean Squared Error (root_mean_squared_errors): It is the square root of the mean squared error. It provides a measure of the average magnitude of the errors in the original scale of the target variable.

    Mean of True Values (y_true_mean): It calculates the average of the actual values in the target variable. It can be useful for establishing a baseline for the model's performance.

    Mean of Predicted Values (y_pred_mean): It calculates the average of the predicted values. It can be compared with the y_true_mean to get an idea of the model's bias."""


@register(domain="ml.eval", version="1.0.0")
class RegressionEval(Component):
    '''
    Statistics evaluation for a regression model on a dataset.
    Contained Statistics:
        R2 Score (r2_score): It is a statistical measure that represents the proportion of the variance in the dependent variable that can be predicted from the independent variables. It ranges from -inf to 1, where a higher value indicates a better fit. (the value can be negative because the
        model can be arbitrarily worse). In the general case when the true y is non-constant, a constant model that always predicts the average y
        disregarding the input features would get a :math:`R^2` score of 0.0.

        Mean Absolute Error (mean_abs_err): It calculates the average absolute difference between the predicted and actual values. It provides a measure of the average magnitude of the errors.

        Mean Absolute Percentage Error (mean_abs_percent_err): It calculates the average absolute percentage difference between the predicted and actual values. It measures the average magnitude of the errors in terms of percentages.

        Sum of Squared Errors (sum_squared_errors): It calculates the sum of the squared differences between the predicted and actual values. It provides an overall measure of the model's performance.

        Mean Squared Error (mean_squared_errors): It calculates the average of the squared differences between the predicted and actual values. It is widely used as a loss function in regression problems.

        Root Mean Squared Error (root_mean_squared_errors): It is the square root of the mean squared error. It provides a measure of the average magnitude of the errors in the original scale of the target variable.

        Mean of True Values (y_true_mean): It calculates the average of the actual values in the target variable. It can be useful for establishing a baseline for the model's performance.

        Mean of Predicted Values (y_pred_mean): It calculates the average of the predicted values. It can be compared with the y_true_mean to get an idea of the model's bias.

        Residual Histograms (residual_hists): It represents the distribution of the differences between the predicted and actual values. It helps to understand the spread and pattern of the errors.
    '''

    bucket_size: int = Field.attr(
        desc="Number of buckets for residual histogram.",
        default=10,
        bound_limit=Interval.closed(1, 10000),
    )
    label: str = Field.table_column_attr(
        "input_ds",
        desc="The label name to use in the dataset.",
    )
    prediction: str = Field.table_column_attr(
        "input_ds",
        desc="The prediction result column name to use in the dataset.",
    )
    input_ds: Input = Field.input(
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
        # FIXME: avoid to_pandas, use pa.Table
        label_prediction_df = ctx.load_table(tbl).to_pandas(check_null=False)

        with ctx.trace_running():
            result = StatsRegressionEval(
                y_true=label_prediction_df[[self.label]].values,
                y_pred=label_prediction_df[[self.prediction]].values,
                bins=self.bucket_size,
            )
            result.gen_all_reports()

        r = Reporter(name="reports", system_info=self.input_ds.system_info)

        # build attributes_div
        names = [
            "r2_score",
            "mean_abs_err",
            "mean_abs_percent_err",
            "sum_squared_errors",
            "mean_squared_errors",
            "root_mean_squared_errors",
            "y_true_mean",
            "y_pred_mean",
        ]
        values = reveal(result.result_as_list()[:-1])
        items = {n: float(v) for n, v in zip(names, values)}
        descriptions = Reporter.build_descriptions(
            items, name="calculated_results", desc=f"{STATS_DESC}"
        )
        div = r.build_div(descriptions, name="metrics", desc="statistical metrics")
        r.add_tab(div, name="metrics", desc="regression evaluation metrics")

        # build histogram_div
        hist_result = reveal(result.residual_hist)
        hist, bin_edges = hist_result[0], hist_result[1]
        values = list(hist) + [0]
        n = hist.size
        column_names = [f"bin_{i}" for i in range(n + 1)]
        hist_df = pd.DataFrame(
            [bin_edges, values],
            index=["edge_values", "sample_counts"],
            columns=column_names,
        )
        hist_tbl = Reporter.build_table(
            hist_df.astype(float),
            name="histogram data",
            desc=f"boundary for bins and value for each bin. there are {n} bins and {n+1} edges. (pad last non-existent bin count with 0)",
            index=hist_df.index.to_list(),
        )
        hist_div = Reporter.build_div(
            hist_tbl,
            name="residual histogram",
            desc="Residual Histograms (residual_hists): It represents the distribution of the differences between the predicted and actual values. It helps to understand the spread and pattern of the errors.",
        )
        r.add_tab(
            hist_div,
            name="residual histogram",
            desc="regression residual histogram",
        )
        logging.info(f'\n--\n*report* \n\n{MessageToJson(r.report())}\n--\n')
        self.report.data = r.to_distdata()
