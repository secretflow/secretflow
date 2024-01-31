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
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab, Table
from secretflow.stats.regression_eval import RegressionEval


regression_eval_comp = Component(
    name="regression_eval",
    domain="ml.eval",
    version="0.0.1",
    desc="""Statistics evaluation for a regression model on a dataset.
        Contained Statistics:
            R2 Score (r2_score): It is a statistical measure that represents the proportion of the variance in the dependent variable that can be predicted from the independent variables. It ranges from 0 to 1, where a higher value indicates a better fit.

            Mean Absolute Error (mean_abs_err): It calculates the average absolute difference between the predicted and actual values. It provides a measure of the average magnitude of the errors.

            Mean Absolute Percentage Error (mean_abs_percent_err): It calculates the average absolute percentage difference between the predicted and actual values. It measures the average magnitude of the errors in terms of percentages.

            Sum of Squared Errors (sum_squared_errors): It calculates the sum of the squared differences between the predicted and actual values. It provides an overall measure of the model's performance.

            Mean Squared Error (mean_squared_errors): It calculates the average of the squared differences between the predicted and actual values. It is widely used as a loss function in regression problems.

            Root Mean Squared Error (root_mean_squared_errors): It is the square root of the mean squared error. It provides a measure of the average magnitude of the errors in the original scale of the target variable.

            Mean of True Values (y_true_mean): It calculates the average of the actual values in the target variable. It can be useful for establishing a baseline for the model's performance.

            Mean of Predicted Values (y_pred_mean): It calculates the average of the predicted values. It can be compared with the y_true_mean to get an idea of the model's bias.

            Residual Histograms (residual_hists): It represents the distribution of the differences between the predicted and actual values. It helps to understand the spread and pattern of the errors.
        """,
)

regression_eval_comp.int_attr(
    name="bucket_size",
    desc="Number of buckets for residual histogram.",
    is_list=False,
    is_optional=True,
    default_value=10,
    lower_bound=1,
    lower_bound_inclusive=True,
    upper_bound=10000,
    upper_bound_inclusive=True,
)


regression_eval_comp.io(
    io_type=IoType.INPUT,
    name="in_ds",
    desc="Input table with prediction and label, usually is a result from a prediction component.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="label",
            desc="The label name to use in the dataset.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        ),
        TableColParam(
            name="prediction",
            desc="The prediction result column name to use in the dataset.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        ),
    ],
)


regression_eval_comp.io(
    io_type=IoType.OUTPUT,
    name="reports",
    desc="Output report.",
    types=[DistDataType.REPORT],
    col_params=None,
)


@regression_eval_comp.eval_fn
def regression_eval_fn(
    *,
    ctx,
    bucket_size,
    in_ds,
    in_ds_label,
    in_ds_prediction,
    reports,
):
    label_prediction_df = load_table(
        ctx,
        in_ds,
        load_labels=True,
        col_selects=in_ds_label + in_ds_prediction,
    )

    with ctx.tracer.trace_running():
        result = RegressionEval(
            y_true=label_prediction_df[in_ds_label].values,
            y_pred=label_prediction_df[in_ds_prediction].values,
            bins=bucket_size,
        )
        result.gen_all_reports()

    return {"reports": dump_regression_reports(reports, in_ds.system_info, result)}


def dump_regression_reports(name, system_info, reports):
    ret = DistData(
        name=name,
        system_info=system_info,
        type=str(DistDataType.REPORT),
    )

    meta = Report(
        name="reports",
        desc="",
        tabs=[
            Tab(
                name="metrics",
                desc="regression evaluation metrics",
                divs=[get_attributes_div_from_regression_eval_report(reports)],
            ),
            Tab(
                name="residual histogram",
                desc="regression residual histogram",
                divs=[get_histogram_div_from_gression_eval_report(reports)],
            ),
        ],
    )
    ret.meta.Pack(meta)
    return ret


def list_to_attr_list(values):
    return [Attribute(f=value) for value in values]


def get_histogram_div_from_gression_eval_report(
    regression_result: RegressionEval,
):
    hist_result = reveal(regression_result.residual_hist)
    hist, bin_edges = hist_result[0], hist_result[1]
    n = hist.size
    header_names = [f"bin_{i}" for i in range(n + 1)]
    values = list(hist) + [0]
    headers = [
        Table.HeaderItem(
            name=header_name,
            type="float",
        )
        for header_name in header_names
    ]
    rows = [
        Table.Row(name="edge_values", items=list_to_attr_list(bin_edges)),
        Table.Row(name="sample_counts", items=list_to_attr_list(values)),
    ]
    return Div(
        name="residual histogram",
        desc="Residual Histograms (residual_hists): It represents the distribution of the differences between the predicted and actual values. It helps to understand the spread and pattern of the errors.",
        children=[
            Div.Child(
                type="table",
                table=Table(
                    name="histogram data",
                    desc=f"boundary for bins and value for each bin. there are {n} bins and {n+1} edges. (pad last non-existent bin count with 0)",
                    headers=headers,
                    rows=rows,
                ),
            ),
        ],
    )


STATS_DESC = """
    R2 Score (r2_score): It is a statistical measure that represents the proportion of the variance in the dependent variable that can be predicted from the independent variables. It ranges from 0 to 1, where a higher value indicates a better fit.

    Mean Absolute Error (mean_abs_err): It calculates the average absolute difference between the predicted and actual values. It provides a measure of the average magnitude of the errors.

    Mean Absolute Percentage Error (mean_abs_percent_err): It calculates the average absolute percentage difference between the predicted and actual values. It measures the average magnitude of the errors in terms of percentages.

    Sum of Squared Errors (sum_squared_errors): It calculates the sum of the squared differences between the predicted and actual values. It provides an overall measure of the model's performance.

    Mean Squared Error (mean_squared_errors): It calculates the average of the squared differences between the predicted and actual values. It is widely used as a loss function in regression problems.

    Root Mean Squared Error (root_mean_squared_errors): It is the square root of the mean squared error. It provides a measure of the average magnitude of the errors in the original scale of the target variable.

    Mean of True Values (y_true_mean): It calculates the average of the actual values in the target variable. It can be useful for establishing a baseline for the model's performance.

    Mean of Predicted Values (y_pred_mean): It calculates the average of the predicted values. It can be compared with the y_true_mean to get an idea of the model's bias."""


def get_attributes_div_from_regression_eval_report(
    regression_result: RegressionEval,
):
    header_names, values = [
        "r2_score",
        "mean_abs_err",
        "mean_abs_percent_err",
        "sum_squared_errors",
        "mean_squared_errors",
        "root_mean_squared_errors",
        "y_true_mean",
        "y_pred_mean",
    ], reveal(regression_result.result_as_list()[:-1])

    items = []
    for header_name, value in zip(header_names, values):
        items.append(
            Descriptions.Item(
                name=f"{header_name}",
                type="float",
                value=Attribute(f=value),
            )
        )
    desc = Descriptions(name="calculated_results", desc=f"{STATS_DESC}", items=items)

    return Div(
        name="metrics",
        desc="statistical metrics",
        children=[
            Div.Child(
                type="descriptions",
                descriptions=desc,
            ),
        ],
    )
