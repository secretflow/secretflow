# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

from typing import Union

import pandas as pd

from secretflow.data.vertical import VDataFrame


def table_statistics(table: Union[pd.DataFrame, VDataFrame]) -> pd.DataFrame:
    """Get table statistics for a pd.DataFrame or VDataFrame.

    Args:
        table: Union[pd.DataFrame, VDataFrame]
    Returns:
        table_statistics: pd.DataFrame
            including each column's datatype, total_count, count, count_na, min, max,
            var, std, sem, skewness, kurtosis, q1, q2, q3, moment_2, moment_3, moment_4,
            central_moment_2, central_moment_3, central_moment_4, sum, sum_2, sum_3 and sum_4.

            moment_2 means E[X^2].

            central_moment_2 means E[(X - mean(X))^2].

            sum_2 means sum(X^2).
    """
    assert isinstance(
        table, (pd.DataFrame, VDataFrame)
    ), "table must be a pd.DataFrame or VDataFrame"
    index = table.columns
    result = pd.DataFrame(index=index)
    result["datatype"] = table.dtypes
    result["total_count"] = table.shape[0]
    result["count(non-NA count)"] = table.count()
    result["count_na(NA count)"] = table.isna().sum()
    result["na_ratio"] = table.isna().sum() / table.shape[0]
    result["min"] = table.min(numeric_only=True)
    result["max"] = table.max(numeric_only=True)
    result["mean"] = table.mean(numeric_only=True)
    result["var(variance)"] = table.var(numeric_only=True)
    result["std(standard deviation)"] = table.std(numeric_only=True)
    result["sem(standard error)"] = table.sem(numeric_only=True)
    result["skew"] = table.skew(numeric_only=True)
    result["kurtosis"] = table.kurtosis(numeric_only=True)
    result["q1(first quartile)"] = table.quantile(0.25)
    result["q2(second quartile, median)"] = table.quantile(0.5)
    result["q3(third quartile)"] = table.quantile(0.75)
    result["moment_2"] = table.select_dtypes("number").pow(2).mean(numeric_only=True)
    result["moment_3"] = table.select_dtypes("number").pow(3).mean(numeric_only=True)
    result["moment_4"] = table.select_dtypes("number").pow(4).mean(numeric_only=True)
    result["central_moment_2"] = (
        table.subtract(result["mean"])
        .select_dtypes("number")
        .pow(2)
        .mean(numeric_only=True)
    )
    result["central_moment_3"] = (
        table.subtract(result["mean"])
        .select_dtypes("number")
        .pow(3)
        .mean(numeric_only=True)
    )
    result["central_moment_4"] = (
        table.subtract(result["mean"])
        .select_dtypes("number")
        .pow(4)
        .mean(numeric_only=True)
    )
    result["sum"] = table.sum(numeric_only=True)
    result["sum_2"] = table.select_dtypes("number").pow(2).sum(numeric_only=True)
    result["sum_3"] = table.select_dtypes("number").pow(3).sum(numeric_only=True)
    result["sum_4"] = table.select_dtypes("number").pow(4).sum(numeric_only=True)
    return result
