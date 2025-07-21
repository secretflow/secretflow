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
)
from secretflow.stats.table_statistics import categorical_statistics, table_statistics


@register(domain="stats", version="1.0.2")
class TableStatistics(Component):
    '''
    Get a table of statistics,
    including each column's

    1. datatype
    2. total_count
    3. count
    4. count_na
    5. na_ratio
    6. min
    7. max
    8. mean
    9. var
    10. std
    11. sem
    12. skewness
    13. kurtosis
    14. q1
    15. q2
    16. q3
    17. moment_2
    18. moment_3
    19. moment_4
    20. central_moment_2
    21. central_moment_3
    22. central_moment_4
    23. sum
    24. sum_2
    25. sum_3
    26. sum_4

    - moment_2 means E[X^2].
    - central_moment_2 means E[(X - mean(X))^2].
    - sum_2 means sum(X^2).

    All of the object or string class columns will not be included in the above statistics, but in a separate report.

    The second report is a table of the object or string class columns.
    Note that please do not include individual information (like address, phone number, etc.) for table statistics.

    The categorical report will be with the following columns:
    1. column dtype (the data type of the column)
    2. count (the number of non-null values)
    3. nunique (the number of unique values in this column)

    if no numeric or categorical columns, the report will be dummy report.
    '''

    features: list[str] = Field.table_column_attr(
        "input_ds",
        desc="perform statistics on these columns",
        limit=Interval.closed(1, None),
    )

    input_ds: Input = Field.input(
        desc="Input table.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    report: Output = Field.output(
        desc="Output table statistics report.",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        input_df = ctx.load_table(self.input_ds, columns=self.features).to_pandas(
            check_null=False
        )

        with ctx.tracer.trace_running():
            stat = table_statistics(input_df)
            categorical_stat = categorical_statistics(input_df)
        if stat.empty:
            stat = pd.DataFrame({'dummy': [0]})
        if categorical_stat.empty:
            categorical_stat = pd.DataFrame({'dummy': [0]})
        stat_tbl = Reporter.build_table(stat.astype(str), index=stat.index.tolist())
        categorical_stat_tbl = Reporter.build_table(
            categorical_stat.astype(str), index=categorical_stat.index.tolist()
        )
        r = Reporter(name="table statistics", system_info=self.input_ds.system_info)
        r.add_tab(stat_tbl)
        r.add_tab(categorical_stat_tbl)
        self.report.data = r.to_distdata()
        logging.info(f'\n--\n*report* \n\n{MessageToJson(r.report())}\n--\n')
