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
    register,
)
from secretflow.stats.table_statistics import table_statistics


@register(domain="stats", version="1.0.0")
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
    '''

    features: list[str] = Field.table_column_attr(
        "input_ds",
        desc="perform statistics on these columns",
        limit=Interval.closed(1, None),
    )

    input_ds: Input = Field.input(  # type: ignore
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

        r = Reporter(name="table statistics")
        r.add_tab(stat.astype(str))
        r.dump_to(self.report, self.input_ds.system_info)
