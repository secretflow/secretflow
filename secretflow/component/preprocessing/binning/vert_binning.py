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
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    ServingBuilder,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.device import PYU
from secretflow.preprocessing.binning.vert_binning import (
    VertBinning as VertBinningProcessor,
)

from .base import VertBinningBase


@register(domain="preprocessing", version="1.0.0", name="vert_binning")
class VertBinning(VertBinningBase):
    '''
    Generate equal frequency or equal range binning rules for vertical partitioning datasets.
    '''

    binning_method: str = Field.attr(
        desc="How to bin features with numeric types: "
        '"quantile"(equal frequency)/"eq_range"(equal range)',
        default="eq_range",
        choices=["eq_range", "quantile"],
    )
    bin_num: int = Field.attr(
        desc="Max bin counts for one features.",
        default=10,
        bound_limit=Interval.closed(2, None),
    )
    report_rules: bool = Field.attr(
        desc="Whether report binning rules.",
        default=False,
    )
    feature_selects: list[str] = Field.table_column_attr(
        "input_ds",
        desc="which features should be binned.",
        limit=Interval.closed(1, None),
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_rule: Output = Field.output(
        desc="Output bin rule.",
        types=[DistDataType.BINNING_RULE],
    )
    report: Output = Field.output(
        desc="report rules details if report_rules is true",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        input_tbl = VTable.from_distdata(self.input_ds)
        trans_tbl = input_tbl.select(self.feature_selects)
        trans_tbl.check_kinds(VTableFieldKind.FEATURE)
        input_df = ctx.load_table(input_tbl)
        with ctx.trace_running():
            bining = VertBinningProcessor()
            bin_names = {
                PYU(party): p.columns for party, p in trans_tbl.parties.items()
            }
            rules = bining.binning(
                input_df[trans_tbl.columns],
                self.binning_method,
                self.bin_num,
                bin_names,
            )

        self.do_evaluate(
            ctx, self.output_ds, self.output_rule, input_df, trans_tbl, rules
        )

        self.dump_report(
            self.report, rules, self.report_rules, self.input_ds.system_info
        )

    def export(self, ctx: Context, builder: ServingBuilder) -> None:
        self.do_export(ctx, builder, self.input_ds, self.output_rule.data)
