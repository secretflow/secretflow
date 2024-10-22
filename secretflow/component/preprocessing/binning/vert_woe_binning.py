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
from secretflow.preprocessing.binning.vert_woe_binning import (
    VertWoeBinning as VertWoeBinningProcessor,
)

from .base import VertBinningBase


@register(domain="preprocessing", version="1.0.0", name="vert_woe_binning")
class VertWoeBinning(VertBinningBase):
    '''
    Generate Weight of Evidence (WOE) binning rules for vertical partitioning datasets.
    '''

    secure_device_type: str = Field.attr(
        desc="Use SPU(Secure multi-party computation or MPC) or HEU(Homomorphic encryption or HE) to secure bucket summation.",
        default="spu",
        choices=["spu", "heu"],
    )
    binning_method: str = Field.attr(
        desc="How to bin features with numeric types: "
        '"quantile"(equal frequency)/"chimerge"(ChiMerge from AAAI92-019: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)/"eq_range"(equal range)',
        default="quantile",
        choices=["quantile", "chimerge", "eq_range"],
    )
    bin_num: int = Field.attr(
        desc="Max bin counts for one features.",
        default=10,
        bound_limit=Interval.open(0, None),
    )
    positive_label: str = Field.attr(
        desc="Which value represent positive value in label.",
        default="1",
    )
    chimerge_init_bins: int = Field.attr(
        desc="Max bin counts for initialization binning in ChiMerge.",
        default=100,
        bound_limit=Interval.open(2, None),
    )
    chimerge_target_bins: int = Field.attr(
        desc="Stop merging if remaining bin counts is less than or equal to this value.",
        default=10,
        bound_limit=Interval.closed(2, None),
    )
    chimerge_target_pvalue: float = Field.attr(
        desc="Stop merging if biggest pvalue of remaining bins is greater than this value.",
        default=0.1,
        bound_limit=Interval.open_closed(0.0, 1.0),
    )
    report_rules: bool = Field.attr(
        desc="Whether report binning rules.",
        default=False,
    )
    feature_selects: list[str] = Field.table_column_attr(
        "input_ds",
        desc="which features should be binned. WARNING: WOE won't be effective for features with enumeration count <=2.",
        limit=Interval.closed(1, None),
    )
    label: str = Field.table_column_attr(
        "input_ds",
        desc="Label of input data.",
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
        desc="Output WOE rule.",
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
        label_tbl = input_tbl.select([self.label])
        label_party = label_tbl.party(0).party

        if self.secure_device_type == "spu":
            secure_device = ctx.make_spu()
        elif self.secure_device_type == "heu":
            secure_device = ctx.make_heu(
                label_party, [p for p in trans_tbl.parties.keys()]
            )
        else:
            raise ValueError(
                f"unsupported secure_device_type {self.secure_device_type}"
            )

        input_df = ctx.load_table(input_tbl)
        with ctx.trace_running():
            bining = VertWoeBinningProcessor(secure_device)
            bin_names = {
                PYU(party): p.columns for party, p in trans_tbl.parties.items()
            }
            rules = bining.binning(
                input_df[self.feature_selects + [self.label]],
                self.binning_method,
                self.bin_num,
                bin_names,
                self.label,
                self.positive_label,
                self.chimerge_init_bins,
                self.chimerge_target_bins,
                self.chimerge_target_pvalue,
            )

        self.do_evaluate(
            ctx, self.output_ds, self.output_rule, input_df, trans_tbl, rules
        )

        self.dump_report(
            self.report, rules, self.report_rules, self.input_ds.system_info
        )

    def export(self, ctx: Context, builder: ServingBuilder) -> None:
        self.do_export(ctx, builder, self.input_ds, self.output_rule.data)
