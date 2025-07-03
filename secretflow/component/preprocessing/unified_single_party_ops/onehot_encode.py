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


from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
from secretflow_spec.v1.component_pb2 import Attribute
from secretflow_spec.v1.data_pb2 import DistData
from secretflow_spec.v1.report_pb2 import Descriptions, Div, Report, Tab

import secretflow.compute as sc
from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    IServingExporter,
    Output,
    ServingBuilder,
    UnionSelection,
    VTable,
    VTableUtils,
    float_almost_equal,
    register,
)
from secretflow.device import PYUObject
from secretflow.device.driver import reveal
from secretflow.utils.errors import InvalidArgumentError

from ..preprocessing import PreprocessingMixin


def apply_onehot_rule_on_table(table: sc.Table, additional_info: Dict) -> sc.Table:
    for col_name in additional_info:
        col_additional_info = additional_info[col_name]
        col = table.column(col_name)
        col_field = table.field(col_name)
        table = table.remove_column(col_name)
        for idx, rule in enumerate(col_additional_info):
            assert len(rule)
            onehot_cond = None
            for v in rule:
                if isinstance(v, float) or isinstance(v, np.floating):
                    cond = float_almost_equal(col, v)
                else:
                    cond = sc.equal(col, v)
                if onehot_cond is None:
                    onehot_cond = cond
                else:
                    onehot_cond = sc.or_(onehot_cond, cond)

            new_col = sc.if_else(onehot_cond, np.float32(1), np.float32(0))
            new_field = VTableUtils.pa_field_from(
                f"{col_name}_{idx}", new_col.dtype, col_field
            )
            table = table.append_column(new_field, new_col)

    return table


def fit_col(
    col_name: str, col: pa.ChunkedArray, min_frequency: int, drop_mode: str
) -> Tuple[List, List]:
    # remove null / nan
    col = pc.filter(col, pc.invert(pc.is_null(col, nan_is_null=True)))
    if len(col) == 0:
        raise InvalidArgumentError(
            "the feature only contains null and nan",
            detail={"feature": col_name},
        )
    value_counts = pc.value_counts(col)
    if len(value_counts) >= 100:
        raise InvalidArgumentError(
            "the feature has too many categories",
            detail={"feature": col_name, "categories": len(value_counts)},
        )
    if drop_mode == "mode":
        value_counts = value_counts.sort(order="descending", by=1)
        drop_category = value_counts[0][0].as_py()
    elif drop_mode == "first":
        drop_category = value_counts[0][0].as_py()
    elif drop_mode == "no_drop":
        drop_category = None
    else:
        raise InvalidArgumentError(
            "unsupported drop_mode", detail={"drop_mode": drop_mode}
        )

    category = [
        [vc[0].as_py()]
        for vc in value_counts
        if vc[1].as_py() >= min_frequency and vc[0].as_py() != drop_category
    ]
    infrequent_category = [
        vc[0].as_py()
        for vc in value_counts
        if vc[1].as_py() < min_frequency and vc[0].as_py() != drop_category
    ]

    if infrequent_category:
        category.append(infrequent_category)

    return category, drop_category


def _onehot_encode_fit(trans_data: pa.Table, drop: str, min_frequency: float):
    rows = trans_data.shape[0]
    min_frequency = round(rows * min_frequency)

    onehot_rules = {}
    drop_rules = {}
    for name in trans_data.column_names:
        # TODO: streaming read and fit
        category, drop_category = fit_col(
            name, trans_data.column(name), min_frequency, drop
        )
        onehot_rules[name] = category
        if drop_category:
            drop_rules[name] = drop_category

    return onehot_rules, drop_rules


@register(domain="preprocessing", version="1.0.0")
class OnehotEncode(PreprocessingMixin, Component, IServingExporter):
    '''
    onehot_encode
    '''

    drop: str = Field.union_attr(
        desc="drop unwanted category based on selection",
        selections=[
            UnionSelection(
                name="no_drop",
                desc="do not drop",
            ),
            UnionSelection(
                name="first",
                desc="drop the first category in each feature.",
            ),
            UnionSelection(
                name="mode",
                desc="drop the mode category in each feature",
            ),
        ],
    )
    min_frequency: float = Field.attr(
        desc="Specifies the minimum frequency below which a category will be considered infrequent, [0, 1), 0 disable",
        default=0.0,
        bound_limit=Interval.closed_open(0.0, 1.0),
    )
    report_rules: bool = Field.attr(
        desc="Whether to report rule details",
        default=True,
    )
    features: list[str] = Field.table_column_attr(
        "input_ds",
        desc="Features to encode.",
    )
    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="output dataset",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_rule: Output = Field.output(
        desc="onehot rule",
        types=[DistDataType.PREPROCESSING_RULE],
    )
    report: Output = Field.output(
        desc="report rules details if report_rules is true",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        def _fit_rule(df: pa.Table, columns: list[str]) -> dict:
            onehot_rules, drop_rules = _onehot_encode_fit(
                df.select(columns), self.drop, self.min_frequency
            )
            return {"onehot_rules": onehot_rules, "drop_rules": drop_rules}

        in_tbl = VTable.from_distdata(self.input_ds)
        tran_tbl = in_tbl.select(self.features)
        df = ctx.load_table(in_tbl)

        rule: dict[str, PYUObject] = {}
        for pyu, p in df.partitions.items():
            if pyu.party not in tran_tbl.parties:
                continue
            party_columns = tran_tbl.parties[pyu.party].schema.names
            party_rule_obj = pyu(_fit_rule)(p.data, party_columns)
            rule[pyu.party] = party_rule_obj

        def _fit_model(df: sc.Table, rule: dict) -> sc.Table:
            onehot_rules = rule['onehot_rules']
            out = apply_onehot_rule_on_table(df, onehot_rules)
            return out

        model = self.fit(ctx, self.output_rule, tran_tbl, _fit_model, rule)
        self.transform(ctx, self.output_ds, df, model, streaming=False)
        self.dump_report(rule)

    def dump_report(self, dist_rules_obj: dict[str, PYUObject]):
        if self.report_rules:
            divs = []
            drop_divs = []
            for party, party_rules in dist_rules_obj.items():
                rules = reveal(party_rules)
                onehot_rules = rules["onehot_rules"]
                drop_rules = rules["drop_rules"]

                # build onehot rules
                descs = []
                for col_name in onehot_rules:
                    items = []
                    col_rules = onehot_rules[col_name]

                    for idx, rule in enumerate(col_rules):
                        items.append(
                            Descriptions.Item(
                                name=f"{col_name}_{idx}",
                                type="str",
                                value=Attribute(s=",".join(map(str, rule))),
                            )
                        )
                    descs.append(Descriptions(name=col_name, desc="", items=items))

                div = Div(
                    name=party,
                    desc="per party onehot rules",
                    children=[
                        Div.Child(type="descriptions", descriptions=d) for d in descs
                    ],
                )
                divs.append(div)

                # build drop rules
                if drop_rules is not None and len(drop_rules) > 0:
                    drop_descs = []
                    for col_name, drop_value in drop_rules.items():
                        item = Descriptions.Item(
                            name=col_name,
                            type="str",
                            value=Attribute(s=str(drop_value)),
                        )
                        drop_descs.append(
                            Descriptions(name=col_name, desc="", items=[item])
                        )
                    drop_divs.append(
                        Div(
                            name=party,
                            desc="per party drop rules",
                            children=[
                                Div.Child(type="descriptions", descriptions=d)
                                for d in drop_descs
                            ],
                        )
                    )

            tabs = [Tab(name="onehot rules", divs=divs)]
            if len(drop_divs) > 0:
                tabs.append(Tab(name="drop rules", divs=drop_divs))
            report_mate = Report(name="onehot rules", tabs=tabs)
        else:
            report_mate = Report(name="onehot rules")

        report_dd = DistData(
            name=self.report.uri,
            type=str(DistDataType.REPORT),
            system_info=self.input_ds.system_info,
        )
        report_dd.meta.Pack(report_mate)
        self.report.data = report_dd

    def export(self, ctx: Context, builder: ServingBuilder) -> None:
        self.do_export(ctx, builder, self.input_ds, self.output_rule.data)
