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
import pandas as pd
import pyarrow as pa
from pyarrow import compute as pc
from sklearn.preprocessing import OneHotEncoder

import secretflow.compute as sc
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.core.table_utils import (
    float_almost_equal,
    v_preprocessing_transform,
)
from secretflow.device.driver import reveal
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab

onehot_encode = Component(
    "onehot_encode",
    domain="preprocessing",
    version="0.0.3",
    desc="onehot_encode",
)

_SUPPORTED_ONEHOT_DROP = ["no_drop", "first", "mode"]

onehot_encode.union_attr_group(
    name="drop",
    desc="drop unwanted category based on selection",
    group=[
        onehot_encode.union_selection_attr(
            name="no_drop",
            desc="do not drop",
        ),
        onehot_encode.union_selection_attr(
            name="first",
            desc="drop the first category in each feature.",
        ),
        onehot_encode.union_selection_attr(
            name="mode",
            desc="drop the mode category in each feature",
        ),
    ],
)

onehot_encode.float_attr(
    name="min_frequency",
    desc="Specifies the minimum frequency below which a category will be considered infrequent, [0, 1), 0 disable",
    is_list=False,
    is_optional=True,
    default_value=0.0,
    lower_bound=0.0,
    upper_bound=1.0,
    lower_bound_inclusive=True,
    upper_bound_inclusive=False,
)

onehot_encode.bool_attr(
    name="report_rules",
    desc="Whether to report rule details",
    is_list=False,
    is_optional=True,
    default_value=True,
)

onehot_encode.io(
    io_type=IoType.INPUT,
    name="input_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[TableColParam(name="features", desc="Features to encode.")],
)

onehot_encode.io(
    io_type=IoType.OUTPUT,
    name="output_dataset",
    desc="output_dataset",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)

onehot_encode.io(
    io_type=IoType.OUTPUT,
    name="out_rules",
    desc="onehot rule",
    types=[DistDataType.PREPROCESSING_RULE],
    col_params=None,
)

onehot_encode.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="report rules details if report_rules is true",
    types=[DistDataType.REPORT],
)


def apply_onehot_rule_on_table(table: sc.Table, additional_info: Dict) -> sc.Table:
    for col_name in additional_info:
        col_additional_info = additional_info[col_name]
        col = table.column(col_name)
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
            table = table.append_column(f"{col_name}_{idx}", new_col)

    return table


def fit_col(
    col_name: str, col: pa.ChunkedArray, min_frequency: int, drop_mode: str
) -> Tuple[List, List]:
    # remove null / nan
    col = pc.filter(col, pc.invert(pc.is_null(col, nan_is_null=True)))
    if len(col) == 0:
        raise RuntimeError(
            f"feature {col_name} contains only null and nan, can not onehotencode on this feature"
        )
    value_counts = pc.value_counts(col)
    if len(value_counts) >= 100:
        raise RuntimeError(
            f"feature {col_name} has too many categories {len(value_counts)}"
        )
    if drop_mode == "mode":
        value_counts = value_counts.sort(order="descending", by=1)
        drop_category = value_counts[0][0].as_py()
    elif drop_mode == "first":
        drop_category = value_counts[0][0].as_py()
    elif drop_mode == "no_drop":
        drop_category = None
    else:
        raise AttributeError(f"unknown drop_mode {drop_mode}")

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


@onehot_encode.eval_fn
def onehot_encode_eval_fn(
    *,
    ctx,
    drop,
    min_frequency,
    report_rules,
    input_dataset,
    input_dataset_features,
    out_rules,
    output_dataset,
    report,
):
    assert (
        input_dataset.type == DistDataType.VERTICAL_TABLE
    ), "only support vtable for now"

    assert drop in _SUPPORTED_ONEHOT_DROP, f"unsupported drop type {drop}"

    def onehot_fit_transform(trans_data: pa.Table):
        onehot_rules, drop_rules = _onehot_encode_fit(trans_data, drop, min_frequency)
        trans_data = apply_onehot_rule_on_table(
            sc.Table.from_pyarrow(trans_data), onehot_rules
        )
        return trans_data, [], {"onehot_rules": onehot_rules, "drop_rules": drop_rules}

    (output_dd, model_dd, dist_rules_obj) = v_preprocessing_transform(
        ctx,
        input_dataset,
        input_dataset_features,
        onehot_fit_transform,
        output_dataset,
        out_rules,
        "OneHot Encode",
        assert_one_party=False,
    )

    # build report
    if report_rules:
        divs = []
        drop_divs = []
        for party_rules in dist_rules_obj:
            party = party_rules.device.party
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
        name=report,
        type=str(DistDataType.REPORT),
        system_info=input_dataset.system_info,
    )
    report_dd.meta.Pack(report_mate)

    return {
        "out_rules": model_dd,
        "output_dataset": output_dd,
        "report": report_dd,
    }
