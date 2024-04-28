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


from typing import Dict

import numpy as np
import pandas as pd
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

onehot_encode.str_attr(
    name="drop",
    desc="""no_drop is default setting, it won't do anything.
    If it is set to first, it will drop the first category in each feature.
    If it is set to mode, it will drop the mode category in each feature.
    If only one category is present and the setting is not no_drop, the feature will be dropped entirely""",
    is_list=False,
    is_optional=True,
    default_value="no_drop",
    allowed_values=_SUPPORTED_ONEHOT_DROP,
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


def _onehot_encode_fit(trans_data: pd.DataFrame, drop: str, min_frequency: float):
    min_frequency = min_frequency if min_frequency > 0 else None
    is_drop_mode = drop == "mode"
    skdrop = "first" if drop == "first" else None

    enc = OneHotEncoder(
        drop=skdrop,
        min_frequency=min_frequency,
        handle_unknown='ignore',
        dtype=np.float32,
        sparse_output=False,
    )

    enc.fit(trans_data)

    categories = enc.categories_
    assert len(categories) == len(trans_data.columns)

    infrequent_categories = getattr(
        enc, "infrequent_categories_", [None] * len(categories)
    )

    drop_categories = [None] * len(categories)
    if is_drop_mode:
        drop_categories = trans_data.mode().iloc[0].values
    elif enc.drop_idx_ is not None:
        for feature_idx, category_idx in enumerate(enc.drop_idx_):
            drop_categories[feature_idx] = categories[feature_idx][category_idx]

    onehot_rules = {}
    for col_name, category, infrequent_category, drop_category in zip(
        trans_data.columns, categories, infrequent_categories, drop_categories
    ):
        col_rules = []

        for value in category:
            if drop_category is not None and value == drop_category:
                continue
            if infrequent_category is not None and value in infrequent_category:
                continue
            col_rules.append([value])

        if infrequent_category is not None and infrequent_category.size > 0:
            if drop_category is not None:
                infrequent_category = np.setdiff1d(infrequent_category, drop_category)
            col_rules.append(list(infrequent_category))

        assert (
            len(col_rules) < 100
        ), f"feature {col_name} has too many categories {len(col_rules)}"

        onehot_rules[col_name] = col_rules

    return onehot_rules


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

    def onehot_fit_transform(trans_data):
        onehot_rules = _onehot_encode_fit(trans_data, drop, min_frequency)
        trans_data = apply_onehot_rule_on_table(
            sc.Table.from_pandas(trans_data), onehot_rules
        )
        return trans_data, [], onehot_rules

    (output_dd, model_dd, dist_rules_obj) = v_preprocessing_transform(
        ctx,
        input_dataset,
        input_dataset_features,
        onehot_fit_transform,
        output_dataset,
        out_rules,
        "OneHot Encode",
        load_ids=False,
        assert_one_party=False,
    )

    # build report
    if report_rules:
        divs = []
        for party_rules in dist_rules_obj:
            party = party_rules.device.party
            onehot_rules = reveal(party_rules)

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
                desc="per party rules",
                children=[
                    Div.Child(type="descriptions", descriptions=d) for d in descs
                ],
            )
            divs.append(div)

        report_mate = Report(
            name="onehot rules", tabs=[Tab(name="onehot rules", divs=divs)]
        )
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
