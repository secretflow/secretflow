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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.∏
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from typing import Tuple, Dict, List

import secretflow.compute as sc
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import (
    DistDataType,
    VerticalTableWrapper,
    dump_vertical_table,
    load_table,
    model_dumps,
    model_loads,
)
from secretflow.data.core import partition
from secretflow.device.driver import reveal
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab

from .table_utils import apply_onehot_rule_on_table

onehot_encode = Component(
    "onehot_encode",
    domain="preprocessing",
    version="0.0.2",
    desc="onehot_encode",
)

onehot_encode.bool_attr(
    name="drop_first",
    desc="If true drop the first category in each feature. If only one category is present, the feature will be dropped entirely",
    is_list=False,
    is_optional=True,
    default_value=False,
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
    col_params=[TableColParam(name="encode_features", desc="Features to encode.")],
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
    types=[DistDataType.ONEHOT_RULE],
    col_params=None,
)

onehot_encode.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="report rules details if report_rules is true",
    types=[DistDataType.REPORT],
)


# current version 0.1
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 2


def _update_meta(
    meta: VerticalTableWrapper, changes: Dict[str, Tuple[List, List]]
) -> VerticalTableWrapper:
    for party, (drop_key, append_key) in changes.items():
        table_schema = meta.schema_map[party]
        new_schema = TableSchema()
        new_schema.CopyFrom(table_schema)
        new_schema.features[:] = []
        new_schema.feature_types[:] = []
        for idx in range(len(table_schema.features)):
            if table_schema.features[idx] not in drop_key:
                new_schema.features.append(table_schema.features[idx])
                new_schema.feature_types.append(table_schema.feature_types[idx])
        for new_k in append_key:
            new_schema.features.append(new_k)
            new_schema.feature_types.append("float32")
        meta.schema_map[party] = new_schema
    return meta


@onehot_encode.eval_fn
def onehot_encode_eval_fn(
    *,
    ctx,
    drop_first,
    min_frequency,
    report_rules,
    input_dataset,
    input_dataset_encode_features,
    out_rules,
    output_dataset,
    report,
):
    assert (
        input_dataset.type == DistDataType.VERTICAL_TABLE
    ), "only support vtable for now"

    trans_x = load_table(
        ctx,
        input_dataset,
        load_features=True,
        feature_selects=input_dataset_encode_features,
    ).to_pandas()

    remain_x = load_table(
        ctx,
        input_dataset,
        load_features=True,
        load_ids=True,
        load_labels=True,
        col_excludes=input_dataset_encode_features,
    ).to_pandas()

    drop = 'first' if drop_first else None
    min_frequency = min_frequency if min_frequency > 0 else None

    def _fit(trans_data):
        enc = OneHotEncoder(
            drop=drop,
            min_frequency=min_frequency,
            handle_unknown='ignore',
            dtype=np.float32,
            sparse=False,
        )
        enc.fit(trans_data)

        categories = enc.categories_
        assert len(categories) == len(trans_data.columns)
        infrequent_categories = getattr(
            enc, "infrequent_categories_", [None] * len(categories)
        )
        onehot_rules = {}
        for col_name, category, infrequent_category in zip(
            trans_data.columns, categories, infrequent_categories
        ):
            col_rules = []

            def infrequent_check(c):
                if infrequent_category is not None:
                    return c in infrequent_category
                else:
                    return False

            infrequent_is_first = True
            for k, value in enumerate(category):
                if infrequent_check(value):
                    continue
                infrequent_is_first = False
                if enc.drop is not None and k == 0:
                    continue
                col_rules.append([value])

            if infrequent_category is not None and (
                enc.drop is None or infrequent_is_first is False
            ):
                col_rules.append(list(infrequent_category))

            onehot_rules[col_name] = col_rules
        return onehot_rules

    def fit_transform(trans_data, remain_data):
        assert trans_data is not None
        trans_columns = trans_data.columns
        onehot_rules = _fit(trans_data)
        trans_data = apply_onehot_rule_on_table(
            sc.Table.from_pandas(trans_data), onehot_rules
        )
        runner = trans_data.dump_runner()
        trans_data = trans_data.to_pandas()
        append_columns = trans_data.columns
        if remain_data is not None:
            new_data = pd.concat([remain_data, trans_data], axis=1)
        else:
            new_data = trans_data

        return new_data, trans_columns, append_columns, onehot_rules, runner

    cols_change = {}
    new_datas = {}
    dist_rules_obj = []
    runner_obj = []

    for pyu in trans_x.partitions.keys():
        data = trans_x.partitions[pyu].data
        if pyu in remain_x.partitions.keys():
            remain_data = remain_x.partitions.pop(pyu).data
        else:
            remain_data = None

        new_data, drop_key, append_key, onehot_rules, runner = pyu(
            fit_transform, num_returns=5
        )(data, remain_data)

        new_datas[pyu] = new_data
        cols_change[pyu.party] = [drop_key, append_key]
        dist_rules_obj.append(onehot_rules)
        runner_obj.append(runner)

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
                desc="pre party rules",
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

    # build output dataset
    for pyu in remain_x.partitions.keys():
        trans_x.partitions[pyu] = remain_x.partitions[pyu]
    for pyu in new_datas:
        trans_x.partitions[pyu] = partition(new_datas[pyu])
    cols_change = reveal(cols_change)
    meta = VerticalTableWrapper.from_dist_data(input_dataset, trans_x.shape[0])
    meta = _update_meta(meta, cols_change)

    output_dd = dump_vertical_table(
        ctx,
        trans_x,
        output_dataset,
        meta,
        input_dataset.system_info,
    )

    # build rules for onehot_substitution
    model_dd = model_dumps(
        "onehot",
        DistDataType.ONEHOT_RULE,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        runner_obj,
        "",
        ctx.local_fs_wd,
        out_rules,
        input_dataset.system_info,
    )

    return {"out_rules": model_dd, "report": report_dd, "output_dataset": output_dd}


onehot_substitution = Component(
    "onehot_substitution",
    domain="preprocessing",
    version="0.0.2",
    desc="onehot_substitution",
)

onehot_substitution.io(
    io_type=IoType.INPUT,
    name="input_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)

onehot_substitution.io(
    io_type=IoType.INPUT,
    name="input_rules",
    desc="Input onehot rules",
    types=[DistDataType.ONEHOT_RULE],
    col_params=None,
)

onehot_substitution.io(
    io_type=IoType.OUTPUT,
    name="output_dataset",
    desc="output_dataset",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)


@onehot_substitution.eval_fn
def onehot_substitution_eval_fn(
    *,
    ctx,
    input_dataset,
    input_rules,
    output_dataset,
):
    assert (
        input_dataset.type == DistDataType.VERTICAL_TABLE
    ), "only support vtable for now"

    x = load_table(
        ctx,
        input_dataset,
        load_features=True,
        load_ids=True,
        load_labels=True,
    ).to_pandas()
    pyus = {p.party: p for p in x.partitions.keys()}

    trace_runner_objs, _ = model_loads(
        input_rules,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.ONEHOT_RULE,
        # only local fs is supported at this moment.
        ctx.local_fs_wd,
        pyus=pyus,
    )

    assert set([o.device for o in trace_runner_objs]).issubset(set(x.partitions.keys()))

    def transform(data, runner):
        trans_columns = list(runner.get_input_features())
        assert set(trans_columns).issubset(
            set(data.columns)
        ), f"can not find rule keys {trans_columns} in dataset columns {data.columns}"
        append_columns = []

        if len(trans_columns) > 0:
            trans_data = data[trans_columns]
            remain_data = data.drop(trans_columns, axis=1)

            trans_data = runner.run(trans_data)

            append_columns = list(trans_data.columns)
            data = pd.concat([remain_data, trans_data], axis=1)

        return data, trans_columns, append_columns

    cols_change = {}
    new_datas = {}
    for r in trace_runner_objs:
        pyu = r.device
        new_data, drop_key, append_key = pyu(transform, num_returns=3)(
            x.partitions[pyu].data, r
        )
        new_datas[pyu] = new_data
        cols_change[pyu.party] = [drop_key, append_key]

    for pyu in new_datas:
        x.partitions[pyu] = partition(new_datas[pyu])

    cols_change = reveal(cols_change)
    meta = VerticalTableWrapper.from_dist_data(input_dataset, x.shape[0])
    meta = _update_meta(meta, cols_change)

    output_dd = dump_vertical_table(
        ctx,
        x,
        output_dataset,
        meta,
        input_dataset.system_info,
    )

    return {"output_dataset": output_dd}
