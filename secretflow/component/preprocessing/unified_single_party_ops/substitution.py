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

import copy
import json
from typing import List, Tuple

import pyarrow as pa

from secretflow.component.component import Component, IoType
from secretflow.component.data_utils import DistDataType, model_loads
from secretflow.component.dataframe import CompDataFrame
from secretflow.component.preprocessing.core.version import (
    PREPROCESSING_RULE_MAX_MAJOR_VERSION,
    PREPROCESSING_RULE_MAX_MINOR_VERSION,
)
from secretflow.data.core import partition
from secretflow.device import reveal

substitution = Component(
    "substitution",
    domain="preprocessing",
    version="0.0.2",
    desc="unified substitution component",
)

substitution.io(
    io_type=IoType.INPUT,
    name="input_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)

substitution.io(
    io_type=IoType.INPUT,
    name="input_rules",
    desc="Input preprocessing rules",
    types=[DistDataType.PREPROCESSING_RULE],
    col_params=None,
)

substitution.io(
    io_type=IoType.OUTPUT,
    name="output_dataset",
    desc="output_dataset",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)


@substitution.eval_fn
def substitution_eval_fn(
    *,
    ctx,
    input_dataset,
    input_rules,
    output_dataset,
):
    assert (
        input_dataset.type == DistDataType.VERTICAL_TABLE
    ), "only support vtable for now"

    x = CompDataFrame.from_distdata(
        ctx,
        input_dataset,
        load_features=True,
        load_ids=True,
        load_labels=True,
    )
    pyus = {p.party: p for p in x.partitions.keys()}

    trace_runner_objs, add_labels = model_loads(
        ctx,
        input_rules,
        PREPROCESSING_RULE_MAX_MAJOR_VERSION,
        PREPROCESSING_RULE_MAX_MINOR_VERSION,
        DistDataType.PREPROCESSING_RULE,
        pyus=pyus,
    )

    assert set([o.device for o in trace_runner_objs]).issubset(set(x.partitions.keys()))

    add_labels = json.loads(add_labels)

    def transform(data, runner) -> Tuple[pa.Table, List, List]:
        trans_columns = list(runner.get_input_features())
        assert set(trans_columns).issubset(
            set(data.column_names)
        ), f"can not find rule keys {trans_columns} in dataset columns {data.column_names}"

        if len(trans_columns) > 0:
            trans_data = data.select(trans_columns)
            remain_data = data.drop(trans_columns)

            trans_data = runner.run(trans_data)
            drop_columns, add_columns, _ = runner.column_changes()

            for i in range(trans_data.shape[1]):
                remain_data = remain_data.append_column(
                    trans_data.field(i), trans_data.column(i)
                )

        return remain_data, drop_columns, add_columns

    new_datas = {}
    drop_columns = {}
    add_columns = {}
    for r in trace_runner_objs:
        pyu = r.device
        new_data, drop_column, add_column = pyu(transform)(x.data(pyu), r)
        new_datas[pyu] = new_data
        drop_columns[pyu] = drop_column
        add_columns[pyu] = add_column

    drop_columns = reveal(drop_columns)
    add_columns = reveal(add_columns)

    new_partitions = copy.deepcopy(x.partitions)
    for pyu in new_datas:
        new_partitions[pyu].data = new_datas[pyu]
        drop_col = set(drop_columns[pyu])
        add_label = add_labels[pyu.party]
        add_feature = [c for c in add_columns[pyu] if c not in add_label]
        orig_feature = new_partitions[pyu].feature_cols
        new_partitions[pyu].feature_cols = [
            c for c in orig_feature if c not in drop_col
        ] + add_feature
        orig_label = new_partitions[pyu].label_cols
        new_partitions[pyu].label_cols = [
            c for c in orig_label if c not in drop_col
        ] + add_label

    new_ds = CompDataFrame(new_partitions, x.system_info)

    return {"output_dataset": new_ds.to_distdata(ctx, output_dataset)}
