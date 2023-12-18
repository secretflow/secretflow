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


import pandas as pd

from secretflow.component.component import Component, IoType
from secretflow.component.data_utils import (
    DistDataType,
    dump_vertical_table,
    load_table,
    model_loads,
    VerticalTableWrapper,
)
from secretflow.component.preprocessing.core.meta_utils import (
    apply_meta_change,
    str_to_dict,
)
from secretflow.component.preprocessing.core.version import (
    PREPROCESSING_RULE_MAX_MAJOR_VERSION,
    PREPROCESSING_RULE_MAX_MINOR_VERSION,
)
from secretflow.data.core import partition


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

    x = load_table(
        ctx,
        input_dataset,
        load_features=True,
        load_ids=True,
        load_labels=True,
    ).to_pandas()
    pyus = {p.party: p for p in x.partitions.keys()}

    trace_runner_objs, meta_change_str = model_loads(
        input_rules,
        PREPROCESSING_RULE_MAX_MAJOR_VERSION,
        PREPROCESSING_RULE_MAX_MINOR_VERSION,
        DistDataType.PREPROCESSING_RULE,
        # only local fs is supported at this moment.
        ctx.local_fs_wd,
        pyus=pyus,
    )

    assert set([o.device for o in trace_runner_objs]).issubset(set(x.partitions.keys()))

    meta_change_dict = str_to_dict(meta_change_str)

    def transform(data, runner):
        trans_columns = list(runner.get_input_features())
        assert set(trans_columns).issubset(
            set(data.columns)
        ), f"can not find rule keys {trans_columns} in dataset columns {data.columns}"

        if len(trans_columns) > 0:
            trans_data = data[trans_columns]
            remain_data = data.drop(trans_columns, axis=1)

            trans_data = runner.run(trans_data)

            data = pd.concat([remain_data, trans_data], axis=1)

        return data

    new_datas = {}
    for r in trace_runner_objs:
        pyu = r.device
        new_data = pyu(transform)(x.partitions[pyu].data, r)
        new_datas[pyu] = new_data

    for pyu in new_datas:
        x.partitions[pyu] = partition(new_datas[pyu])

    meta = VerticalTableWrapper.from_dist_data(input_dataset, x.shape[0])
    meta = apply_meta_change(meta, meta_change_dict)

    output_dd = dump_vertical_table(
        ctx,
        x,
        output_dataset,
        meta,
        input_dataset.system_info,
    )

    return {"output_dataset": output_dd}
