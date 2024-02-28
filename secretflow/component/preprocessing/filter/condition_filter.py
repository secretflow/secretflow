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

import math

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import (
    DistDataType,
    dump_vertical_table,
    load_table,
    VerticalTableWrapper,
)
from secretflow.preprocessing.cond_filter_v import ConditionFilter, conversion_mapping

condition_filter_comp = Component(
    "condition_filter",
    domain="data_filter",
    version="0.0.1",
    desc="""Filter the table based on a single column's values and condition.
    Warning: the party responsible for condition filtering will directly send the sample distribution to other participants.
    Malicious participants can obtain the distribution of characteristics by repeatedly calling with different filtering values.
    Audit the usage of this component carefully.
    """,
)

condition_filter_comp.str_attr(
    name="comparator",
    desc="Comparator to use for comparison. Must be one of '==','<','<=','>','>=','IN'",
    is_list=False,
    is_optional=False,
    allowed_values=['==', '<', '<=', '>', '>=', 'IN'],
)

condition_filter_comp.str_attr(
    name="value_type",
    desc=f"Type of the value to compare with. Must be one of {list(conversion_mapping.keys())}",
    is_list=False,
    is_optional=False,
    allowed_values=list(conversion_mapping.keys()),
)
condition_filter_comp.str_attr(
    name="bound_value",
    desc="Input a str with values separated by ','. List of values to compare with. If comparator is not 'IN', we only support one element in this list.",
    is_optional=False,
    is_list=False,
)
condition_filter_comp.float_attr(
    name="float_epsilon",
    desc="Epsilon value for floating point comparison. WARNING: due to floating point representation in computers, set this number slightly larger if you want filter out the values exactly at desired boundary. for example, abs(1.001 - 1.002) is slightly larger than 0.001, and therefore may not be filter out using == and epsilson = 0.001",
    is_list=False,
    is_optional=True,
    lower_bound=0,
    lower_bound_inclusive=True,
    default_value=0.000001,
)

condition_filter_comp.io(
    io_type=IoType.INPUT,
    name="in_ds",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="features",
            desc="Feature(s) to operate on.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        )
    ],
)

condition_filter_comp.io(
    io_type=IoType.OUTPUT,
    name="out_ds",
    desc="Output vertical table that satisfies the condition.",
    types=[DistDataType.VERTICAL_TABLE],
)

condition_filter_comp.io(
    io_type=IoType.OUTPUT,
    name="out_ds_else",
    desc="Output vertical table that does not satisfies the condition.",
    types=[DistDataType.VERTICAL_TABLE],
)


@condition_filter_comp.eval_fn
def condition_filter_comp_eval_fn(
    *,
    ctx,
    comparator,
    value_type,
    bound_value,
    float_epsilon,
    in_ds,
    in_ds_features,
    out_ds,
    out_ds_else,
):
    # Load data from train_dataset
    x = load_table(ctx, in_ds, load_features=True, load_ids=True, load_labels=True)
    bound_value_list = bound_value.split(",")
    # Initialize and run training algorithm
    with ctx.tracer.trace_running():
        filter = ConditionFilter(
            field_name=in_ds_features[0],
            comparator=comparator,
            value_type=value_type,
            bound_value=bound_value_list,
            float_epsilon=float_epsilon,
        )
        ds = filter.fit_transform(x)
        else_ds = filter.get_else_table()

    assert math.prod(
        ds.shape
    ), f"empty dataset is not allowed, yet the table satisfied the condition is empty, \
    skip this condition filter step and use alternative pipeline please."

    assert math.prod(
        else_ds.shape
    ), f"empty dataset is not allowed, yet the table not satisfied the condition is empty, \
    skip this condition filter step and use alternative pipeline please."

    out_db = dump_vertical_table(
        ctx,
        ds,
        out_ds,
        VerticalTableWrapper.from_dist_data(in_ds, ds.shape[0]),
        in_ds.system_info,
    )

    out_db_else = dump_vertical_table(
        ctx,
        else_ds,
        out_ds_else,
        VerticalTableWrapper.from_dist_data(in_ds, else_ds.shape[0]),
        in_ds.system_info,
    )

    return {"out_ds": out_db, "out_ds_else": out_db_else}
