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

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import (
    DistDataType,
    dump_vertical_table,
    load_table,
    model_dumps,
    model_loads,
    move_feature_to_label,
    VerticalTableWrapper,
)
from secretflow.device.device.pyu import PYUObject
from secretflow.preprocessing.binning.vert_bin_substitution import VertBinSubstitution
from secretflow.preprocessing.binning.vert_binning import VertBinning

vert_binning_comp = Component(
    "vert_binning",
    domain="feature",
    version="0.0.1",
    desc="Generate equal frequency or equal range binning rules for vertical partitioning datasets.",
)

vert_binning_comp.str_attr(
    name="binning_method",
    desc="How to bin features with numeric types: "
    '"quantile"(equal frequency)/"eq_range"(equal range)',
    is_list=False,
    is_optional=True,
    default_value="eq_range",
    allowed_values=["eq_range", "quantile"],
)
vert_binning_comp.int_attr(
    name="bin_num",
    desc="Max bin counts for one features.",
    is_list=False,
    is_optional=True,
    default_value=10,
    lower_bound=0,
    lower_bound_inclusive=False,
)

vert_binning_comp.bool_attr(
    name="select_all_features",
    desc="Select all features for binning.",
    is_list=False,
    is_optional=True,
    default_value=False,
)

vert_binning_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="feature_selects",
            desc="which features should be binned.",
            col_min_cnt_inclusive=0,
        )
    ],
)
vert_binning_comp.io(
    io_type=IoType.OUTPUT,
    name="bin_rule",
    desc="Output bin rule.",
    types=[DistDataType.BIN_RUNNING_RULE],
    col_params=None,
)

# current version 0.1
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 1


@vert_binning_comp.eval_fn
def vert_binning_eval_fn(
    *,
    ctx,
    binning_method,
    bin_num,
    select_all_features,
    input_data,
    input_data_feature_selects,
    bin_rule,
):
    assert (
        select_all_features or len(input_data_feature_selects) > 0
    ), "select at least one feature"
    if select_all_features:
        input_df = load_table(
            ctx,
            input_data,
            load_features=True,
            load_labels=False,
        )
        input_data_feature_selects = input_df.columns
    else:
        input_df = load_table(
            ctx,
            input_data,
            load_features=True,
            feature_selects=input_data_feature_selects,
            load_labels=False,
        )
    with ctx.tracer.trace_running():
        bining = VertBinning()
        col_index = input_df._col_index(input_data_feature_selects)
        bin_names = {
            party: bins if isinstance(bins, list) else [bins]
            for party, bins in col_index.items()
        }
        rules = bining.binning(input_df, binning_method, bin_num, bin_names)

        model_dist_data = model_dumps(
            "bin_rule",
            DistDataType.BIN_RUNNING_RULE,
            MODEL_MAX_MAJOR_VERSION,
            MODEL_MAX_MINOR_VERSION,
            [o for o in rules.values()],
            {
                "input_data_feature_selects": input_data_feature_selects,
            },
            ctx.local_fs_wd,
            bin_rule,
            input_data.system_info,
        )

    return {"bin_rule": model_dist_data}


vert_bin_substitution_comp = Component(
    "vert_bin_substitution",
    domain="feature",
    version="0.0.1",
    desc="Substitute datasets' value by bin substitution rules.",
)

vert_bin_substitution_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Vertical partitioning dataset to be substituted.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)
vert_bin_substitution_comp.io(
    io_type=IoType.INPUT,
    name="bin_rule",
    desc="Input bin substitution rule.",
    types=[DistDataType.BIN_RUNNING_RULE],
    col_params=None,
)
vert_bin_substitution_comp.io(
    io_type=IoType.OUTPUT,
    name="output_data",
    desc="Output vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)


@vert_bin_substitution_comp.eval_fn
def vert_bin_substitution_eval_fn(
    *,
    ctx,
    input_data,
    bin_rule,
    output_data,
):
    input_df = load_table(
        ctx, input_data, load_features=True, load_labels=True, load_ids=True
    )

    pyus = {p.party: p for p in input_df.partitions}

    model_objs, public_info = model_loads(
        bin_rule,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.BIN_RUNNING_RULE,
        ctx.local_fs_wd,
        pyus=pyus,
    )

    bin_rule = {}
    for obj in model_objs:
        assert isinstance(obj, PYUObject)
        bin_rule[obj.device] = obj

    with ctx.tracer.trace_running():
        output_df = VertBinSubstitution().substitution(input_df, bin_rule)

    vt_wrapper = VerticalTableWrapper.from_dist_data(input_data, output_df.shape[0])

    # modify types of feature_selects to float
    for v in vt_wrapper.schema_map.values():
        for i, f in enumerate(list(v.features)):
            if f in public_info['input_data_feature_selects']:
                v.feature_types[i] = 'float'

    # change cols from feature to label according to model public info.
    if 'input_data_label' in public_info:
        for k, v in vt_wrapper.schema_map.items():
            vt_wrapper.schema_map[k] = move_feature_to_label(
                v, public_info['input_data_label']
            )

    return {
        "output_data": dump_vertical_table(
            ctx,
            output_df,
            output_data,
            vt_wrapper,
            input_data.system_info,
        )
    }
