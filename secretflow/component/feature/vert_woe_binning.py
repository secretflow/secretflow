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


import spu

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import (
    DistDataType,
    VerticalTableWrapper,
    dump_vertical_table,
    extract_table_header,
    load_table,
    model_dumps,
    model_loads,
)
from secretflow.device.device.heu import HEU
from secretflow.device.device.pyu import PYUObject
from secretflow.device.device.spu import SPU
from secretflow.preprocessing.binning.vert_woe_binning import VertWoeBinning
from secretflow.preprocessing.binning.vert_woe_substitution import VertWOESubstitution

vert_woe_binning_comp = Component(
    "vert_woe_binning",
    domain="feature",
    version="0.0.1",
    desc="Generate Weight of Evidence (WOE) binning rules for vertical partitioning datasets.",
)

vert_woe_binning_comp.str_attr(
    name="secure_device_type",
    desc="Use SPU(Secure multi-party computation or MPC) or HEU(Homomorphic encryption or HE) to secure bucket summation.",
    is_list=False,
    is_optional=True,
    default_value="spu",
    allowed_values=["spu", "heu"],
)
vert_woe_binning_comp.str_attr(
    name="binning_method",
    desc="How to bin features with numeric types: "
    '"quantile"(equal frequency)/"chimerge"(ChiMerge from AAAI92-019: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)',
    is_list=False,
    is_optional=True,
    default_value="quantile",
    allowed_values=["quantile", "chimerge"],
)
vert_woe_binning_comp.int_attr(
    name="bin_num",
    desc="Max bin counts for one features.",
    is_list=False,
    is_optional=True,
    default_value=10,
    lower_bound=0,
    lower_bound_inclusive=False,
)
vert_woe_binning_comp.str_attr(
    name="positive_label",
    desc="Which value represent positive value in label.",
    is_list=False,
    is_optional=True,
    default_value="1",
    allowed_values=None,
)
vert_woe_binning_comp.int_attr(
    name="chimerge_init_bins",
    desc="Max bin counts for initialization binning in ChiMerge.",
    is_list=False,
    is_optional=True,
    default_value=100,
    lower_bound=2,
    lower_bound_inclusive=False,
)
vert_woe_binning_comp.int_attr(
    name="chimerge_target_bins",
    desc="Stop merging if remaining bin counts is less than or equal to this value.",
    is_list=False,
    is_optional=True,
    default_value=10,
    lower_bound=2,
    lower_bound_inclusive=True,
)
vert_woe_binning_comp.float_attr(
    name="chimerge_target_pvalue",
    desc="Stop merging if biggest pvalue of remaining bins is greater than this value.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    lower_bound_inclusive=False,
    upper_bound=1,
    upper_bound_inclusive=True,
)
vert_woe_binning_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="feature_selects",
            desc="which features should be binned.",
            col_min_cnt_inclusive=1,
        )
    ],
)
vert_woe_binning_comp.io(
    io_type=IoType.OUTPUT,
    name="woe_rule",
    desc="Output WOE rule.",
    types=[DistDataType.WOE_RUNNING_RULE],
    col_params=None,
)

# current version 0.1
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 1


@vert_woe_binning_comp.eval_fn
def vert_woe_binning_eval_fn(
    *,
    ctx,
    secure_device_type,
    binning_method,
    bin_num,
    positive_label,
    chimerge_init_bins,
    chimerge_target_bins,
    chimerge_target_pvalue,
    input_data,
    input_data_feature_selects,
    woe_rule,
):
    input_df = load_table(
        ctx,
        input_data,
        load_features=True,
        feature_selects=input_data_feature_selects,
        load_labels=True,
    )

    input_info = extract_table_header(input_data, load_labels=True)
    assert len(input_info) == 1, "only support one party has label"
    label_party = next(iter(input_info.keys()))
    smeta = input_info[label_party]
    assert len(smeta) == 1, "only support one label col"
    label_name = next(iter(smeta.keys()))

    if secure_device_type == "spu":
        if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
            raise CompEvalError("spu config is not found.")
        if len(ctx.spu_configs) > 1:
            raise CompEvalError("only support one spu")
        spu_config = next(iter(ctx.spu_configs.values()))
        secure_device = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    elif secure_device_type == "heu":
        assert ctx.heu_config is not None, "need heu config in SFClusterDesc"
        heu_config = {
            "sk_keeper": {"party": label_party},
            "evaluators": [
                {"party": p.party}
                for p in input_df.partitions
                if p.party != label_party
            ],
            "mode": ctx.heu_config["mode"],
            "he_parameters": {
                "schema": ctx.heu_config["schema"],
                "key_pair": {"generate": {"bit_size": ctx.heu_config["key_size"]}},
            },
        }
        secure_device = HEU((heu_config), spu.spu_pb2.FM64)
    else:
        raise CompEvalError(f"unsupported secure_device_type {secure_device_type}")

    with ctx.tracer.trace_running():
        bining = VertWoeBinning(secure_device)
        col_index = input_df._col_index(input_data_feature_selects)
        bin_names = {
            party: bins if isinstance(bins, list) else [bins]
            for party, bins in col_index.items()
        }
        rules = bining.binning(
            input_df,
            binning_method,
            bin_num,
            bin_names,
            label_name,
            positive_label,
            chimerge_init_bins,
            chimerge_target_bins,
            chimerge_target_pvalue,
        )

        model_dist_data = model_dumps(
            "woe_rule",
            DistDataType.WOE_RUNNING_RULE,
            MODEL_MAX_MAJOR_VERSION,
            MODEL_MAX_MINOR_VERSION,
            [o for o in rules.values()],
            input_data_feature_selects,
            ctx.local_fs_wd,
            woe_rule,
            input_data.sys_info,
        )

    return {"woe_rule": model_dist_data}


vert_woe_substitution_comp = Component(
    "vert_woe_substitution",
    domain="feature",
    version="0.0.1",
    desc="Substitute datasets' value by WOE substitution rules.",
)

vert_woe_substitution_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Vertical partitioning dataset to be substituted.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)
vert_woe_substitution_comp.io(
    io_type=IoType.INPUT,
    name="woe_rule",
    desc="Input WOE substitution rule.",
    types=[DistDataType.WOE_RUNNING_RULE],
    col_params=None,
)
vert_woe_substitution_comp.io(
    io_type=IoType.OUTPUT,
    name="output_data",
    desc="Output vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)


@vert_woe_substitution_comp.eval_fn
def vert_woe_substitution_eval_fn(
    *,
    ctx,
    input_data,
    woe_rule,
    output_data,
):
    input_df = load_table(
        ctx, input_data, load_features=True, load_labels=True, load_ids=True
    )

    pyus = {p.party: p for p in input_df.partitions}

    model_objs, feature_selects = model_loads(
        woe_rule,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.WOE_RUNNING_RULE,
        ctx.local_fs_wd,
        pyus=pyus,
    )

    woe_rule = {}
    for obj in model_objs:
        assert isinstance(obj, PYUObject)
        woe_rule[obj.device] = obj

    with ctx.tracer.trace_running():
        output_df = VertWOESubstitution().substitution(input_df, woe_rule)

    vt_wrapper = VerticalTableWrapper.from_dist_data(input_data, output_df.shape[0])

    # modify types of feature_selects to float
    for v in vt_wrapper.schema_map.values():
        for i, f in enumerate(list(v.features)):
            if f in feature_selects:
                v.feature_types[i] = 'float'

    return {
        "output_data": dump_vertical_table(
            ctx,
            output_df,
            output_data,
            vt_wrapper,
            input_data.sys_info,
        )
    }
