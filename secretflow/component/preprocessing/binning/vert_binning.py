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

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import (
    DistDataType,
    VerticalTableWrapper,
    dump_vertical_table,
    generate_random_string,
    load_table,
    model_dumps,
    model_loads,
    move_feature_to_label,
)
from secretflow.component.io.core.bins.bin_utils import pad_inf_to_split_points
from secretflow.device.device.pyu import PYU, PYUObject
from secretflow.device.driver import reveal
from secretflow.preprocessing.binning.vert_bin_substitution import VertBinSubstitution
from secretflow.preprocessing.binning.vert_binning import VertBinning
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Div, Report, Tab, Table

vert_binning_comp = Component(
    "vert_binning",
    domain="feature",
    version="0.0.2",
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
    lower_bound=2,
    lower_bound_inclusive=True,
)

vert_binning_comp.bool_attr(
    name="report_rules",
    desc="Whether report binning rules.",
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
            col_min_cnt_inclusive=1,
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

vert_binning_comp.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="report rules details if report_rules is true",
    types=[DistDataType.REPORT],
)

# current version 0.1
# only update if bin rule ser-de process changes
BINNING_RULE_MAX_MAJOR_VERSION = 0
BINNING_RULE_MAX_MINOR_VERSION = 1


def gen_one_col_table(rule_dict: dict) -> Table:
    headers = [
        Table.HeaderItem(
            name="from",
            desc="'from' can be range or nan values or single string.\
                bin rule map 'from' value to 'to' value",
            type="str",
        ),
        Table.HeaderItem(
            name="to",
            desc="'to' value is a float. bin rule map 'from' value to 'to' value",
            type="str",
        ),
        Table.HeaderItem(
            name="count",
            desc="Number of samples that fall into each bin",
            type="str",
        ),
    ]
    rows = []
    if rule_dict["type"] == "numeric":
        split_points = pad_inf_to_split_points(rule_dict["split_points"])
        for i in range(len(split_points) - 1):
            from_val = f"({split_points[i]}, {split_points[i+1]}]"
            to_val = str(rule_dict["filling_values"][i])
            count = str(rule_dict["total_counts"][i])
            rows.append(
                Table.Row(
                    name=f"{i}",
                    items=[
                        Attribute(s=from_val),
                        Attribute(s=to_val),
                        Attribute(s=count),
                    ],
                )
            )
        rows.append(
            Table.Row(
                name=f"{len(split_points)-1}",
                items=[
                    Attribute(s="nan values"),
                    Attribute(s=str(rule_dict["else_filling_value"])),
                    Attribute(s=str(rule_dict["else_counts"])),
                ],
            )
        )
    else:
        categories = rule_dict["categories"]
        for i in range(len(categories)):
            from_val = f"{categories[i]}"
            to_val = rule_dict["filling_values"][i]
            count = str(rule_dict["total_counts"][i])
            rows.append(
                Table.Row(
                    name=f"{i}",
                    items=[
                        Attribute(s=from_val),
                        Attribute(s=to_val),
                        Attribute(s=count),
                    ],
                )
            )
        rows.append(
            Table.Row(
                name=f"{len(categories)}",
                items=[
                    Attribute(s="nan values"),
                    Attribute(s=str(rule_dict.get("else_filling_value", "null"))),
                    Attribute(s=str(rule_dict.get("else_counts", "0"))),
                ],
            )
        )
    r_table = Table(
        headers=headers,
        rows=rows,
        name=f"{rule_dict['name']}",
        desc=f"rule for {rule_dict['name']} (type:{rule_dict['type']})",
    )
    return r_table


def gen_bin_rules_report(rules: Dict[PYU, PYUObject]) -> Report:
    tabs = []
    for rule in rules.values():
        rule = reveal(rule)
        variable_data_list = rule["variables"] if "variables" in rule else []
        for variable_data in variable_data_list:
            tabs.append(
                Tab(
                    name=variable_data["name"],
                    desc=f"bin rules for {variable_data['name']}",
                    divs=[
                        Div(
                            children=[
                                Div.Child(
                                    type="table",
                                    table=gen_one_col_table(variable_data),
                                )
                            ],
                        )
                    ],
                )
            )
    report_mate = Report(
        name="bin rule reports",
        desc="report for bin rules for each feature",
        tabs=tabs,
    )
    return report_mate


def dump_binning_rules(
    name, system_info, rules: Dict[PYU, PYUObject], report_rules: bool
) -> DistData:
    res = DistData(
        name=name,
        system_info=system_info,
        type=str(DistDataType.REPORT),
        data_refs=[],
    )
    if report_rules:
        report_mate = gen_bin_rules_report(rules)
        res.meta.Pack(report_mate)
    return res


@vert_binning_comp.eval_fn
def vert_binning_eval_fn(
    *,
    ctx,
    binning_method,
    bin_num,
    report_rules,
    input_data,
    input_data_feature_selects,
    bin_rule,
    report,
):
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
            ctx,
            "bin_rule",
            DistDataType.BIN_RUNNING_RULE,
            BINNING_RULE_MAX_MAJOR_VERSION,
            BINNING_RULE_MAX_MINOR_VERSION,
            [o for o in rules.values()],
            {
                "input_data_feature_selects": input_data_feature_selects,
                "model_hash": generate_random_string(next(iter(rules.keys()))),
            },
            bin_rule,
            input_data.system_info,
        )
        report_dist_data = dump_binning_rules(
            report,
            input_data.system_info,
            rules,
            report_rules,
        )

    return {"bin_rule": model_dist_data, "report": report_dist_data}


vert_bin_substitution_comp = Component(
    "vert_bin_substitution",
    domain="preprocessing",
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
        ctx,
        bin_rule,
        BINNING_RULE_MAX_MAJOR_VERSION,
        BINNING_RULE_MAX_MINOR_VERSION,
        DistDataType.BIN_RUNNING_RULE,
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
                v.feature_types[i] = 'float32'

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
