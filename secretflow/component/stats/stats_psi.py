# Copyright 2024 Ant Group Co., Ltd.
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

import logging
import math

import pandas as pd

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import (
    DistDataType,
    extract_data_infos,
    model_loads,
)
from secretflow.component.dataframe import CompDataFrame
from secretflow.component.io.core.bins.bin_utils import pad_inf_to_split_points
from secretflow.component.preprocessing.binning.vert_binning import (
    BINNING_RULE_MAX_MAJOR_VERSION,
    BINNING_RULE_MAX_MINOR_VERSION,
)
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.pyu import PYU, PYUObject
from secretflow.device.driver import reveal
from secretflow.preprocessing.binning.vert_binning import VertBinning
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Div, Report, Tab, Table

stats_psi_comp = Component(
    "stats_psi",
    domain="stats",
    version="0.0.1",
    desc="population stability index.",
)
stats_psi_comp.io(
    io_type=IoType.INPUT,
    name="input_base_data",
    desc="Input base vertical table.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    # col_params=None,
    col_params=[
        TableColParam(
            name="feature_selects",
            desc="which features should be binned.",
            col_min_cnt_inclusive=1,
        ),
    ],
)
stats_psi_comp.io(
    io_type=IoType.INPUT,
    name="input_test_data",
    desc="Input test vertical table.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)
stats_psi_comp.io(
    io_type=IoType.INPUT,
    name="bin_rule",
    desc="Input bin rule.",
    types=[DistDataType.BIN_RUNNING_RULE],
    col_params=None,
)
stats_psi_comp.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="Output population stability index.",
    types=[DistDataType.REPORT],
)

kEpsilon = 1e-3


@stats_psi_comp.eval_fn
def stats_psi_eval_fn(
    *,
    ctx,
    input_base_data,
    input_base_data_feature_selects,
    input_test_data,
    bin_rule,
    report,
):
    infos = extract_data_infos(
        input_base_data,
        load_features=True,
    )
    pyus = {p: PYU(p) for p in infos}
    bin_rules: dict[PYU, PYUObject] = load_bin_rules(ctx, bin_rule, pyus)
    feature_rules: dict[str, dict] = get_feature_rules(bin_rules)

    input_base_df = CompDataFrame.from_distdata(
        ctx,
        input_base_data,
        load_features=True,
        col_selects=input_base_data_feature_selects,
        load_ids=False,
        load_labels=False,
    ).to_pandas(
        check_null=False
    )  # FIXME: avoid to_pandas, use pa.Table
    input_test_df = CompDataFrame.from_distdata(
        ctx,
        input_test_data,
        load_features=True,
        col_selects=input_base_data_feature_selects,
        load_ids=False,
        load_labels=False,
    ).to_pandas(
        check_null=False
    )  # FIXME: avoid to_pandas, use pa.Table

    all_feature_psi = calculate_stats_psi(
        feature_rules, input_base_data_feature_selects, input_base_df, input_test_df
    )

    return {
        "report": dump_stats_psi(report, input_base_data.system_info, all_feature_psi)
    }


def load_bin_rules(ctx, bin_rule, pyus: dict[str, PYU]):
    model_objs, _ = model_loads(
        ctx,
        bin_rule,
        BINNING_RULE_MAX_MAJOR_VERSION,
        BINNING_RULE_MAX_MINOR_VERSION,
        DistDataType.BIN_RUNNING_RULE,
        pyus,
    )

    bin_rules = {}
    for obj in model_objs:
        assert isinstance(obj, PYUObject)
        bin_rules[obj.device] = obj

    return bin_rules


def get_feature_rules(bin_rules: dict[PYU, PYUObject]):
    feature_rules: dict[str, dict] = {}
    for _, rule in bin_rules.items():
        rule = reveal(rule)
        variable_data_list = rule["variables"] if "variables" in rule else []
        for rule_dict in variable_data_list:
            assert (
                rule_dict["name"] not in feature_rules.keys()
            ), f'{rule_dict["name"]} already exist'
            feature_rules[rule_dict["name"]] = rule_dict
    return feature_rules


def calculate_stats_psi(
    feature_rules: dict[str, dict],
    feature_selects: list[str],
    base_df: VDataFrame,
    test_df: VDataFrame,
):
    feature_psi_infos = []
    for feature in feature_selects:
        for device, table in base_df.partitions.items():
            assert device in test_df.partitions.keys(), f'{device} not in test df'
            if feature in table.columns:
                feature_psi_info: PsiInfos = reveal(
                    device(get_stats_psi_one_feature)(
                        feature_rules[feature],
                        feature,
                        table.data,
                        test_df.partitions[device].data,
                    )
                )
                feature_psi_infos.append(feature_psi_info)

    return feature_psi_infos


def get_stats_psi_one_feature(
    rule_dict: dict, feature: str, base_df: pd.DataFrame, test_df: pd.DataFrame
):
    base_bin_counts = get_bin_counts_one_feature(rule_dict, feature, base_df)
    test_bin_counts = get_bin_counts_one_feature(rule_dict, feature, test_df)
    feature_psi, psi_infos = calculate_stats_psi_one_feature(
        base_bin_counts, test_bin_counts
    )
    feature_psi_info = PsiInfos(feature, feature_psi, psi_infos)
    return feature_psi_info


def get_bin_counts_one_feature(rule_dict: dict, feature: str, df: pd.DataFrame):
    assert feature == rule_dict["name"], f"{feature} not in rule dict"
    assert feature in df.columns, f'{feature} not in {df.columns}'
    rows = []
    if rule_dict["type"] == "numeric":
        split_points = pad_inf_to_split_points(rule_dict["split_points"])
        counts = statistic_data_numeric(df, feature, split_points)

        for i in range(len(split_points) - 1):
            from_val = f"({split_points[i]}, {split_points[i+1]}]"
            to_val = str(rule_dict["filling_values"][i])
            rows.append([from_val, to_val, str(counts[i])])
        rows.append(
            [
                "nan values",
                str(rule_dict["else_filling_value"]),
                str(counts[-1]),
            ]
        )
    else:
        categories = rule_dict["categories"]
        counts_dict = statistic_data_category(df, feature)
        for i in range(len(categories)):
            from_val = f"{categories[i]}"
            to_val = rule_dict["filling_values"][i]
            count = str(counts_dict.get(categories[i], 0))
            rows.append([from_val, to_val, count])
        rows.append(
            [
                "nan values",
                str(rule_dict.get("else_filling_value", "null")),
                str(counts_dict['nan']),
            ]
        )
    return rows


def calculate_stats_psi_one_feature(base_bin_stat: list, test_bin_stat: list):
    assert len(base_bin_stat) == len(
        test_bin_stat
    ), f"base_bin_stat: {len(base_bin_stat)} and test_bin_stat: {len(test_bin_stat)} size not match."

    # calculate total sample count
    base_row_count = 0
    test_row_count = 0
    for i in range(len(base_bin_stat)):
        base_bin_label = base_bin_stat[i][0]
        test_bin_label = test_bin_stat[i][0]
        assert (
            base_bin_label == test_bin_label
        ), f'base_bin_label: {base_bin_label} and test_bin_label: {test_bin_label} not match.'
        base_row_count += int(base_bin_stat[i][2])
        test_row_count += int(test_bin_stat[i][2])
    assert (
        base_row_count > 0 and test_row_count > 0
    ), f'base_row_count: {base_row_count} and test_row_count: {test_row_count} should be greater than 0.'

    feature_psi = 0.0
    psi_infos = []
    for i in range(len(base_bin_stat)):
        # div 0 check
        base_ratio = int(base_bin_stat[i][2]) * 1.0 / base_row_count
        test_ratio = int(test_bin_stat[i][2]) * 1.0 / test_row_count
        p_diff = test_ratio - base_ratio
        ln = 0.0
        if int(base_bin_stat[i][2]) == 0 and int(test_bin_stat[i][2]) == 0:
            ln = 0.0
        else:
            ln = math.log(max(test_ratio, kEpsilon) / max(base_ratio, kEpsilon))

        psi = p_diff * ln
        feature_psi += psi
        psi_info = PsiBinInfo(base_bin_stat[i][0], psi, base_ratio, test_ratio)
        psi_infos.append(psi_info)

    return feature_psi, psi_infos


def statistic_data_numeric(df: pd.DataFrame, feature, bins):
    series = df[feature]
    categorized = pd.cut(series, bins=bins)
    counts = categorized.value_counts().sort_index().values.tolist()
    counts.append(series.isna().sum())
    return counts


def statistic_data_category(df: pd.DataFrame, feature):
    counts_dict = df[feature].value_counts()
    counts_dict['nan'] = df[feature].isna().sum()
    return counts_dict


class PsiBinInfo:
    def __init__(
        self, bin_label: str, psi: float, base_ratio: float, test_ratio: float
    ):
        self.bin_label = bin_label
        self.psi = psi
        self.base_ratio = base_ratio
        self.test_ratio = test_ratio


class PsiInfos:
    def __init__(
        self, feature_name: str, feature_psi: float, details: list[PsiBinInfo]
    ):
        self.feature = feature_name
        self.psi = feature_psi
        self.details = details


def gen_stats_psi_report(stats_psis: list[PsiInfos]) -> Report:
    tabs = []

    summary_headers = [
        Table.HeaderItem(
            name="feature",
            desc="selected feature",
            type="str",
        ),
        Table.HeaderItem(
            name="PSI",
            desc="population stable index",
            type="float",
        ),
    ]
    summary_rows = []
    for i in range(len(stats_psis)):
        summary_rows.append(
            Table.Row(
                name=f'{i}',
                items=[
                    Attribute(s=stats_psis[i].feature),
                    Attribute(f=stats_psis[i].psi),
                ],
            )
        )
    table = Table(headers=summary_headers, rows=summary_rows)
    tabs.append(
        Tab(
            name='稳定性评估总表',
            desc="stats psi summary of all selected features",
            divs=[
                Div(
                    children=[
                        Div.Child(
                            type="table",
                            table=table,
                        ),
                    ],
                )
            ],
        )
    )

    detail_headers = [
        Table.HeaderItem(
            name="Label",
            desc="bin label",
            type="str",
        ),
        Table.HeaderItem(
            name="PSI",
            desc="population stable index of every bin",
            type="float",
        ),
        Table.HeaderItem(
            name="Base Ratio",
            desc="base data ratio",
            type="float",
        ),
        Table.HeaderItem(
            name="Test Ratio",
            desc="test data ratio",
            type="float",
        ),
    ]

    for psi in stats_psis:
        detail_rows = []
        for i in range(len(psi.details)):
            detail_rows.append(
                Table.Row(
                    name=f'{i}',
                    items=[
                        Attribute(s=str(psi.details[i].bin_label)),
                        Attribute(f=psi.details[i].psi),
                        Attribute(f=psi.details[i].base_ratio),
                        Attribute(f=psi.details[i].test_ratio),
                    ],
                )
            )
        detail_table = Table(headers=detail_headers, rows=detail_rows)
        tabs.append(
            Tab(
                name=psi.feature,
                desc="stats psi details of one feature",
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="table",
                                table=detail_table,
                            ),
                        ],
                    )
                ],
            )
        )

    return Report(name="stats psi", desc="", tabs=tabs)


def dump_stats_psi(name, system_info, stats_psis: list[PsiInfos]) -> DistData:  # type: ignore
    report_mate = gen_stats_psi_report(stats_psis)
    res = DistData(
        name=name,
        system_info=system_info,
        type=str(DistDataType.REPORT),
        data_refs=[],
    )
    res.meta.Pack(report_mate)
    return res
