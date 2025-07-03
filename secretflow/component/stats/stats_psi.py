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

import math

import pandas as pd

from secretflow.component.core import (
    BINNING_RULE_MAX,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    Reporter,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.component.io.core.bins.bin_utils import pad_inf_to_split_points
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.pyu import PYU, PYUObject
from secretflow.device.driver import reveal
from secretflow.utils.errors import InvalidArgumentError

kEpsilon = 1e-3


def get_feature_rules(bin_rules: dict[PYU, PYUObject]):
    feature_rules: dict[str, dict] = {}
    for _, rule in bin_rules.items():
        rule = reveal(rule)
        variable_data_list = rule["variables"] if "variables" in rule else []
        for rule_dict in variable_data_list:
            if rule_dict["name"] in feature_rules.keys():
                raise InvalidArgumentError(
                    message="feature already exist",
                    detail={"feature": rule_dict["name"]},
                )
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
    if feature != rule_dict["name"] or feature not in df.columns:
        raise InvalidArgumentError(
            message=f"feature name mismatch or not in df.columns",
            detail={
                "feature": feature,
                "name": rule_dict["name"],
                "columns": df.columns,
            },
        )
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
    if len(base_bin_stat) != len(test_bin_stat):
        raise InvalidArgumentError(
            message="length of bin_stat mismatch",
            detail={
                "base_bin_stat": len(base_bin_stat),
                "test_bin_stat": len(test_bin_stat),
            },
        )

    # calculate total sample count
    base_row_count = 0
    test_row_count = 0
    for i in range(len(base_bin_stat)):
        base_bin_label = base_bin_stat[i][0]
        test_bin_label = test_bin_stat[i][0]
        if base_bin_label != test_bin_label:
            raise InvalidArgumentError(
                message="bin_label mismatch",
                detail={
                    "base_bin_label": base_bin_label,
                    "test_bin_label": test_bin_label,
                },
            )
        base_row_count += int(base_bin_stat[i][2])
        test_row_count += int(test_bin_stat[i][2])
    if not (base_row_count > 0 and test_row_count > 0):
        raise InvalidArgumentError(
            message="base_row_count and test_row_count should be greater than 0",
            detail={
                "base_row_count": base_row_count,
                "test_row_count": test_row_count,
            },
        )

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


@register(domain="stats", version="1.0.0", name="stats_psi")
class StatsPSI(Component):
    '''population stability index.'''

    feature_selects: list[str] = Field.table_column_attr(
        "input_base_ds",
        desc="which features should be binned.",
        limit=Interval.closed(1, None),
    )
    input_base_ds: Input = Field.input(
        desc="Input base vertical table.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    input_test_ds: Input = Field.input(
        desc="Input test vertical table.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    input_rule: Input = Field.input(
        desc="Input bin rule.",
        types=[DistDataType.BINNING_RULE],
    )
    report: Output = Field.output(
        desc="Output population stability index.",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        base_tbl = VTable.from_distdata(self.input_base_ds)
        pyus = {p: PYU(p) for p in base_tbl.parties.keys()}
        model = ctx.load_model(
            self.input_rule,
            DistDataType.BINNING_RULE,
            BINNING_RULE_MAX,
            pyus=pyus,
        )
        bin_rules = {}
        for obj in model.objs:
            assert isinstance(obj, PYUObject)
            bin_rules[obj.device] = obj

        feature_rules: dict[str, dict] = get_feature_rules(bin_rules)

        base_tbl = base_tbl.select(self.feature_selects)
        base_tbl.check_kinds(VTableFieldKind.FEATURE)
        test_tbl = VTable.from_distdata(
            self.input_test_ds, columns=self.feature_selects
        )
        test_tbl.check_kinds(VTableFieldKind.FEATURE)
        # FIXME: avoid to_pandas, use pa.Table
        input_base_df = ctx.load_table(base_tbl).to_pandas(check_null=False)
        input_test_df = ctx.load_table(test_tbl).to_pandas(check_null=False)

        all_feature_psi = calculate_stats_psi(
            feature_rules, self.feature_selects, input_base_df, input_test_df
        )

        self.dump_report(all_feature_psi)

    def dump_report(self, stats_psis: list[PsiInfos]):
        r = Reporter(system_info=self.input_base_ds.system_info)

        totals = []
        for info in stats_psis:
            totals.append([info.feature, info.psi])
        total_hd = {"feature": "selected feature", "PSI": "population stable index"}
        total_df = pd.DataFrame(data=totals, columns=list(total_hd.keys()))
        total_tbl = Reporter.build_table(total_df, columns=total_hd)
        r.add_tab(
            total_tbl,
            name='稳定性评估总表',
            desc="stats psi summary of all selected features",
        )

        detail_hd = {
            "Label": "bin label",
            "PSI": "population stable index of every bin",
            "Base Ratio": "base data ratio",
            "Test Ratio": "test data ratio",
        }
        for info in stats_psis:
            details = []
            for d in info.details:
                details.append([d.bin_label, d.psi, d.base_ratio, d.test_ratio])
            detail_df = pd.DataFrame(data=details, columns=list(detail_hd.keys()))
            detail_tbl = Reporter.build_table(detail_df, columns=detail_hd)
            r.add_tab(
                detail_tbl, name=info.feature, desc="stats psi details of one feature"
            )

        self.report.data = r.to_distdata()
