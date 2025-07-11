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

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
from secretflow_serving_lib import compute_trace_pb2
from secretflow_spec.v1.data_pb2 import SystemInfo

import secretflow.compute as sc
from secretflow.component.core import (
    BINNING_RULE_MAX,
    Component,
    CompVDataFrame,
    Context,
    DistDataType,
    Input,
    Model,
    Output,
    Registry,
    Reporter,
    Version,
    VTable,
    VTableUtils,
    pad_inf_to_split_points,
)
from secretflow.device import PYU, PYUObject, reveal
from secretflow.preprocessing.binning.vert_bin_substitution import apply_binning_rules

from ..preprocessing import ArrowRunner, IRunner, PreprocessingMixin, build_schema


@dataclass
class BinningRunner(IRunner):
    rule: dict

    def __getitem__(self, key):
        """
        Only for compatibility with io/read_data,write_data
        """
        return self.rule[key]

    def __setitem__(self, key, value):
        """
        Only for compatibility with io/read_data,write_data
        """
        self.rule[key] = value

    def __delitem__(self, key):
        """
        Only for compatibility with io/read_data,write_data
        """
        del self.rule[key]

    def keys(self):
        """
        Only for compatibility with io/read_data,write_data
        """
        return self.rule.keys()

    def values(self):
        """
        Only for compatibility with io/read_data,write_data
        """
        return self.rule.values()

    def items(self):
        """
        Only for compatibility with io/read_data,write_data
        """
        return self.rule.items()

    def __contains__(self, key):
        """
        Only for compatibility with io/read_data,write_data
        """
        return key in self.rule

    def __iter__(self):
        """
        Only for compatibility with io/read_data,write_data
        """
        return iter(self.rule)

    def __len__(self):
        """
        Only for compatibility with io/read_data,write_data
        """
        return len(self.rule)

    def get_input_features(self) -> list[str]:
        features = [v["name"] for v in self.rule["variables"]]
        return features

    def run(self, df: pa.Table) -> pa.Table:
        res = apply_binning_rules(self.rule, df)

        return build_schema(res.to_table(), df, None)

    def column_changes(self, input_schema: pa.Schema) -> tuple[list, list, list]:
        table = apply_binning_rules(self.rule, input_schema)
        return table.column_changes()

    def dump_serving_pb(
        self, name: str, input_schema: pa.Schema
    ) -> tuple[compute_trace_pb2.ComputeTrace, pa.Schema, pa.Schema]:
        res = apply_binning_rules(self.rule, input_schema)
        return res.dump_serving_pb(name)


class VertBinningBase(PreprocessingMixin, Component):
    def model_info(self) -> tuple[DistDataType, Version]:
        return DistDataType.BINNING_RULE, BINNING_RULE_MAX

    def fit(
        self,
        ctx: Context,
        out_rule: Output,
        input: Input | VTable,
        fn: (
            Callable[[sc.Table], sc.Table | IRunner]
            | Callable[[sc.Table, object], sc.Table | IRunner]
        ),
        extras: dict[str, PYUObject | Any] = None,
        label_party: str | None = None,
    ) -> Model:
        """
        build a rule by a funtion and input schema

        NOTE: the function SHOULD not rely on the input data,table statistical info can be passed through 'extras'.
        """
        signature = inspect.signature(fn)
        is_one_param = len(signature.parameters) == 1

        def _fit(trans_data: sc.Table, extra: object) -> IRunner:
            if is_one_param:
                out_data = fn(trans_data)
            else:
                out_data = fn(trans_data, extra)

            assert isinstance(out_data, (sc.Table, IRunner))

            if isinstance(out_data, sc.Table):
                runner = ArrowRunner.from_table(out_data, in_tbl.column_names)
            else:
                runner = out_data

            return runner

        tbl = input if isinstance(input, VTable) else VTable.from_distdata(input)
        runner_objs = []
        for p in tbl.parties.values():
            pa_schema = VTableUtils.to_arrow_schema(p.schema)
            in_tbl = sc.Table.from_schema(pa_schema)
            extra = extras.get(p.party) if extras else None
            out_runner = PYU(p.party)(_fit)(in_tbl, extra)
            runner_objs.append(out_runner)
        if label_party is not None:
            extra = extras.get(label_party) if extras else None
            out_runner = PYU(label_party)(_fit)(in_tbl, extra)
            runner_objs.append(out_runner)
        defi = Registry.get_definition_by_class(self)
        model_type, model_version = self.model_info()
        rule = self.build_model(
            defi.component_id,
            model_type,
            model_version,
            runner_objs,
            None,
            tbl.system_info,
        )
        ctx.dump_to(rule, out_rule)
        return rule

    def do_evaluate(
        self,
        ctx: Context,
        out_ds: Output,
        out_rule: Output,
        input_df: CompVDataFrame,
        trans_tbl: VTable,
        rule_extras: dict[PYU, PYUObject],
        label_party: str | None = None,
    ):
        def _fit(_: sc.Table, rule: dict) -> BinningRunner:
            return BinningRunner(rule)

        extras = {pyu.party: obj for pyu, obj in rule_extras.items()}
        rule_model = self.fit(ctx, out_rule, trans_tbl, _fit, extras, label_party)
        self.transform(ctx, out_ds, input_df, rule_model, streaming=False)

    def dump_report(
        self,
        out_report: Output,
        rules: dict[PYU, PYUObject],
        report_rule: bool,
        system_info: SystemInfo,
    ):
        r = Reporter(
            name="bin rule reports",
            desc="report for bin rules for each feature",
            system_info=system_info,
        )
        if report_rule:
            for rule in rules.values():
                rule = reveal(rule)
                variable_data_list = rule["variables"] if "variables" in rule else []
                for variable_data in variable_data_list:
                    name = variable_data["name"]
                    desc = f"bin rules for {variable_data['name']}"

                    r.add_tab(
                        self.build_report_table(variable_data), name=name, desc=desc
                    )

        out_report.data = r.to_distdata()

    def build_report_table(self, rule_dict: dict):
        rows = []

        if rule_dict["type"] == "numeric":
            split_points = pad_inf_to_split_points(rule_dict["split_points"])
            for i in range(len(split_points) - 1):
                from_val = f"({split_points[i]}, {split_points[i + 1]}]"
                to_val = str(rule_dict["filling_values"][i])
                count = str(rule_dict["total_counts"][i])
                optional_values = []
                if "postive_rates" in rule_dict:
                    optional_values.append(
                        str(rule_dict["postive_rates"][i]).replace("nan", "-")
                    )
                if "total_rates" in rule_dict:
                    optional_values.append(
                        str(rule_dict["total_rates"][i]).replace("nan", "-")
                    )
                rows.append([from_val, to_val, count] + optional_values)

            optional_last = []
            if "postive_rates" in rule_dict:
                optional_last.append(
                    str(rule_dict["else_positive_rate"]).replace("nan", "-")
                )
            if "total_rates" in rule_dict:
                optional_last.append(
                    str(rule_dict["else_total_rate"]).replace("nan", "-")
                )
            last = [
                "nan values",
                str(rule_dict["else_filling_value"]),
                str(rule_dict["else_counts"]),
            ] + optional_last
            rows.append(last)
        else:
            categories = rule_dict["categories"]
            for i in range(len(categories)):
                from_val = f"{categories[i]}"
                to_val = str(rule_dict["filling_values"][i])
                count = str(rule_dict["total_counts"][i])
                optional_values = []
                if "postive_rates" in rule_dict:
                    optional_values.append(
                        str(rule_dict["postive_rates"][i]).replace("nan", "-")
                    )
                if "total_rates" in rule_dict:
                    optional_values.append(
                        str(rule_dict["total_rates"][i]).replace("nan", "-")
                    )
                rows.append([from_val, to_val, count] + optional_values)

            optional_last = []
            if "postive_rates" in rule_dict:
                optional_last.append(
                    str(rule_dict.get("else_positive_rate", "-")).replace("nan", "-")
                )
            if "total_rates" in rule_dict:
                optional_last.append(
                    str(rule_dict.get("else_positive_rate", "-")).replace("nan", "-")
                )

            last = [
                "nan values",
                str(rule_dict.get("else_filling_value", "null")),
                str(rule_dict.get("else_counts", "0")),
            ] + optional_last
            rows.append(last)

        headers = {
            "from": "'from' can be range or nan values or single string.\
                    bin rule map 'from' value to 'to' value",
            "to": "'to' value is a float. bin rule map 'from' value to 'to' value",
            "count": "Number of samples that fall into each bin",
        }
        if "postive_rates" in rule_dict:
            headers["positive_rate"] = "positive bin sample count/ bin sample count"
        if "total_rates" in rule_dict:
            headers["total_rate"] = "bin sample count/ total sample count"

        df = pd.DataFrame(data=rows, columns=list(headers.keys()))
        r_table = Reporter.build_table(
            df,
            name=f"{rule_dict['name']}",
            desc=f"rule for {rule_dict['name']} (type:{rule_dict['type']})",
            columns=headers,
        )
        return r_table
