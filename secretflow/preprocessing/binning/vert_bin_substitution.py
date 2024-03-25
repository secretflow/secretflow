# Copyright 2022 Ant Group Co., Ltd.
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

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import secretflow.compute as sc
from secretflow.compute import Table
from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, PYUObject, reveal, wait


def binning_rules_to_sc(rules: Dict, input_schema: Dict[str, np.dtype]) -> sc.Table:
    rules = {v['name']: v for v in rules["variables"]}
    assert set(rules).issubset(set(input_schema))

    table = Table.from_schema(input_schema)

    for v in rules:
        col = table.column(v)
        rule = rules[v]
        conds = []
        if rule["type"] == "string":
            conds = [sc.equal(col, c) for c in rule["categories"]]
        else:
            split_points = rule["split_points"]
            if len(split_points) == 0:
                conds = []
            else:
                conds = [sc.less_equal(col, c) for c in split_points]
                conds.append(sc.greater(col, split_points[-1]))

        if conds:
            cases = rule["filling_values"] + [rule["else_filling_value"]]
            cases = list(map(np.float32, cases))
            new_col = sc.case_when(sc.make_struct(*conds), *cases)
            table = table.set_column(table.column_names.index(v), v, new_col)

    return table


class VertBinSubstitution:
    @staticmethod
    def _sub(data: pd.DataFrame, r: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """
        PYU functions for binning substitution.

        Args:
            data: input dataset for this party.
            r: bin substitution rules.

        Returns:
            data: dataset after substituted.
        """
        assert isinstance(r, dict) and "variables" in r, f"not support rule format {r}"

        rules = {v['name']: v for v in r["variables"]}
        assert np.isin(
            list(rules.keys()), data.columns
        ).all(), "rule feature names [%s] mismatch with input dataset [%s]" % (
            str(rules.keys()),
            str(data.columns),
        )

        rules_table = binning_rules_to_sc(r, dict(data.dtypes))
        changed_columns = rules_table.column_changes()[1]
        data = rules_table.dump_runner().run(data)

        return data, changed_columns

    def substitution(
        self, vdata: VDataFrame, rules: Dict[PYU, PYUObject]
    ) -> VDataFrame:
        """
        substitute dataset's value by binning substitution rules.

        Args:
            vdata: vertical slice dataset to be substituted.
            rules: binning substitution rules build by VertBinning.

        Returns:
            new_vdata: vertical slice dataset after substituted.
        """
        pyu_new_data = {}
        pyu_changed_columns = {}
        for device in rules:
            # all rules must corresponds to some party
            assert (
                device in vdata.partitions.keys()
            ), f"device {device} not exist in vdata"
            new_data, changed_columns = device(VertBinSubstitution._sub)(
                vdata.partitions[device].data, rules[device]
            )
            pyu_new_data[device] = new_data
            pyu_changed_columns[device] = changed_columns

        wait(pyu_new_data)
        pyu_changed_columns = reveal(pyu_changed_columns)
        changed_columns = {c for pc in pyu_changed_columns.values() for c in pc}

        # but some party may have no substitution rule
        def sub_if_exists(d):
            return (
                partition(
                    data=pyu_new_data[d],
                    backend=vdata.partitions[d].backend,
                )
                if d in pyu_new_data
                else vdata.partitions[d]
            )

        new_vdata = VDataFrame({d: sub_if_exists(d) for d in vdata.partitions.keys()})

        return new_vdata, changed_columns
