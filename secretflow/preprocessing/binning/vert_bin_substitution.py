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

from typing import Dict, Union

import numpy as np
import pyarrow as pa

import secretflow.compute as sc
from secretflow.component.core import CompVDataFrame
from secretflow.compute import Table
from secretflow.device import PYU, PYUObject, wait


def apply_binning_rules(
    rules: Dict, input: Union[Dict[str, np.dtype], pa.Table, pa.Schema]
) -> sc.Table:
    rules = {v['name']: v for v in rules["variables"]}
    if isinstance(input, pa.Table):
        table = Table.from_pyarrow(input)
    else:
        table = Table.from_schema(input)

    input_schema = set(table.column_names)

    for v in rules:
        rule = rules[v]
        conds = []
        if rule["type"] == "string":
            assert v in input_schema
            col = table.column(v)
            conds = [sc.equal(col, c) for c in rule["categories"]]
        else:
            split_points = rule["split_points"]
            if len(split_points) == 0:
                conds = []
            else:
                assert v in input_schema
                col = table.column(v)
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
    def _sub(data: pa.Table, r: Dict) -> pa.Table:
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
            list(rules.keys()), data.column_names
        ).all(), "rule feature names [%s] mismatch with input dataset [%s]" % (
            str(rules.keys()),
            str(data.column_names),
        )

        return apply_binning_rules(r, data).to_table()

    def substitution(
        self, vdata: CompVDataFrame, rules: Dict[PYU, PYUObject]
    ) -> CompVDataFrame:
        """
        substitute dataset's value by binning substitution rules.

        Args:
            vdata: vertical slice dataset to be substituted.
            rules: binning substitution rules build by VertBinning.

        Returns:
            new_vdata: vertical slice dataset after substituted.
        """
        pyu_new_data = []
        for device in rules:
            # all rules must corresponds to some party
            new_data = device(VertBinSubstitution._sub)(
                vdata.data(device), rules[device]
            )
            pyu_new_data.append(new_data)

        wait(pyu_new_data)

        new_df = CompVDataFrame({}, vdata.system_info)
        for d in pyu_new_data:
            new_df.set_data(d)

        return new_df
