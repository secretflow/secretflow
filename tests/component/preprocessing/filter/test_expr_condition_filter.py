# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pytest
from pyarrow import orc

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.preprocessing.filter.expr_condition_filter import (
    ExprConditionFilter,
)
from secretflow.utils.errors import InvalidArgumentError


@pytest.mark.mpc
def test_expr_condition_filter(sf_production_setup_comp):
    work_dir = "test_expr_condition_filter"
    alice_input_path = "test_expr_condition_filter/alice.csv"
    bob_input_path = "test_expr_condition_filter/bob.csv"
    output_hit_path = "test_expr_condition_filter/hit.csv"
    output_else_path = "test_expr_condition_filter/else.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    input_datasets = {
        "alice": pd.DataFrame(
            {
                "id1": ["1", "2", "3", "4"],
                "a1": ["K5", "K1", None, "K6"],
                "a2": ["A5", "A1", "A2", "A6"],
                "a3": [5, 1, 2, 6],
                "y": [0, 1, 1, 0],
            }
        ),
        "bob": pd.DataFrame(
            {
                "id2": ["1", "2", "3", "4"],
                "b4": [10.2, 20.5, None, -0.4],
                "b5": ["B3", None, "B9", "B4"],
                "b6": [3, 1, 9, 4],
            }
        ),
    }

    if self_party in input_datasets:
        path = f"{work_dir}/{self_party}.csv"
        df = input_datasets[self_party]
        df.to_csv(storage.get_writer(path), index=False)

    alice_meta = VTableParty.from_dict(
        party="alice",
        uri=alice_input_path,
        format="csv",
        null_strs=[""],
        ids={"id1": "str"},
        features={"a1": "str", "a2": "str", "a3": "float32"},
        labels={"y": "float32"},
    )
    bob_meta = VTableParty.from_dict(
        party="bob",
        uri=bob_input_path,
        format="csv",
        null_strs=[""],
        ids={"id2": "str"},
        features={"b4": "float32", "b5": "str", "b6": "float32"},
    )

    test_cases = [
        {
            "expr": "b4 < 11",
            "parties": [alice_meta, bob_meta],
            "expected": [["1", "4"], ["2", "3"]],
        },
        {
            "expr": "b4 < 11 and b5 != 'B4'",
            "parties": [bob_meta],
            "expected": [["1"], ["2", "3", "4"]],
        },
    ]
    for tc in test_cases:
        parties = tc["parties"]
        param = build_node_eval_param(
            domain="data_filter",
            name="expr_condition_filter",
            version="1.0.0",
            attrs={"expr": tc["expr"]},
            inputs=[VTable(name="input_ds", parties=parties)],
            output_uris=[output_hit_path, output_else_path],
        )

        res = comp_eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        assert len(res.outputs) == 2

        if self_party in [p.party for p in parties]:
            id_name = "id1" if self_party == "alice" else "id2"
            hit_ds_info = VTable.from_distdata(res.outputs[0], columns=[id_name])
            else_ds_info = VTable.from_distdata(res.outputs[1], columns=[id_name])

            hit_ds = orc.read_table(
                storage.get_reader(hit_ds_info.get_party(self_party).uri)
            ).to_pandas()
            else_ds = orc.read_table(
                storage.get_reader(else_ds_info.get_party(self_party).uri)
            ).to_pandas()
            expected = tc["expected"]
            assert list(hit_ds[id_name]) == expected[0]
            assert list(else_ds[id_name]) == expected[1]


def test_expr_condition_filter_error():
    # test expr empty
    comp = ExprConditionFilter(expr="")
    with pytest.raises(InvalidArgumentError, match="empty expr"):
        comp.evaluate(None)

    # test no columns
    comp = ExprConditionFilter(expr="1 == 1")
    with pytest.raises(
        InvalidArgumentError, match="cannot parse columns from the expr"
    ):
        comp.evaluate(None)

    # test columns must belong to one party
    input_tbl = VTable(
        name="input_ds",
        parties=[
            VTableParty.from_dict(
                party="alice",
                uri="alice_input_path.csv",
                format="csv",
                null_strs=[""],
                ids={"id1": "str"},
                features={"a1": "str", "a2": "str", "a3": "float32"},
                labels={"y": "float32"},
            ),
            VTableParty.from_dict(
                party="bob",
                uri="bob_input_path.csv",
                format="csv",
                null_strs=[""],
                ids={"id2": "str"},
                features={"b4": "float32", "b5": "str", "b6": "float32"},
            ),
        ],
    )
    comp = ExprConditionFilter(
        expr="b4 < 11 AND a1 IS NOT NULL", input_ds=input_tbl.to_distdata()
    )
    with pytest.raises(
        InvalidArgumentError, match="The columns of expr must appears in one party"
    ):
        comp.evaluate(None)


def test_parse_columns():
    columns = ExprConditionFilter.parse_columns(
        "(age > 20 AND age < 30) OR (type = 2) OR (name = 1)"
    )
    assert set(columns) == set(['age', 'type', 'name'])
    sql = "(field_A > -3.1415926 and field_A <3.1415926 and field_B =1) or (field_C >= 100 and field_C <= 1000 and field_B != 1) or (field_D ='match_1' and field_E != 'MATCH_2' and field_B =1)"
    columns = ExprConditionFilter.parse_columns(sql)
    assert set(columns) == set(['field_A', 'field_B', 'field_C', 'field_D', 'field_E'])
    columns = ExprConditionFilter.parse_columns(
        "ABS(level - 10) < 1 AND name LIKE 'XX%'"
    )
    assert set(columns) == set(['level', 'name'])
