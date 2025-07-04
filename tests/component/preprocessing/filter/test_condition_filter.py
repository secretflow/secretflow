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
from secretflow.component.preprocessing.filter.condition_filter import ConditionFilter


@pytest.mark.mpc
def test_condition_filter(sf_production_setup_comp):
    work_dir = "test_condition_filter"
    alice_input_path = "test_condition_filter/alice.csv"
    bob_input_path = "test_condition_filter/bob.csv"
    hit_output_path = "test_condition_filter/hit.csv"
    else_output_path = "test_condition_filter/else.csv"

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

    tbl_info = VTable(
        name="input_data",
        parties=[
            VTableParty.from_dict(
                uri=alice_input_path,
                party="alice",
                format="csv",
                null_strs=[""],
                ids={"id1": "str"},
                features={"a1": "str", "a2": "str", "a3": "float32"},
                labels={"y": "float32"},
            ),
            VTableParty.from_dict(
                uri=bob_input_path,
                party="bob",
                format="csv",
                null_strs=[""],
                ids={"id2": "str"},
                features={"b4": "float32", "b5": "str", "b6": "float32"},
            ),
        ],
    )

    test_cases = [
        {
            "attrs": {
                'input/input_ds/feature': ["b4"],
                'comparator': "<",
                'bound_value': "11",
                'float_epsilon': 0.01,
            },
            "expected": [["1", "4"], ["2", "3"]],
            "desc": "test filter",
        },
        {
            "attrs": {'input/input_ds/feature': ["a1"], 'comparator': "NOTNULL"},
            "expected": [["1", "2", "4"], ["3"]],
            "desc": "test null",
        },
    ]

    for tc in test_cases:
        param = build_node_eval_param(
            domain="data_filter",
            name="condition_filter",
            version="1.0.0",
            attrs=tc["attrs"],
            inputs=[tbl_info],
            output_uris=[
                hit_output_path,
                else_output_path,
            ],
        )

        res = comp_eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        assert len(res.outputs) == 2

        if self_party in input_datasets:
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


def test_condition_filter_error():
    # test kind mismatch
    input_tbl = VTable(
        name="input_data",
        parties=[
            VTableParty.from_dict(
                uri="alice_input_path.csv",
                party="alice",
                format="csv",
                null_strs=[""],
                ids={"id1": "str"},
                features={"a1": "str", "a2": "str", "a3": "float32"},
                labels={"y": "float32"},
            ),
            VTableParty.from_dict(
                uri="bob_input_path.csv",
                party="bob",
                format="csv",
                null_strs=[""],
                ids={"id2": "str"},
                features={"b4": "float32", "b5": "str", "b6": "float32"},
            ),
        ],
    )

    comp = ConditionFilter(
        feature="y",
        comparator="<",
        bound_value="11",
        float_epsilon=0.01,
        input_ds=input_tbl.to_distdata(),
    )
    with pytest.raises(ValueError, match=r"kind of .* mismatch, expected .*"):
        comp.evaluate(None)
