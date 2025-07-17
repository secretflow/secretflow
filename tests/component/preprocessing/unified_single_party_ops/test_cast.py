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

import logging

import pandas as pd
import pytest
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import VerticalTable

import secretflow.compute as sc
from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.preprocessing.unified_single_party_ops.cast import Cast


@pytest.mark.mpc
def test_cast(sf_production_setup_comp):
    alice_input_path = "test_cast/alice_input.csv"
    bob_input_path = "test_cast/bob_input.csv"
    output_path = "test_cast/output.csv"
    output_rule = "test_cast/output.rule"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    alice_data = {"A": ["s1", "s2"], "B": [1, 2], "C": ["0.1", "0.2"]}
    bob_data = {"D": [0.1, 1.2], "E": [True, False]}

    expected_alice = {"A": ["s1", "s2"], "B": [1.0, 2.0], "C": [0.1, 0.2]}
    expected_bob = {"D": [0.1, 1.2], "E": [1.0, 0.0]}

    if self_party == "alice":
        pd.DataFrame(alice_data).to_csv(
            storage.get_writer(alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        pd.DataFrame(bob_data).to_csv(
            storage.get_writer(bob_input_path),
            index=False,
        )

    param = build_node_eval_param(
        domain="preprocessing",
        name="cast",
        version="1.0.0",
        attrs={
            "astype": "float",
            "input/input_ds/columns": ["B", "C", "D", "E"],
        },
        inputs=[
            VTable(
                name="input_ds",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                        features={"A": "str", "B": "int", "C": "str"},
                    ),
                    VTableParty.from_dict(
                        uri=bob_input_path,
                        party="bob",
                        format="csv",
                        features={"D": "float"},
                        labels={"E": "bool"},
                    ),
                ],
            )
        ],
        output_uris=[output_path, output_rule],
    )
    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    if self_party in ["alice", "bob"]:
        out_meta = VerticalTable()
        res.outputs[0].meta.Unpack(out_meta)
        logging.warning(f"...meta.... \n{out_meta}\n.......")
        assert out_meta.schemas[0].feature_types == ["str", "float", "float"]
        assert out_meta.schemas[1].feature_types == ["float"]
        assert out_meta.schemas[1].label_types == ["float"]

        real_data = orc.read_table(storage.get_reader(output_path)).to_pandas()
        logging.warning(f"...vertical_res:{self_party}... \n{real_data}\n.....")
        expected_data = expected_alice if self_party == "alice" else expected_bob
        pd.testing.assert_frame_equal(
            pd.DataFrame(expected_data),
            real_data,
            check_dtype=False,
        )
    # test substitution
    sub_path = "test_cast/substitution.csv"
    param2 = build_node_eval_param(
        domain="preprocessing",
        name="substitution",
        version="1.0.0",
        attrs=None,
        inputs=[param.inputs[0], res.outputs[1]],
        output_uris=[sub_path],
    )
    res = comp_eval(
        param=param2,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    if self_party in ["alice", "bob"]:
        sub_out = orc.read_table(storage.get_reader(sub_path)).to_pandas()
        original_out = orc.read_table(storage.get_reader(output_path)).to_pandas()

        logging.info(f"....... \nsub_out\n{sub_out}\n.,......")

        assert sub_out.equals(original_out)


@pytest.mark.mpc
def test_cast_error(sf_production_setup_comp):
    work_dir = "test_cast_error"
    alice_input_path = f"{work_dir}/alice_input.csv"
    bob_input_path = f"{work_dir}/bob_input.csv"
    output_path = f"{work_dir}/output.csv"
    output_rule = f"{work_dir}/output.rule"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    alice_data = {"A": ["s1", "s2"], "B": [1, 2], "C": ["0.1", "0.2"]}
    bob_data = {"D": [0.1, 1.2], "E": [True, False]}

    if self_party == "alice":
        pd.DataFrame(alice_data).to_csv(
            storage.get_writer(alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        pd.DataFrame(bob_data).to_csv(
            storage.get_writer(bob_input_path),
            index=False,
        )

    param = build_node_eval_param(
        domain="preprocessing",
        name="cast",
        version="1.0.0",
        attrs={
            "astype": "float",
            "input/input_ds/columns": ["A"],
        },
        inputs=[
            VTable(
                name="input_ds",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                        features={"A": "str", "B": "int", "C": "str"},
                    ),
                    VTableParty.from_dict(
                        uri=bob_input_path,
                        party="bob",
                        format="csv",
                        features={"D": "float"},
                        labels={"E": "bool"},
                    ),
                ],
            )
        ],
        output_uris=[output_path, output_rule],
    )
    with pytest.raises(Exception):
        comp_eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )


def test_cast_apply_table():
    good_cases = [
        {"data": ["\' 1\' ", "\"2\"", " 3 "], "except": [1, 2, 3], "target": "int"},
        {
            "data": ["\' 1\' ", "\"2\"", " 1.7976931348623157e+309 "],
            "except": [1.0, 2.0, float('inf')],
            "target": "float",
        },
        {
            "data": [" 1 ", "2", " 3.0 "],
            "except": [" 1 ", "2", " 3.0 "],
            "target": "str",
        },
        {"data": [1, 2, 3], "except": [1, 2, 3], "target": "int"},
        {"data": [1, 2, 3], "except": [1.0, 2.0, 3.0], "target": "float"},
        {"data": [1, 2, 3], "except": ["1", "2", "3"], "target": "str"},
        {"data": [1.0, 2.0, 3.0], "except": [1, 2, 3], "target": "int"},
        {"data": [1.0, 2.0, 3.0], "except": [1.0, 2.0, 3.0], "target": "float"},
        {"data": [1.0, 2.0, 3.0], "except": ["1", "2", "3"], "target": "str"},
        {"data": [True, False, True], "except": [1, 0, 1], "target": "int"},
        {"data": [True, False, True], "except": [1.0, 0.0, 1.0], "target": "float"},
        {
            "data": [True, False, True],
            "except": ["true", "false", "true"],
            "target": "str",
        },
    ]

    for index, item in enumerate(good_cases):
        df = pd.DataFrame({"A": item["data"]})
        table = sc.Table.from_pandas(df)
        res_tbl = Cast.apply(table, item["target"])
        res = res_tbl.to_pandas()["A"].tolist()
        excepted = item["except"]
        assert res == excepted, f"not equal: {index}, {res}, {excepted}"

    bad_cases = [
        {"data": ["1.0"], "target": "int"},
        {"data": [""], "target": "int"},
        {"data": [""], "target": "float"},
        {"data": ["a"], "target": "float"},
        {"data": ["a"], "target": "int"},
    ]
    for index, item in enumerate(bad_cases):
        with pytest.raises(ValueError) as exc_info:
            df = pd.DataFrame({"A": item["data"]})
            table = sc.Table.from_pandas(df)
            Cast.apply(table, item["target"])
        logging.info(f"Caught expected Exception: {index},{exc_info}")

    test_overflow = [
        {"data": [3.40e38], "target": "int"},
        {"data": ["300000000000000000000"], "target": "int"},
    ]
    for index, item in enumerate(test_overflow):
        with pytest.raises(Exception) as exc_info:
            df = pd.DataFrame({"A": item["data"]})
            table = sc.Table.from_pandas(df)
            Cast.apply(table, item["target"])
        logging.info(f"Caught expected Exception: {df},{exc_info}")
