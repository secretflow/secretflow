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
import time

import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import IndividualTable, VerticalTable
from secretflow_spec.v1.report_pb2 import Report

from secretflow.component.core import (
    VTable,
    VTableParty,
    assert_almost_equal,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


@pytest.mark.mpc
def test_psi_orc(sf_production_setup_comp):
    receiver_input_path = "test_psi/input_ds1.orc"
    sender_input_path = "test_psi/input_ds2.csv"
    output_path = "test_psi/psi_output.csv"
    output_report_path = "test_psi/psi_output"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        da.to_orc(storage.get_writer(receiver_input_path), index=False)

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K500", "K200", "K300", "K400", "K600", "K700"],
                "feature2": ["DD", "AA", "BB", "CC", "EE", "FF"],
            }
        )

        db.to_csv(storage.get_writer(sender_input_path), index=False)

    expected_result_a = pd.DataFrame(
        {
            "item": ["B", "D", "E", "G"],
            "feature1": ["BBB", "DDD", "EEE", "GGG"],
            "id1": ["K200", "K300", "K400", "K500"],
        }
    )
    expected_result_b = pd.DataFrame(
        {
            "feature2": ["AA", "BB", "CC", None],
            "id2": ["K200", "K300", "K400", "K500"],
        }
    )

    param = build_node_eval_param(
        domain="data_prep",
        name="psi",
        version="1.0.0",
        attrs={
            "protocol": "PROTOCOL_ECDH",
            "sort_result": True,
            "input/input_ds1/keys": ["id1"],
            "protocol/PROTOCOL_ECDH": "CURVE_FOURQ",
            "input/input_ds2/keys": ["id2"],
        },
        inputs=[
            VTable(
                name="input_ds1",
                parties=[
                    VTableParty.from_dict(
                        uri=receiver_input_path,
                        party="alice",
                        format="orc",
                        features={"item": "str", "feature1": "str", "id1": "str"},
                    )
                ],
            ),
            VTable(
                name="input_ds2",
                parties=[
                    VTableParty.from_dict(
                        uri=sender_input_path,
                        party="bob",
                        format="csv",
                        null_strs=["DD"],
                        features={"feature2": "str", "id2": "str"},
                    )
                ],
            ),
        ],
        output_uris=[output_path, output_report_path],
    )

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    if "alice" == sf_cluster_config.private_config.self_party:
        assert_almost_equal(
            expected_result_a,
            orc.read_table(storage.get_reader(output_path)),
            ignore_order=True,
            check_dtype=False,
        )
        # storage.remove(output_path)
    if "bob" == sf_cluster_config.private_config.self_party:
        csv_b = orc.read_table(storage.get_reader(output_path))

        assert_almost_equal(
            expected_result_b,
            csv_b,
            ignore_order=True,
            check_like=True,
            check_dtype=False,
        )
        # storage.remove(output_path)

    output_vt = VerticalTable()

    assert res.outputs[0].meta.Unpack(output_vt)
    assert len(output_vt.schemas) == 2

    assert output_vt.line_count == 4
    assert output_vt.schemas[0].ids == ["id1"]
    assert output_vt.schemas[0].features == ["item", "feature1"]
    assert output_vt.schemas[1].ids == ["id2"]
    assert output_vt.schemas[1].features == ["feature2"]


@pytest.mark.mpc
def test_psi(sf_production_setup_comp):
    receiver_input_path = "test_psi/input_ds1.csv"
    sender_input_path = "test_psi/input_ds2.csv"
    output_path = "test_psi/psi_output.csv"
    output_report_path = "test_psi/psi_output"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        da.to_csv(storage.get_writer(receiver_input_path), index=False)

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K500", "K200", "K300", "K400", "K600", "K700"],
                "feature2": ["DD", "AA", "BB", "CC", "EE", "FF"],
            }
        )

        db.to_csv(storage.get_writer(sender_input_path), index=False)

    expected_result_a = pd.DataFrame(
        {
            "item": ["B", "D", "E", "G"],
            "feature1": ["BBB", "DDD", "EEE", None],
            "id1": ["K200", "K300", "K400", "K500"],
        }
    )
    expected_result_b = pd.DataFrame(
        {
            "feature2": ["AA", "BB", "CC", None],
            "id2": ["K200", "K300", "K400", "K500"],
        }
    )

    param = build_node_eval_param(
        domain="data_prep",
        name="psi",
        version="1.0.0",
        attrs={
            "protocol": "PROTOCOL_ECDH",
            "sort_result": True,
            "input/input_ds1/keys": ["id1"],
            "input/input_ds2/keys": ["id2"],
        },
        inputs=[
            VTable(
                name="input_ds1",
                parties=[
                    VTableParty.from_dict(
                        uri=receiver_input_path,
                        party="alice",
                        format="csv",
                        null_strs=["GGG"],
                        ids={"id1": "str"},
                        features={"item": "str", "feature1": "str"},
                    )
                ],
            ),
            VTable(
                name="input_ds2",
                parties=[
                    VTableParty.from_dict(
                        uri=sender_input_path,
                        party="bob",
                        format="csv",
                        null_strs=["DD"],
                        ids={"id2": "str"},
                        features={"feature2": "str"},
                    )
                ],
            ),
        ],
        output_uris=[output_path, output_report_path],
    )

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    logging.info(f"========output_path:{output_path}=====")

    if "alice" == sf_cluster_config.private_config.self_party:
        csv_a = orc.read_table(storage.get_reader(output_path)).to_pandas()
        logging.info(f"========csv_a:{csv_a}=====")

        assert_almost_equal(
            expected_result_a,
            orc.read_table(storage.get_reader(output_path)),
            ignore_order=True,
            check_dtype=False,
        )
        # storage.remove(output_path)
    if "bob" == sf_cluster_config.private_config.self_party:
        csv_b = orc.read_table(storage.get_reader(output_path)).to_pandas()
        logging.info(f"========csv_b:{csv_b}=====")

        assert_almost_equal(
            expected_result_b,
            csv_b,
            ignore_order=True,
            check_like=True,
            check_dtype=False,
        )
        # storage.remove(output_path)

    output_vt = VerticalTable()

    assert res.outputs[0].meta.Unpack(output_vt)
    assert len(output_vt.schemas) == 2

    assert output_vt.line_count == 4
    assert output_vt.schemas[0].ids == ["id1"]
    assert output_vt.schemas[0].features == ["item", "feature1"]
    assert output_vt.schemas[1].ids == ["id2"]
    assert output_vt.schemas[1].features == ["feature2"]

    report = Report()
    assert res.outputs[1].meta.Unpack(report)


@pytest.mark.mpc
def test_psi_left(sf_production_setup_comp):
    receiver_input_path = "test_psi/input_ds1.csv"
    sender_input_path = "test_psi/input_ds2.csv"
    output_path = "test_psi/psi_output.csv"
    output_report_path = "test_psi/psi_output"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        da.to_csv(storage.get_writer(receiver_input_path), index=False)

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K500", "K200", "K300", "K400", "K600", "K700"],
                "feature2": ["DD", "AA", "BB", "CC", "EE", "FF"],
                "feature3": [1 for _ in range(6)],
            }
        )

        db.to_csv(storage.get_writer(sender_input_path), index=False)

    expected_result_a = pd.DataFrame(
        {
            "item": ["B", "D", "E", "G", "A"],
            "feature1": ["BBB", "DDD", "EEE", "GGG", "AAA"],
            "id1": ["K200", "K300", "K400", "K500", "K100"],
        }
    )
    expected_result_b = pd.DataFrame(
        {
            "feature2": ["AA", "BB", "CC", "DD", None],
            "feature3": [1, 1, 1, 1, None],
            "id2": ["K200", "K300", "K400", "K500", None],
        }
    )

    param = build_node_eval_param(
        domain="data_prep",
        name="psi",
        version="1.0.0",
        attrs={
            "protocol": "PROTOCOL_ECDH",
            "sort_result": True,
            "join_type": "left_join",
            "join_type/left_join/left_side": ["alice"],
            "input/input_ds1/keys": ["id1"],
            "input/input_ds2/keys": ["id2"],
        },
        inputs=[
            VTable(
                name="input_ds1",
                parties=[
                    VTableParty.from_dict(
                        uri=receiver_input_path,
                        party="alice",
                        format="csv",
                        ids={"id1": "str"},
                        features={"item": "str", "feature1": "str"},
                    )
                ],
            ),
            VTable(
                name="input_ds2",
                parties=[
                    VTableParty.from_dict(
                        uri=sender_input_path,
                        party="bob",
                        format="csv",
                        ids={"id2": "str"},
                        features={"feature2": "str", "feature3": "int"},
                    )
                ],
            ),
        ],
        output_uris=[output_path, output_report_path],
    )

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    if "alice" == sf_cluster_config.private_config.self_party:
        assert_almost_equal(
            expected_result_a,
            orc.read_table(storage.get_reader(output_path)),
            ignore_order=True,
            check_dtype=False,
        )
        # storage.remove(output_path)

    if "bob" == sf_cluster_config.private_config.self_party:
        csv_b = orc.read_table(storage.get_reader(output_path))

        csv_b.equals(pa.Table.from_pandas(expected_result_b))
        # storage.remove(output_path)

    output_vt = VerticalTable()

    assert res.outputs[0].meta.Unpack(output_vt)
    assert len(output_vt.schemas) == 2

    assert output_vt.line_count == 5
    assert output_vt.schemas[0].ids == ["id1"]
    assert output_vt.schemas[0].features == ["item", "feature1"]
    assert output_vt.schemas[1].ids == ["id2"]
    assert output_vt.schemas[1].features == ["feature2", "feature3"]


@pytest.mark.mpc
def test_psi_one_receiver(sf_production_setup_comp):
    receiver_input_path = "test_psi/input_ds1.csv"
    sender_input_path = "test_psi/input_ds2.csv"
    output_path = "test_psi/psi_output.csv"
    output_report_path = "test_psi/psi_output"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        da.to_csv(storage.get_writer(receiver_input_path), index=False)

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K500", "K200", "K300", "K400", "K600", "K700"],
                "feature2": ["DD", "AA", "BB", "CC", "EE", "FF"],
                "feature3": [1 for _ in range(6)],
            }
        )

        db.to_csv(storage.get_writer(sender_input_path), index=False)

    expected_result_a = pd.DataFrame(
        {
            "id1": ["K200", "K300", "K400", "K500"],
            "item": ["B", "D", "E", "G"],
            "feature1": ["BBB", "DDD", "EEE", "GGG"],
        }
    )

    param = build_node_eval_param(
        domain="data_prep",
        name="psi",
        version="1.0.0",
        attrs={
            "protocol": "PROTOCOL_ECDH",
            "sort_result": True,
            "input/input_ds1/keys": ["id1"],
            "input/input_ds2/keys": ["id2"],
            "receiver_parties": ["alice"],
        },
        inputs=[
            VTable(
                name="input_ds1",
                parties=[
                    VTableParty.from_dict(
                        uri=receiver_input_path,
                        party="alice",
                        format="csv",
                        ids={"id1": "str"},
                        features={"item": "str", "feature1": "str"},
                    )
                ],
            ),
            VTable(
                name="input_ds2",
                parties=[
                    VTableParty.from_dict(
                        uri=sender_input_path,
                        party="bob",
                        format="csv",
                        ids={"id2": "str"},
                        features={"feature2": "str", "feature3": "int"},
                    )
                ],
            ),
        ],
        output_uris=[output_path, output_report_path],
    )

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    if "alice" == sf_cluster_config.private_config.self_party:
        assert_almost_equal(
            expected_result_a,
            orc.read_table(storage.get_reader(output_path)),
            ignore_order=True,
            check_dtype=False,
            check_like=True,
        )
        # storage.remove(output_path)

    if "bob" == sf_cluster_config.private_config.self_party:
        assert not storage.exists(output_path)

    output_vt = IndividualTable()

    assert res.outputs[0].meta.Unpack(output_vt)

    assert output_vt.line_count == 4
    assert output_vt.schema.ids == ["id1"]
    assert set(output_vt.schema.features) == set(["item", "feature1"])


@pytest.mark.mpc
def test_psi_left_long_output_path(sf_production_setup_comp):
    receiver_input_path = "test_psi/input_ds1.csv"
    sender_input_path = "test_psi/input_ds2.csv"
    output_path = "test_psi/xxx/uuu/ccc/psi_output.csv"
    output_report_path = "test_psi/psi_output"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        da.to_csv(storage.get_writer(receiver_input_path), index=False)

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K500", "K200", "K300", "K400", "K600", "K700"],
                "feature2": ["DD", "AA", "BB", "CC", "EE", "FF"],
                "feature3": [1 for _ in range(6)],
            }
        )

        db.to_csv(storage.get_writer(sender_input_path), index=False)

    expected_result_a = pd.DataFrame(
        {
            "item": ["B", "D", "E", "G", "A"],
            "feature1": ["BBB", "DDD", "EEE", "GGG", "AAA"],
            "id1": ["K200", "K300", "K400", "K500", "K100"],
        }
    )
    expected_result_b = pd.DataFrame(
        {
            "feature2": ["AA", "BB", "CC", "DD", None],
            "feature3": [1, 1, 1, 1, None],
            "id2": ["K200", "K300", "K400", "K500", None],
        }
    )

    param = build_node_eval_param(
        domain="data_prep",
        name="psi",
        version="1.0.0",
        attrs={
            "protocol": "PROTOCOL_ECDH",
            "sort_result": True,
            "join_type": "left_join",
            "join_type/left_join/left_side": ["alice"],
            "input/input_ds1/keys": ["id1"],
            "input/input_ds2/keys": ["id2"],
        },
        inputs=[
            VTable(
                name="input_ds1",
                parties=[
                    VTableParty.from_dict(
                        uri=receiver_input_path,
                        party="alice",
                        format="csv",
                        ids={"id1": "str"},
                        features={"item": "str", "feature1": "str"},
                    )
                ],
            ),
            VTable(
                name="input_ds2",
                parties=[
                    VTableParty.from_dict(
                        uri=sender_input_path,
                        party="bob",
                        format="csv",
                        ids={"id2": "str"},
                        features={"feature2": "str", "feature3": "int"},
                    )
                ],
            ),
        ],
        output_uris=[output_path, output_report_path],
    )

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    if "alice" == sf_cluster_config.private_config.self_party:
        assert_almost_equal(
            expected_result_a,
            orc.read_table(storage.get_reader(output_path)),
            ignore_order=True,
            check_dtype=False,
        )
        # storage.remove(output_path)

    if "bob" == sf_cluster_config.private_config.self_party:
        csv_b = orc.read_table(
            storage.get_reader(output_path),
        )

        csv_b.equals(pa.Table.from_pandas(expected_result_b))
        # storage.remove(output_path)

    output_vt = VerticalTable()

    assert res.outputs[0].meta.Unpack(output_vt)
    assert len(output_vt.schemas) == 2

    assert output_vt.line_count == 5
    assert output_vt.schemas[0].ids == ["id1"]
    assert output_vt.schemas[0].features == ["item", "feature1"]
    assert output_vt.schemas[1].ids == ["id2"]
    assert output_vt.schemas[1].features == ["feature2", "feature3"]


@pytest.mark.mpc
def test_psi_orc_empty_intersect(sf_production_setup_comp):
    receiver_input_path = "test_psi/input_ds1.orc"
    sender_input_path = "test_psi/input_ds2.csv"
    output_path = "test_psi/psi_output.csv"
    output_report_path = "test_psi/psi_output"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        da.to_orc(storage.get_writer(receiver_input_path), index=False)

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K600", "K700"],
                "feature2": ["EE", "FF"],
            }
        )

        db.to_csv(storage.get_writer(sender_input_path), index=False)

    param = build_node_eval_param(
        domain="data_prep",
        name="psi",
        version="1.0.0",
        attrs={
            "protocol": "PROTOCOL_ECDH",
            "sort_result": True,
            "allow_empty_result": True,
            "input/input_ds1/keys": ["id1"],
            "input/input_ds2/keys": ["id2"],
            "receiver_parties": ["alice", "bob"],
        },
        inputs=[
            VTable(
                name="input_ds1",
                parties=[
                    VTableParty.from_dict(
                        uri=receiver_input_path,
                        party="alice",
                        format="orc",
                        # null_strs=["GGG"],
                        ids={"id1": "str"},
                        features={"item": "str", "feature1": "str"},
                    )
                ],
            ),
            VTable(
                name="input_ds2",
                parties=[
                    VTableParty.from_dict(
                        uri=sender_input_path,
                        party="bob",
                        format="csv",
                        null_strs=["DD"],
                        ids={"id2": "str"},
                        features={"feature2": "str"},
                    )
                ],
            ),
        ],
        output_uris=[output_path, output_report_path],
    )

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    output_vt = VerticalTable()

    assert res.outputs[0].meta.Unpack(output_vt)
    assert len(output_vt.schemas) == 2

    assert output_vt.line_count == 0
    assert output_vt.schemas[0].ids == ["id1"]
    assert output_vt.schemas[0].features == ["item", "feature1"]
    assert output_vt.schemas[1].ids == ["id2"]
    assert output_vt.schemas[1].features == ["feature2"]

    if "alice" == sf_cluster_config.private_config.self_party:
        shape = orc.read_table(storage.get_reader(output_path)).to_pandas().shape
        assert shape[0] == 0
        # storage.remove(output_path)
    if "bob" == sf_cluster_config.private_config.self_party:
        shape = orc.read_table(storage.get_reader(output_path)).to_pandas().shape
        assert shape[0] == 0
        # storage.remove(output_path)


@pytest.mark.mpc
def test_psi_orc_empty_intersect_error(sf_production_setup_comp):
    receiver_input_path = "test_psi/input_ds1.orc"
    sender_input_path = "test_psi/input_ds2.csv"
    output_path = "test_psi/psi_output.csv"
    output_report_path = "test_psi/psi_output"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        da.to_orc(storage.get_writer(receiver_input_path), index=False)

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K600", "K700"],
                "feature2": ["EE", "FF"],
            }
        )

        db.to_csv(storage.get_writer(sender_input_path), index=False)

    param = build_node_eval_param(
        domain="data_prep",
        name="psi",
        version="1.0.0",
        attrs={
            "protocol": "PROTOCOL_ECDH",
            "sort_result": True,
            "input/input_ds1/keys": ["id1"],
            "input/input_ds2/keys": ["id2"],
            "receiver_parties": ["alice", "bob"],
        },
        inputs=[
            VTable(
                name="input_ds1",
                parties=[
                    VTableParty.from_dict(
                        uri=receiver_input_path,
                        party="alice",
                        format="orc",
                        # null_strs=["GGG"],
                        ids={"id1": "str"},
                        features={"item": "str", "feature1": "str"},
                    )
                ],
            ),
            VTable(
                name="input_ds2",
                parties=[
                    VTableParty.from_dict(
                        uri=sender_input_path,
                        party="bob",
                        format="csv",
                        null_strs=["DD"],
                        ids={"id2": "str"},
                        features={"feature2": "str"},
                    )
                ],
            ),
        ],
        output_uris=[output_path, output_report_path],
    )

    with pytest.raises(
        Exception,
        match="Empty result is not allowed, please check your input data or set allow_empty_result to true.",
    ):
        comp_eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )
