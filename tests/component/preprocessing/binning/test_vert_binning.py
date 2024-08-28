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
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import orc
from sklearn.datasets import load_breast_cancer

from secretflow.component.data_utils import DistDataType, extract_data_infos
from secretflow.component.model_export.serving_utils.preprocessing_converter import (
    apply_binning_rules,
)
from secretflow.component.preprocessing.binning.vert_binning import (
    vert_bin_substitution_comp,
    vert_binning_comp,
)
from secretflow.component.storage import ComponentStorage
from secretflow.compute import Table
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report
from secretflow.utils import secure_pickle as pickle


def test_vert_binning(comp_prod_sf_cluster_config):
    alice_path = "test_vert_binning/x_alice.csv"
    bob_path = "test_vert_binning/x_bob.csv"
    rule_path = "test_vert_binning/bin_rule"
    report_path = "test_vert_binning/report"
    output_path = "test_vert_binning/vert.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    ds = load_breast_cancer()
    x, y = ds["data"], ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(comp_storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(comp_storage.get_writer(bob_path), index=False)

    bin_param_01 = NodeEvalParam(
        domain="feature",
        name="vert_binning",
        version="0.0.2",
        attr_paths=["input/input_data/feature_selects", "report_rules"],
        attrs=[
            Attribute(ss=[f"a{i}" for i in range(12)] + [f"b{i}" for i in range(11)]),
            Attribute(b=True),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[rule_path, report_path],
    )

    bin_param_02 = NodeEvalParam(
        domain="feature",
        name="vert_binning",
        version="0.0.2",
        attr_paths=["input/input_data/feature_selects", "report_rules"],
        attrs=[
            Attribute(ss=[f"a{i}" for i in range(11)] + [f"b{i}" for i in range(12)]),
            Attribute(b=True),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[rule_path, report_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
                label_types=["float32"],
                labels=["y"],
            ),
        ],
    )
    bin_param_01.inputs[0].meta.Pack(meta)
    bin_param_02.inputs[0].meta.Pack(meta)
    """
    bin_res = vert_binning_comp.eval(
        param=bin_param_01,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    """
    bin_res = vert_binning_comp.eval(
        param=bin_param_02,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(bin_res.outputs) == 2
    comp_ret = Report()
    bin_res.outputs[1].meta.Unpack(comp_ret)
    logging.info("bin_res.outputs[1]: %s", comp_ret)
    sub_param = NodeEvalParam(
        domain="preprocessing",
        name="vert_bin_substitution",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
        inputs=[
            bin_param_01.inputs[0],
            bin_res.outputs[0],
        ],
        output_uris=[output_path],
    )

    sub_res = vert_bin_substitution_comp.eval(
        param=sub_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(sub_res.outputs) == 1

    input_info = extract_data_infos(
        bin_param_01.inputs[0],
        load_features=True,
        load_labels=True,
        load_ids=True,
    )

    output_info = extract_data_infos(
        sub_res.outputs[0],
        load_features=True,
        load_labels=True,
        load_ids=True,
    )

    comp_storage = ComponentStorage(storage_config)
    if self_party == "alice":
        alice_input = pd.read_csv(comp_storage.get_reader(alice_path), dtype=np.float32)

        alice_rule = os.path.join(rule_path, "1")
        with comp_storage.get_reader(alice_rule) as f:
            alice_rule = pickle.loads(f.read())

        alice_table = apply_binning_rules(alice_rule, input_info["alice"].dtypes)
        alice_runner = alice_table.dump_runner()
        alice_table_out = alice_runner.run(
            pa.Table.from_pandas(alice_input)
        ).to_pandas()

        alice_out = orc.read_table(
            comp_storage.get_reader(output_info["alice"].uri)
        ).to_pandas()

        assert np.isclose(alice_table_out.values, alice_out.values).all()

        table = Table.from_schema(output_info["alice"].dtypes)

        assert table.schema == alice_table.schema

    if self_party == "bob":
        bob_input = pd.read_csv(comp_storage.get_reader(bob_path), dtype=np.float32)

        bob_rule = os.path.join(rule_path, "0")
        with comp_storage.get_reader(bob_rule) as f:
            bob_rule = pickle.loads(f.read())

        bob_table = apply_binning_rules(bob_rule, input_info["bob"].dtypes)
        bob_runner = bob_table.dump_runner()
        bob_table_out = bob_runner.run(pa.Table.from_pandas(bob_input)).to_pandas()

        bob_out = orc.read_table(
            comp_storage.get_reader(output_info["bob"].uri)
        ).to_pandas()

        assert np.isclose(bob_table_out.values, bob_out.values).all()

        table = Table.from_schema(output_info["bob"].dtypes)

        assert table.schema == bob_table.schema
