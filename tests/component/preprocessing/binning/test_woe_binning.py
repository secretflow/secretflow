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


import numpy as np
import pandas as pd
import pytest
from pyarrow import orc
from secretflow_spec.v1.report_pb2 import Report
from sklearn.datasets import load_breast_cancer

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


@pytest.mark.mpc
def test_woe_binning(sf_production_setup_comp):
    alice_path = "test_woe_binning/x_alice.csv"
    bob_path = "test_woe_binning/x_bob.csv"
    bin_out_data_path = "test_vert_binning/output_df"
    bin_out_rule_path = "test_woe_binning/bin_rule"
    bin_out_report_path = "test_vert_binning/report"
    sub_out_data_path = "test_woe_binning/woe.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    ds = load_breast_cancer()
    x, y = ds["data"], ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(storage.get_writer(bob_path), index=False)

    bin_param = build_node_eval_param(
        domain="preprocessing",
        name="vert_woe_binning",
        version="1.0.0",
        attrs={
            "secure_device_type": "heu",
            "input/input_ds/feature_selects": [f"a{i}" for i in range(11)]
            + [f"b{i}" for i in range(12)],
            "input/input_ds/label": ["y"],
            "report_rules": True,
        },
        inputs=[
            VTable(
                name="input_data",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_path,
                        party="alice",
                        format="csv",
                        features={f"a{i}": "float32" for i in range(15)},
                        labels={"y": "float32"},
                    ),
                    VTableParty.from_dict(
                        uri=bob_path,
                        party="bob",
                        format="csv",
                        features={f"b{i}": "float32" for i in range(15)},
                    ),
                ],
            ),
        ],
        output_uris=[bin_out_data_path, bin_out_rule_path, bin_out_report_path],
    )

    bin_res = comp_eval(
        param=bin_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(bin_res.outputs) == 3
    comp_ret = Report()
    assert bin_res.outputs[2].meta.Unpack(comp_ret)

    sub_param = build_node_eval_param(
        domain="preprocessing",
        name="substitution",
        version="1.0.0",
        attrs=None,
        inputs=[bin_param.inputs[0], bin_res.outputs[1]],
        output_uris=[sub_out_data_path],
    )

    sub_res = comp_eval(
        param=sub_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(sub_res.outputs) == 1

    if self_party in ["alice", "bob"]:
        bin_output_info = VTable.from_distdata(bin_res.outputs[0])
        sub_output_info = VTable.from_distdata(sub_res.outputs[0])

        bin_output_df = orc.read_table(
            storage.get_reader(bin_output_info.get_party(self_party).uri)
        ).to_pandas()
        sub_output_df = orc.read_table(
            storage.get_reader(sub_output_info.get_party(self_party).uri)
        ).to_pandas()

        assert np.isclose(bin_output_df.values, sub_output_df.values).all()
