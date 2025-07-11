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
from enum import Enum

import pandas as pd
import pytest
from secretflow_spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow_spec.v1.report_pb2 import Report
from sklearn.datasets import load_breast_cancer

from secretflow.component.core import (
    DistDataType,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


class TableFormat(Enum):
    INDIVIDUAL_TABLE = 0
    VERTICAL_TABLE = 2


@pytest.mark.parametrize(
    "table_format",
    [TableFormat.INDIVIDUAL_TABLE, TableFormat.VERTICAL_TABLE],
)
@pytest.mark.mpc
def test_ss_vif(sf_production_setup_comp, table_format: TableFormat):
    alice_input_path = "test_ss_vif/alice.csv"
    bob_input_path = "test_ss_vif/bob.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    x = load_breast_cancer()["data"]
    if table_format == TableFormat.VERTICAL_TABLE:
        if self_party == "alice":
            ds = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
            ds.to_csv(storage.get_writer(alice_input_path), index=False)

        elif self_party == "bob":
            ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
            ds.to_csv(storage.get_writer(bob_input_path), index=False)

        param = build_node_eval_param(
            domain="stats",
            name="ss_vif",
            version="1.0.0",
            attrs={
                "input/input_ds/feature_selects": ["a1", "b1", "a3", "b13"],
            },
            inputs=[
                DistData(
                    name="input",
                    type=str(DistDataType.VERTICAL_TABLE),
                    data_refs=[
                        DistData.DataRef(
                            uri=alice_input_path, party="alice", format="csv"
                        ),
                        DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                    ],
                ),
            ],
            output_uris=["report"],
        )

        meta = VerticalTable(
            schemas=[
                TableSchema(
                    feature_types=["float32"] * 15,
                    features=[f"a{i}" for i in range(15)],
                ),
                TableSchema(
                    feature_types=["float32"] * 15,
                    features=[f"b{i}" for i in range(15)],
                ),
            ],
        )
    else:
        if self_party == "alice":
            ds = pd.DataFrame(
                x[:, :],
                columns=[f"a{i}" for i in range(15)] + [f"b{i}" for i in range(15)],
            )
            ds.to_csv(storage.get_writer(alice_input_path), index=False)

        param = build_node_eval_param(
            domain="stats",
            name="ss_vif",
            version="1.0.0",
            attrs={
                "input/input_ds/feature_selects": ["a1", "b1", "a3", "b13"],
            },
            inputs=[
                DistData(
                    name="input",
                    type=str(DistDataType.INDIVIDUAL_TABLE),
                    data_refs=[
                        DistData.DataRef(
                            uri=alice_input_path, party="alice", format="csv"
                        ),
                    ],
                ),
            ],
            output_uris=["report"],
        )

        meta = IndividualTable(
            schema=TableSchema(
                feature_types=["float32"] * 30,
                features=[f"a{i}" for i in range(15)] + [f"b{i}" for i in range(15)],
            ),
        )

    param.inputs[0].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    report = Report()
    assert res.outputs[0].meta.Unpack(report)

    logging.info(report)

    assert len(report.tabs) == 1
    tab = report.tabs[0]
    assert len(tab.divs) == 1
    div = tab.divs[0]
    assert len(div.children) == 1
    c = div.children[0]
    assert c.type == "descriptions"
    descriptions = c.descriptions
    assert len(descriptions.items) == 4
