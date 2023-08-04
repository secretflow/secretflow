import logging
import os

import pandas as pd
from sklearn.datasets import load_breast_cancer

from secretflow.component.data_utils import DistDataType
from secretflow.component.stats.ss_pearsonr import ss_pearsonr_comp
from secretflow.protos.component.comp_pb2 import Attribute
from secretflow.protos.component.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.protos.component.evaluation_pb2 import NodeEvalParam
from secretflow.protos.component.report_pb2 import Report


def test_ss_pearsonr(comp_prod_sf_cluster_config):
    alice_input_path = "test_ss_pearsonr/alice.csv"
    bob_input_path = "test_ss_pearsonr/bob.csv"

    self_party = comp_prod_sf_cluster_config.private_config.self_party
    local_fs_wd = comp_prod_sf_cluster_config.private_config.storage_config.local_fs.wd

    x = load_breast_cancer()["data"]
    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, "test_ss_pearsonr"),
            exist_ok=True,
        )
        ds = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        ds.to_csv(os.path.join(local_fs_wd, alice_input_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, "test_ss_pearsonr"),
            exist_ok=True,
        )
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(os.path.join(local_fs_wd, bob_input_path), index=False)

    param = NodeEvalParam(
        domain="stats",
        name="ss_pearsonr",
        version="0.0.1",
        attr_paths=[
            "input/input_data/feature_selects",
        ],
        attrs=[
            Attribute(ss=["a1", "b1", "a3", "b13"]),
        ],
        inputs=[
            DistData(
                name="input",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
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
    param.inputs[0].meta.Pack(meta)

    res = ss_pearsonr_comp.eval(param, comp_prod_sf_cluster_config)

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
    assert c.type == "table"
    table = c.table
    assert len(table.headers) == 4
    assert len(table.rows) == 4
