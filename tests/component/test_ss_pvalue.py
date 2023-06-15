import logging
import os

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from secretflow.component.data_utils import DistDataType
from secretflow.component.ml.linear.ss_sgd import ss_sgd_train_comp
from secretflow.component.stats.ss_pvalue import ss_pvalue_comp
from secretflow.protos.component.comp_pb2 import Attribute
from secretflow.protos.component.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.protos.component.evaluation_pb2 import NodeEvalParam
from secretflow.protos.component.report_pb2 import Report


def test_ss_pvalue(comp_prod_sf_cluster_config):
    alice_input_path = "test_ss_pvalue/alice.csv"
    bob_input_path = "test_ss_pvalue/bob.csv"
    model_path = "test_ss_pvalue/model.sf"

    self_party = comp_prod_sf_cluster_config.private_config.self_party
    local_fs_wd = comp_prod_sf_cluster_config.private_config.storage_config.local_fs.wd

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, "test_ss_pvalue"),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f'a{i}' for i in range(15)])
        y = pd.DataFrame(y, columns=['y'])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(os.path.join(local_fs_wd, alice_input_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, "test_ss_pvalue"),
            exist_ok=True,
        )

        ds = pd.DataFrame(x[:, 15:], columns=[f'b{i}' for i in range(15)])
        ds.to_csv(os.path.join(local_fs_wd, bob_input_path), index=False)

    train_param = NodeEvalParam(
        domain="ml.linear",
        name="ss_sgd_train",
        version="0.0.1",
        attr_paths=[
            "epochs",
            "learning_rate",
            "batch_size",
            "sig_type",
            "reg_type",
            "penalty",
            "l2_norm",
            "decay_epoch",
            "decay_rate",
            "strategy",
        ],
        attrs=[
            Attribute(i64=3),
            Attribute(f=0.3),
            Attribute(i64=128),
            Attribute(s="t1"),
            Attribute(s="logistic"),
            Attribute(s="l2"),
            Attribute(f=0.05),
            Attribute(i64=2),
            Attribute(f=0.5),
            Attribute(s="policy_sgd"),
        ],
        inputs=[
            DistData(
                name="train_dataset",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[model_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                types=["f32"] * 15,
                features=[f"a{i}" for i in range(15)],
                labels=["y"],
            ),
            TableSchema(
                types=["f32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    train_param.inputs[0].meta.Pack(meta)

    train_res = ss_sgd_train_comp.eval(train_param, comp_prod_sf_cluster_config)

    pv_param = NodeEvalParam(
        domain="stats",
        name="ss_pvalue",
        version="0.0.1",
        inputs=[train_res.outputs[0], train_param.inputs[0]],
        output_uris=["report"],
    )

    res = ss_pvalue_comp.eval(pv_param, comp_prod_sf_cluster_config)

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
    assert len(descriptions.items) == 30 + 1
