import logging
import os

import numpy as np
import pandas as pd
from google.protobuf.json_format import MessageToJson
from sklearn.metrics import roc_auc_score

from secretflow.component.data_utils import DistDataType
from secretflow.component.stats.biclassification_eval import biclassification_eval_comp
from secretflow.protos.component.comp_pb2 import Attribute
from secretflow.protos.component.data_pb2 import (
    DistData,
    TableSchema,
    VerticalTable,
    IndividualTable,
)
from secretflow.protos.component.evaluation_pb2 import NodeEvalParam
from secretflow.protos.component.report_pb2 import Report


def test_biclassification_eval(comp_prod_sf_cluster_config):
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.1])
    y_true_df = pd.DataFrame(
        {
            "y_true": y_true,
        }
    )
    y_pred_df = pd.DataFrame(
        {
            "y_pred": y_pred,
        }
    )

    alice_true_path = "biclassification_eval/alice_true.csv"
    alice_pred_path = "biclassification_eval/alice_pred.csv"

    self_party = comp_prod_sf_cluster_config.private_config.self_party
    local_fs_wd = comp_prod_sf_cluster_config.private_config.storage_config.local_fs.wd

    if self_party == "alice":
        os.makedirs(os.path.join(local_fs_wd, "biclassification_eval"), exist_ok=True)
        y_true_df.to_csv(os.path.join(local_fs_wd, alice_true_path), index=False)
        y_pred_df.to_csv(os.path.join(local_fs_wd, alice_pred_path), index=False)

    param = NodeEvalParam(
        domain="stats",
        name="biclassification_eval",
        version="0.0.1",
        attr_paths=["bucket_size", "input/y_true/col", "input/y_score/col"],
        attrs=[Attribute(i64=2), Attribute(ss=["y_true"]), Attribute(ss=["y_pred"])],
        inputs=[
            DistData(
                name="y_true",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_true_path, party="alice", format="csv"),
                ],
            ),
            DistData(
                name="y_score",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_pred_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[""],
    )
    meta = IndividualTable(
        schema=TableSchema(
            labels=["y_true"],
        ),
    )
    param.inputs[0].meta.Pack(meta)
    meta = VerticalTable(
        schemas=[
            TableSchema(
                labels=["y_pred"],
            )
        ],
    )
    param.inputs[1].meta.Pack(meta)

    res = biclassification_eval_comp.eval(
        param=param, cluster_config=comp_prod_sf_cluster_config
    )
    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.warn(MessageToJson(comp_ret))
    np.testing.assert_almost_equal(
        roc_auc_score(y_true.reshape(-1, 1), y_pred.reshape(-1, 1)),
        comp_ret.tabs[0].divs[0].children[0].descriptions.items[3].value.f,
        decimal=2,
    )
