import logging
import os

import numpy as np
import pandas as pd
from google.protobuf.json_format import MessageToJson
from sklearn.metrics import roc_auc_score

from secretflow.component.data_utils import DistDataType
from secretflow.component.ml.eval.biclassification_eval import (
    biclassification_eval_comp,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, TableSchema
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report


def test_biclassification_eval(comp_prod_sf_cluster_config):
    np.random.seed(42)
    labels = np.round(np.random.random((800000,)))
    predictions = np.random.random((800000,))
    label_pred_df = pd.DataFrame(
        {
            "labels": labels,
            "predictions": predictions,
        }
    )

    alice_label_pred_path = "biclassification_eval/alice_label_pred.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

    if self_party == "alice":
        os.makedirs(os.path.join(local_fs_wd, "biclassification_eval"), exist_ok=True)
        label_pred_df.to_csv(
            os.path.join(local_fs_wd, alice_label_pred_path), index=False
        )

    param = NodeEvalParam(
        domain="ml.eval",
        name="biclassification_eval",
        version="0.0.1",
        attr_paths=[
            "bucket_size",
            "min_item_cnt_per_bucket",
            "input/in_ds/label",
            "input/in_ds/prediction",
        ],
        attrs=[
            Attribute(i64=2),
            Attribute(i64=5),
            Attribute(ss=["labels"]),
            Attribute(ss=["predictions"]),
        ],
        inputs=[
            DistData(
                name="in_ds",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=alice_label_pred_path, party="alice", format="csv"
                    ),
                ],
            ),
        ],
        output_uris=[""],
    )
    meta = IndividualTable(
        schema=TableSchema(
            labels=["labels", "predictions"], label_types=["float32", "float32"]
        ),
    )
    param.inputs[0].meta.Pack(meta)

    res = biclassification_eval_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.warning(MessageToJson(comp_ret))
    np.testing.assert_almost_equal(
        roc_auc_score(labels.reshape(-1, 1), predictions.reshape(-1, 1)),
        comp_ret.tabs[0].divs[0].children[0].descriptions.items[3].value.f,
        decimal=2,
    )
