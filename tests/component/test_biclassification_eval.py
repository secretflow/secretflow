import logging
import os

import numpy as np
import pandas as pd
from google.protobuf.json_format import MessageToJson
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report
from sklearn.metrics import roc_auc_score

from secretflow.component.data_utils import DistDataType
from secretflow.component.ml.eval.biclassification_eval import (
    biclassification_eval_comp,
)


def test_biclassification_eval(comp_prod_sf_cluster_config):
    labels = np.array(
        [
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
        ]
    )
    predictions = np.array(
        [
            0.1,
            0.4,
            0.35,
            0.8,
            0.1,
            0.1,
            0.4,
            0.35,
            0.8,
            0.1,
            0.1,
            0.4,
            0.35,
            0.8,
            0.1,
            0.1,
            0.4,
            0.35,
            0.8,
            0.1,
            0.1,
            0.4,
            0.35,
            0.8,
            0.1,
            0.1,
            0.4,
            0.35,
            0.8,
            0.1,
            0.1,
            0.4,
            0.35,
            0.8,
            0.1,
            0.1,
            0.4,
            0.35,
            0.8,
            0.1,
        ]
    )
    labels_df = pd.DataFrame(
        {
            "labels": labels,
        }
    )
    predictions_df = pd.DataFrame(
        {
            "predictions": predictions,
        }
    )

    alice_true_path = "biclassification_eval/alice_true.csv"
    alice_pred_path = "biclassification_eval/alice_pred.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

    if self_party == "alice":
        os.makedirs(os.path.join(local_fs_wd, "biclassification_eval"), exist_ok=True)
        labels_df.to_csv(os.path.join(local_fs_wd, alice_true_path), index=False)
        predictions_df.to_csv(os.path.join(local_fs_wd, alice_pred_path), index=False)

    param = NodeEvalParam(
        domain="ml.eval",
        name="biclassification_eval",
        version="0.0.1",
        attr_paths=[
            "bucket_size",
            "min_item_cnt_per_bucket",
            "input/labels/col",
            "input/y_score/col",
        ],
        attrs=[
            Attribute(i64=2),
            Attribute(i64=5),
            Attribute(ss=["labels"]),
            Attribute(ss=["predictions"]),
        ],
        inputs=[
            DistData(
                name="labels",
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
        schema=TableSchema(labels=["labels"], label_types=["float32"]),
    )
    param.inputs[0].meta.Pack(meta)
    meta = VerticalTable(
        schemas=[TableSchema(labels=["predictions"], label_types=["float32"])],
    )
    param.inputs[1].meta.Pack(meta)

    res = biclassification_eval_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.warn(MessageToJson(comp_ret))
    np.testing.assert_almost_equal(
        roc_auc_score(labels.reshape(-1, 1), predictions.reshape(-1, 1)),
        comp_ret.tabs[0].divs[0].children[0].descriptions.items[3].value.f,
        decimal=2,
    )
