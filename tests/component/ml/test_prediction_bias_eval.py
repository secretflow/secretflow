import logging
import math

import pandas as pd

from secretflow.component.data_utils import DistDataType
from secretflow.component.ml.eval.prediction_bias_eval import prediction_bias_comp
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report


def test_prediction_bias_eval(comp_prod_sf_cluster_config):
    labels = [1, 0, 0, 0, 0, 1, 1, 1]
    predictions = [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8]
    label_pred_df = pd.DataFrame(
        {
            "labels": labels,
            "predictions": predictions,
        }
    )

    alice_label_pred_path = "prediction_bias_eval/alice_label_predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    if self_party == "alice":
        label_pred_df.to_csv(
            comp_storage.get_writer(alice_label_pred_path), index=False
        )

    param = NodeEvalParam(
        domain="ml.eval",
        name="prediction_bias_eval",
        version="0.0.1",
        attr_paths=[
            "bucket_num",
            "min_item_cnt_per_bucket",
            "bucket_method",
            "input/in_ds/label",
            "input/in_ds/prediction",
        ],
        attrs=[
            Attribute(i64=4),
            Attribute(i64=2),
            Attribute(s='equal_frequency'),
            Attribute(ss=["labels"]),
            Attribute(ss=["predictions"]),
        ],
        inputs=[
            DistData(
                name="in_ds",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=alice_label_pred_path, party="alice", format="csv"
                    ),
                ],
            ),
        ],
        output_uris=[""],
    )
    logging.info(param)
    meta = VerticalTable(
        schemas=[
            TableSchema(
                label_types=["float32", "float32"],
                labels=["labels", "predictions"],
            )
        ],
    )
    param.inputs[0].meta.Pack(meta)

    res = prediction_bias_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.info(comp_ret)

    table = comp_ret.tabs[0].divs[0].children[0].table
    assert len(table.rows) == 4

    assert math.isclose(table.rows[0].items[1].f, 0.1, rel_tol=1e-5)
    assert table.rows[0].items[2].b
    assert math.isclose(table.rows[0].items[3].f, 0.3, rel_tol=1e-5)
    assert not table.rows[0].items[4].b
    assert math.isclose(table.rows[0].items[5].f, 0.15, rel_tol=1e-5)
    assert math.isclose(table.rows[0].items[6].f, 0, rel_tol=1e-5)
    assert math.isclose(table.rows[0].items[7].f, 0.15, rel_tol=1e-5)
