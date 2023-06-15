import logging
import os

import numpy as np
import pandas as pd

from secretflow.component.data_utils import DistDataType
from secretflow.component.stats.pva_eval import pva_value_comp
from secretflow.protos.component.comp_pb2 import Attribute
from secretflow.protos.component.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.protos.component.evaluation_pb2 import NodeEvalParam
from secretflow.protos.component.report_pb2 import Report


def test_pva_eval(comp_prod_sf_cluster_config):
    y_actual_df = pd.DataFrame(
        {
            'y_actual': [*range(10)],
        }
    )
    y_prediction_df = pd.DataFrame(
        {
            "y_prediction": [0.1 for _ in range(10)],
        }
    )

    alice_actual_path = "pva_eval/alice_actual.csv"
    alice_predict_path = "pva_eval/bob_predict.csv"

    self_party = comp_prod_sf_cluster_config.private_config.self_party
    local_fs_wd = comp_prod_sf_cluster_config.private_config.storage_config.local_fs.wd

    if self_party == 'alice':
        os.makedirs(os.path.join(local_fs_wd, "pva_eval"), exist_ok=True)
        y_actual_df.to_csv(os.path.join(local_fs_wd, alice_actual_path), index=False)
        os.makedirs(os.path.join(local_fs_wd, "pva_eval"), exist_ok=True)
        y_prediction_df.to_csv(
            os.path.join(local_fs_wd, alice_predict_path), index=False
        )

    param = NodeEvalParam(
        domain="stats",
        name="pva_eval",
        version="0.0.1",
        attr_paths=["target"],
        attrs=[Attribute(f=8)],
        inputs=[
            DistData(
                name="actual",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=alice_actual_path, party="alice", format="csv"
                    ),
                ],
            ),
            DistData(
                name="prediction",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=alice_predict_path, party="alice", format="csv"
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
                types=["f32"],
                features=["y_actual"],
            )
        ],
    )
    param.inputs[0].meta.Pack(meta)
    meta = VerticalTable(
        schemas=[
            TableSchema(
                types=["f32"],
                features=["y_prediction"],
            )
        ],
    )
    param.inputs[1].meta.Pack(meta)

    res = pva_value_comp.eval(param=param, cluster_config=comp_prod_sf_cluster_config)
    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.info(comp_ret)
    np.testing.assert_almost_equal(
        comp_ret.tabs[0].divs[0].children[0].descriptions.items[0].value.f,
        0.0,
        decimal=2,
    )
