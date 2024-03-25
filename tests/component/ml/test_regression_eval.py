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

import numpy as np
import pandas as pd
from google.protobuf.json_format import MessageToJson
from sklearn.metrics import r2_score

from secretflow.component.data_utils import DistDataType
from secretflow.component.ml.eval.regression_eval import regression_eval_comp
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, TableSchema
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report


def test_regression_eval(comp_prod_sf_cluster_config):
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
    comp_storage = ComponentStorage(storage_config)

    if self_party == "alice":
        label_pred_df.to_csv(
            comp_storage.get_writer(alice_label_pred_path), index=False
        )

    param = NodeEvalParam(
        domain="ml.eval",
        name="regression_eval",
        version="0.0.1",
        attr_paths=[
            "bucket_size",
            "input/in_ds/label",
            "input/in_ds/prediction",
        ],
        attrs=[
            Attribute(i64=2),
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

    res = regression_eval_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.warning(MessageToJson(comp_ret))

    np.testing.assert_almost_equal(
        r2_score(labels.reshape(-1, 1), predictions.reshape(-1, 1)),
        comp_ret.tabs[0].divs[0].children[0].descriptions.items[0].value.f,
        decimal=5,
    )
