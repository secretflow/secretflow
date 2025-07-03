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
import pytest
from google.protobuf.json_format import MessageToJson
from secretflow_spec.v1.report_pb2 import Report
from sklearn.metrics import r2_score

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


@pytest.mark.mpc
def test_regression_eval(sf_production_setup_comp):
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

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        label_pred_df.to_csv(storage.get_writer(alice_label_pred_path), index=False)

    param = build_node_eval_param(
        domain="ml.eval",
        name="regression_eval",
        version="1.0.0",
        attrs={
            "bucket_size": 2,
            "input/input_ds/label": ["labels"],
            "input/input_ds/prediction": ["predictions"],
        },
        inputs=[
            VTable(
                name="in_ds",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_label_pred_path,
                        party="alice",
                        format="csv",
                        labels={"labels": "float32", "predictions": "float32"},
                    ),
                ],
            ),
        ],
        output_uris=[""],
    )

    res = comp_eval(
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
