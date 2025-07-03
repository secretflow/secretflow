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
import math

import pandas as pd
import pytest
from secretflow_spec.v1.report_pb2 import Report

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


@pytest.mark.mpc
def test_prediction_bias_eval(sf_production_setup_comp):
    labels = [1, 0, 0, 0, 0, 1, 1, 1]
    predictions = [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8]
    label_pred_df = pd.DataFrame(
        {
            "labels": labels,
            "predictions": predictions,
        }
    )

    alice_label_pred_path = "prediction_bias_eval/alice_label_predict.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        label_pred_df.to_csv(storage.get_writer(alice_label_pred_path), index=False)

    param = build_node_eval_param(
        domain="ml.eval",
        name="prediction_bias_eval",
        version="1.0.0",
        attrs={
            "bucket_num": 4,
            "min_item_cnt_per_bucket": 2,
            "bucket_method": "equal_frequency",
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
