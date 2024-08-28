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

import numpy as np
import pandas as pd
from pyarrow import orc

import secretflow.compute as sc
from secretflow.component.core.storage import Storage
from secretflow.component.data_utils import DistDataType
from secretflow.component.entry import comp_eval
from secretflow.component.postprocessing.score_card_transformer import (
    ScoreCardTransformer,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, TableSchema
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_score_card_transformer(comp_prod_sf_cluster_config):
    input_path = "test_score_card_transformer/input.csv"
    output_path = "test_score_card_transformer/output.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    storage = Storage(storage_config)

    input_data = {
        "id": ["id1", "id2"],
        "pred": [0.1, 0.2],
    }

    expected_result = pd.DataFrame(
        {
            "id": ["id1", "id2"],
            "pred": [0.1, 0.2],
            "predict_score": [576.959938, 553.561438],
        }
    )

    if self_party == 'alice':
        pd.DataFrame(input_data).to_csv(storage.get_writer(input_path), index=False)

    param = NodeEvalParam(
        domain="postprocessing",
        name="score_card_transformer",
        version="1.0.0",
        attr_paths=[
            "positive",
            "predict_score_name",
            "scaled_value",
            "odd_base",
            "pdo",
            "input/input_ds/predict_name",
        ],
        attrs=[
            Attribute(i64=1),
            Attribute(s="predict_score"),
            Attribute(i64=600),
            Attribute(f=20),
            Attribute(f=20),
            Attribute(ss=["pred"]),
        ],
        inputs=[
            DistData(
                name="input_ds",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=input_path, party="alice", format="csv"),
                ],
            )
        ],
        output_uris=[output_path],
    )

    meta = IndividualTable(
        schema=TableSchema(
            id_types=None,
            ids=None,
            feature_types=["str", "float64"],
            features=["id", "pred"],
        )
    )
    param.inputs[0].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1
    if self_party == "alice":
        real_result = orc.read_table(storage.get_reader(output_path)).to_pandas()
        logging.warning(f"...result:{self_party}... \n{real_result}\n.....")
        pd.testing.assert_frame_equal(
            expected_result,
            real_result,
            check_dtype=False,
        )


def test_apply_score_card_transformer_rules():
    def score_card_transform(
        pred: np.ndarray,
        odd_base: float,
        scaled_value: float,
        pdo: float,
        max_score: int = 1000,
        min_score: int = 0,
        bad_label_value: int = 1,
    ):
        assert (pred >= 0).all() and (
            pred <= 1
        ).all(), f"pred should in [0, 1], but got max pred {pred.max()} and min pred {pred.min()}"
        factor = pdo / math.log(2)
        offset = scaled_value - factor * math.log(odd_base)
        if bad_label_value == 1:
            score = offset - factor * np.log(pred / (1 - pred))
        else:
            score = offset + factor * np.log(pred / (1 - pred))

        score = np.select(
            [score > max_score, score < min_score],
            [max_score, min_score],
            score,
        )
        return score

    def compare(
        df: pd.DataFrame,
        pdo,
        odds,
        scaled_value,
        min_score,
        max_score,
        positive,
    ):
        table = sc.Table.from_pandas(df)
        output = ScoreCardTransformer.apply(
            table,
            "pred",
            "predict_score",
            scaled_value,
            odds,
            pdo,
            positive,
            min_score,
            max_score,
        )
        out_df = output.to_pandas()
        logging.warning(f"out scard score: positive={positive}\n {out_df}")

        score = score_card_transform(
            df["pred"].to_numpy(),
            odds,
            scaled_value,
            pdo,
            max_score,
            min_score,
            positive,
        )
        predict_score = out_df["predict_score"].to_numpy()
        logging.warning(f"compare score: {predict_score} {score}")
        np.testing.assert_allclose(predict_score, score)

    data = {"pred": [i * 0.1 for i in range(11)]}
    kwargs = {
        "df": pd.DataFrame(data),
        "pdo": 20.0,
        "odds": 20.0,
        "scaled_value": 600,
        "min_score": 0,
        "max_score": 1000,
        "positive": 1,
    }
    compare(**kwargs)
    kwargs["positive"] = 0
    compare(**kwargs)
