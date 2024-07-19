# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pandas as pd

import secretflow.compute as sc
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, dump_table, load_table
from secretflow.data.core import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.driver import reveal
from secretflow.spec.v1.data_pb2 import IndividualTable

SCORE_CARD_TRANSFORMER_VERSION = "0.0.1"

score_card_transformer_comp = Component(
    "score_card_transformer",
    domain="postprocessing",
    version=SCORE_CARD_TRANSFORMER_VERSION,
    desc="Transform the predicted result (a probability value) produced by the logistic regression model into a more understandable score (for example, a score of up to 1000 points)",
)

score_card_transformer_comp.int_attr(
    name="positive",
    desc="Value for positive cases.",
    is_list=False,
    is_optional=False,
    default_value=1,
    allowed_values=[0, 1],
)

score_card_transformer_comp.str_attr(
    name="predict_score_name",
    desc="",
    is_list=False,
    is_optional=False,
    default_value="predict_score",
)

score_card_transformer_comp.int_attr(
    name="scaled_value",
    desc="Set a benchmark score that can be adjusted for specific business scenarios",
    is_list=False,
    is_optional=False,
    default_value=600,
)

score_card_transformer_comp.float_attr(
    name="odd_base",
    desc="the odds value at given score baseline, odds = p / (1-p)",
    is_list=False,
    is_optional=False,
    default_value=20,
)

score_card_transformer_comp.float_attr(
    name="pdo",
    desc="points to double the odds",
    is_list=False,
    is_optional=False,
    default_value=20,
)

score_card_transformer_comp.int_attr(
    name="min_score",
    desc="An integer of [0,999] is supported",
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=0,
    upper_bound=999,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)

score_card_transformer_comp.int_attr(
    name="max_score",
    desc="An integer of [1,1000] is supported",
    is_list=False,
    is_optional=True,
    default_value=1000,
    lower_bound=1,
    upper_bound=1000,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)


score_card_transformer_comp.io(
    io_type=IoType.INPUT,
    name="input_ds",
    desc="predict result table",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="predict_name",
            desc="",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        )
    ],
)

score_card_transformer_comp.io(
    io_type=IoType.OUTPUT,
    name="output_ds",
    desc="output table",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)


def apply_score_card_transformer_on_table(
    table: sc.Table,
    predict_name: str,
    predict_score_name: str,
    scaled_value: int,
    odd_base: float,
    pdo: float,
    positive: int,
    min_score: int = 0,
    max_score: int = 1000,
) -> sc.Table:
    scaled_value = float(scaled_value)
    min_score = float(min_score)
    max_score = float(max_score)
    factor = pdo / math.log(2)
    offset = scaled_value - factor * math.log(odd_base)

    pred = table.column(predict_name)
    log_odds = sc.ln(sc.divide(pred, sc.subtract(1.0, pred)))
    if positive == 1:
        score = sc.subtract(offset, sc.multiply(factor, log_odds))
    else:
        score = sc.add(offset, sc.multiply(factor, log_odds))
    new_col = sc.if_else(
        sc.less_equal(score, min_score),
        min_score,
        sc.if_else(sc.greater_equal(score, max_score), max_score, score),
    )
    if predict_score_name == predict_name:
        table = table.set_column(
            table.column_names.index(predict_score_name), predict_score_name, new_col
        )
    else:
        table = table.append_column(predict_score_name, new_col)

    return table


@score_card_transformer_comp.eval_fn
def score_card_transformer_eval_fn(
    *,
    ctx,
    predict_score_name,
    positive,
    scaled_value,
    odd_base,
    pdo,
    min_score,
    max_score,
    input_ds,
    input_ds_predict_name,
    output_ds,
):
    predict_name = input_ds_predict_name[0]
    assert (
        predict_score_name != predict_name
    ), f"predict_score_name and predict_name should be different, {predict_score_name} {predict_name}"

    assert odd_base > 0, f"odd_base should be positive, got {odd_base}"
    assert scaled_value > 0, f"scaled_value should be positive, got {scaled_value}"
    assert pdo > 0, f"pdo should be positive, got {pdo}"
    assert (
        max_score >= 0 and max_score > scaled_value
    ), f"max_score should bigger than 0 and scaled_value, got {max_score}"
    assert (
        min_score >= 0 and min_score < scaled_value and scaled_value < max_score
    ), f"min_score should bigger than 0 but less than scaled_value and max_score, got {min_score}"
    assert positive in [0, 1], f"positive should be 0 or 1, got {positive}"

    def _fit_transform(trans_data: pd.DataFrame):
        pred = trans_data[predict_name]
        min_max = pred.aggregate(["min", "max"])
        min_value = min_max["min"]
        max_value = min_max["max"]
        if min_value < 0 or max_value > 1:
            return None, ValueError(
                f"pred should in [0, 1], but got max pred {max_value} and min pred {min_value}"
            )
        input_tbl = sc.Table.from_pandas(trans_data)
        output_tbl = apply_score_card_transformer_on_table(
            input_tbl,
            predict_name,
            predict_score_name,
            scaled_value,
            odd_base,
            pdo,
            positive,
            min_score,
            max_score,
        )

        out_data = output_tbl.to_pandas()

        return out_data, None

    input_tbl = load_table(
        ctx,
        input_ds,
        load_features=True,
        load_labels=True,
        load_ids=True,
    )

    with ctx.tracer.trace_running():
        out_partitions = {}
        for pyu, party in input_tbl.partitions.items():
            (out_data, out_err) = pyu(_fit_transform, num_returns=2)(party.data)
            err = reveal(out_err)
            if err is not None:
                raise err

            out_partitions[pyu] = partition(out_data)

    meta = IndividualTable()
    input_ds.meta.Unpack(meta)
    meta.schema.label_types.append("float")
    meta.schema.labels.append(predict_score_name)

    out_df = VDataFrame(partitions=out_partitions, aligned=input_tbl.aligned)
    out_system_info = input_ds.system_info
    output_dd = dump_table(
        ctx,
        vdata=out_df,
        uri=output_ds,
        meta=meta,
        system_info=out_system_info,
    )

    return {"output_ds": output_dd}
