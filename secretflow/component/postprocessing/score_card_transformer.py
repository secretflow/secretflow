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
from typing import Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from google.protobuf import json_format

import secretflow.compute as sc
from secretflow.component.core import (
    Component,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    IServingExporter,
    Output,
    ServingBuilder,
    ServingNode,
    ServingOp,
    ServingPhase,
    VTableFieldKind,
    VTableUtils,
    register,
)


@register(domain="postprocessing", version="1.0.0")
class ScoreCardTransformer(Component, IServingExporter):
    '''
    Transform the predicted result (a probability value) produced by the logistic regression model into a more understandable score (for example, a score of up to 1000 points)
    '''

    positive: int = Field.attr(
        desc="Value for positive cases.",
        is_optional=False,
        default=1,
        choices=[0, 1],
    )
    predict_score_name: str = Field.attr(
        desc="",
        is_optional=False,
        default="predict_score",
    )
    scaled_value: int = Field.attr(
        desc="Set a benchmark score that can be adjusted for specific business scenarios",
        is_optional=False,
        default=600,
        bound_limit=Interval.open(0, None),
    )
    odd_base: float = Field.attr(
        desc="the odds value at given score baseline, odds = p / (1-p)",
        is_optional=False,
        default=20,
        bound_limit=Interval.open(0, None),
    )
    pdo: float = Field.attr(
        desc="points to double the odds",
        is_optional=False,
        default=20,
        bound_limit=Interval.open(0, None),
    )
    min_score: int = Field.attr(
        desc="An integer of [0,999] is supported",
        is_optional=True,
        default=0,
        bound_limit=Interval.closed(0, 999),
    )
    max_score: int = Field.attr(
        desc="An integer of [1,1000] is supported",
        is_optional=True,
        default=1000,
        bound_limit=Interval.closed(1, 1000),
    )
    predict_name: str = Field.table_column_attr(
        input_name="input_ds",
    )
    input_ds: Input = Field.input(
        desc="predict result table",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="output table",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )

    def to_rule(self) -> dict:
        return {
            "positive": self.positive,
            "pdo": self.pdo,
            "odd_base": self.odd_base,
            "scaled_value": self.scaled_value,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "predict_name": self.predict_name,
            "predict_score_name": self.predict_score_name,
        }

    @staticmethod
    def apply(
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
            index = table.column_names.index(predict_score_name)
            table = table.set_column(index, predict_name, new_col)
        else:
            new_field = VTableUtils.pa_field(
                predict_score_name, new_col.dtype, VTableFieldKind.LABEL
            )
            table = table.append_column(new_field, new_col)

        return table

    def evaluate(self, ctx: Context):
        assert (
            self.predict_score_name != self.predict_name
        ), f"predict_score_name and predict_name should be different, {self.predict_score_name} {self.predict_name}"

        assert (
            self.scaled_value < self.max_score
        ), f"scaled_value<{self.scaled_value}> should be less than max_score<{self.max_score}>"

        assert (
            self.scaled_value > self.min_score
        ), f"scaled_value<{self.scaled_value}> should be bigger than min_score<{self.min_score}>"

        rule = self.to_rule()

        def _fit_transform(input_tbl: pa.Table) -> pa.Table:
            pred = input_tbl.column(self.predict_name)
            min_value = pc.min(pred).as_py()
            max_value = pc.max(pred).as_py()
            if min_value < 0 or max_value > 1:
                raise ValueError(
                    f"pred should in [0, 1], but got max pred {max_value} and min pred {min_value}"
                )
            output_tbl = ScoreCardTransformer.apply(
                sc.Table.from_pyarrow(input_tbl), **rule
            )
            return output_tbl.to_table()

        reader = CompVDataFrameReader(ctx.storage, ctx.tracer, self.input_ds)
        writer = CompVDataFrameWriter(ctx.storage, ctx.tracer, self.output_ds.uri)
        with reader, writer:
            for df in reader:
                with ctx.tracer.trace_running():
                    out_df = df.apply(_fit_transform)
                writer.write(out_df)

        writer.dump_to(self.output_ds)

    def export(self, ctx: Context, builder: ServingBuilder):
        def _dump_runner(
            rules: dict, node_name: str
        ) -> Tuple[bytes, pa.Schema, pa.Schema, bytes, bytes]:
            input_schema = {"pred_y": np.float64}
            input_table = sc.Table.from_schema(input_schema)
            table = ScoreCardTransformer.apply(input_table, **rules)

            dag_pb, dag_input_schema, dag_output_schema = table.dump_serving_pb(
                node_name
            )

            dag_json = json_format.MessageToJson(dag_pb, indent=0).encode("utf-8")
            dag_input_schema_ser = dag_input_schema.serialize().to_pybytes()
            dag_output_schema_ser = dag_output_schema.serialize().to_pybytes()

            return (
                dag_json,
                dag_input_schema,
                dag_output_schema,
                dag_input_schema_ser,
                dag_output_schema_ser,
            )

        rules = self.to_rule()
        rules["predict_score_name"] = "pred_y"
        rules["predict_name"] = "pred_y"

        node_name = f"score_card_transformer_{builder.max_id()}"
        node = ServingNode(
            node_name,
            op=ServingOp.ARROW_PROCESSING,
            phase=ServingPhase.POSTPROCESSING,
        )

        for pyu in builder.pyus:
            dag, in_schema, out_schema, in_schema_ser, out_schema_ser = pyu(
                _dump_runner
            )(rules, node_name)

            kwargs = ServingNode.build_arrow_processing_kwargs(
                in_schema_ser, out_schema_ser, dag
            )
            node.add(pyu, in_schema, out_schema, kwargs)

        builder.add_node(node)
