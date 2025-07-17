# Copyright 2023 Ant Group Co., Ltd.
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
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc
from google.protobuf.json_format import MessageToJson, Parse

import secretflow.compute as sc
from secretflow.component.core import (
    Component,
    CompVDataFrame,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    IServingExporter,
    Output,
    ServingBuilder,
    VTable,
    register,
)
from secretflow.device import PYUObject
from secretflow.spec.extend.calculate_rules_pb2 import CalculateOpRules
from secretflow.utils.errors import InvalidArgumentError, NotSupportedError

from ..preprocessing import PreprocessingMixin


@dataclass
class FeatureInfo:
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    stddev: float | None = None
    nunique: int = 0


def apply_feature_calculate_rule(
    table: sc.Table,
    rules: CalculateOpRules,
    in_ds_features,
    infos: dict[str, FeatureInfo],
) -> sc.Table:
    def _check_numeric(type):
        if not pa.types.is_floating(type) and not pa.types.is_integer(type):
            raise NotSupportedError(f"operator only support float/int, but got {type}")

    def _check_text(type):
        if not pa.types.is_string(type):
            raise NotSupportedError(f"operator only support string, but got {type}")

    # std = (x-mean)/stde
    def _apply_standardize(col: sc.Array, info: FeatureInfo):
        _check_numeric(col.dtype)
        # const column, set column elements to 0s

        if info is None:
            raise InvalidArgumentError("invalid table info")

        if info.nunique == 1:
            new_col = sc.multiply(col, 0)
        else:
            mean = info.mean
            stde = info.stddev
            new_col = sc.divide(sc.subtract(col, mean), stde)
        return new_col

    # norm = (x-min)/(max-min)
    def _apply_normalize(col: sc.Array, info: FeatureInfo):
        _check_numeric(col.dtype)
        if info is None:
            raise InvalidArgumentError("invalid table info")

        # const column, set column elements to 0s
        if info.nunique == 1:
            new_col = sc.multiply(col, 0)
        else:
            max = info.max
            min = info.min
            new_col = sc.divide(sc.subtract(col, min), float(max - min))
        return new_col

    def _apply_range_limit(col: sc.Array):
        _check_numeric(col.dtype)
        op_cnt = len(rules.operands)
        if op_cnt != 2:
            raise InvalidArgumentError(
                f"range limit operator need 2 operands, but got {op_cnt}"
            )
        op0 = float(rules.operands[0])
        op1 = float(rules.operands[1])
        if op0 > op1:
            raise InvalidArgumentError(
                f"range limit operator expect min <= max, but get [{op0}, {op1}]"
            )

        conds = [sc.less(col, op0), sc.greater(col, op1)]
        cases = [op0, op1, col]
        new_col = sc.case_when(sc.make_struct(*conds), *cases)
        return new_col

    def _apply_unary(col: sc.Array):
        _check_numeric(col.dtype)
        op_cnt = len(rules.operands)
        if op_cnt != 3:
            raise InvalidArgumentError(
                f"unary operator needs 3 operands, but got {op_cnt}"
            )
        op0 = rules.operands[0]
        if op0 not in ['+', '-']:
            raise InvalidArgumentError(f"unary op0 should be [+ -], but get {op0}")
        op1 = rules.operands[1]
        if op1 not in ['+', '-', '*', '/']:
            raise InvalidArgumentError(f"unary op1 should be [+ - * /], but get {op1}")
        op3 = float(rules.operands[2])
        if op1 == "+":
            new_col = sc.add(col, op3)
        elif op1 == "-":
            new_col = sc.subtract(col, op3) if op0 == "+" else sc.subtract(op3, col)
        elif op1 == "*":
            new_col = sc.multiply(col, op3)
        elif op1 == "/":
            if op0 == "+":
                if op3 == 0:
                    raise InvalidArgumentError("unary operator divide zero")
                new_col = sc.divide(col, op3)
            else:
                new_col = sc.divide(op3, col)
        return new_col

    def _apply_reciprocal(col: sc.Array):
        _check_numeric(col.dtype)
        new_col = sc.divide(1.0, col)
        return new_col

    def _apply_round(col: sc.Array):
        _check_numeric(col.dtype)
        new_col = sc.round(col)
        return new_col

    def _apply_log_round(col: sc.Array):
        _check_numeric(col.dtype)
        op_cnt = len(rules.operands)
        if op_cnt != 1:
            raise InvalidArgumentError(
                f"log operator needs 1 operands, but got {op_cnt}"
            )
        op0 = float(rules.operands[0])
        new_col = sc.round(sc.log2(sc.add(col, op0)))
        return new_col

    def _apply_sqrt(col: sc.Array):
        _check_numeric(col.dtype)
        # TODO: whether check positive? sqrt will return a NaN when meets negative argument
        new_col = sc.sqrt(col)
        return new_col

    def _apply_log(col: sc.Array):
        _check_numeric(col.dtype)
        op_cnt = len(rules.operands)
        if op_cnt != 2:
            raise InvalidArgumentError(
                f"log operator needs 2 operands, but got {op_cnt}"
            )
        op0 = rules.operands[0]
        op1 = float(rules.operands[1])
        if op0 == "e":
            new_col = sc.multiply(sc.log2(sc.add(col, op1)), math.log(2, math.e))
        else:
            new_col = sc.logb(sc.add(col, op1), float(op0))
        return new_col

    def _apply_exp(col: sc.Array):
        _check_numeric(col.dtype)
        new_col = sc.exp(col)
        return new_col

    def _apply_length(col: sc.Array):
        _check_text(col.dtype)
        new_col = sc.utf8_length(col)
        return new_col

    def _apply_substr(col: sc.Array):
        _check_text(col.dtype)
        op_cnt = len(rules.operands)
        if op_cnt != 2:
            raise InvalidArgumentError(
                f"substr operator needs 2 operands, but got {op_cnt}"
            )
        start = int(rules.operands[0])
        length = int(rules.operands[1])
        new_col = sc.utf8_slice_codeunits(col, start, start + length)
        return new_col

    for feature in in_ds_features:
        if feature not in table.column_names:
            continue
        col = table.column(feature)
        if rules.op == CalculateOpRules.OpType.STANDARDIZE:
            new_col = _apply_standardize(col, infos.get(feature))
        elif rules.op == CalculateOpRules.OpType.NORMALIZATION:
            new_col = _apply_normalize(col, infos.get(feature))
        elif rules.op == CalculateOpRules.OpType.RANGE_LIMIT:
            new_col = _apply_range_limit(col)
        elif rules.op == CalculateOpRules.OpType.UNARY:
            new_col = _apply_unary(col)
        elif rules.op == CalculateOpRules.OpType.RECIPROCAL:
            new_col = _apply_reciprocal(col)
        elif rules.op == CalculateOpRules.OpType.ROUND:
            new_col = _apply_round(col)
        elif rules.op == CalculateOpRules.OpType.LOG_ROUND:
            new_col = _apply_log_round(col)
        elif rules.op == CalculateOpRules.OpType.SQRT:
            new_col = _apply_sqrt(col)
        elif rules.op == CalculateOpRules.OpType.LOG:
            new_col = _apply_log(col)
        elif rules.op == CalculateOpRules.OpType.EXP:
            new_col = _apply_exp(col)
        elif rules.op == CalculateOpRules.OpType.LENGTH:
            new_col = _apply_length(col)
        elif rules.op == CalculateOpRules.OpType.SUBSTR:
            new_col = _apply_substr(col)
        else:
            raise InvalidArgumentError(f"unknown rules.op {rules.op}")
        table = table.set_column(
            table.column_names.index(feature),
            feature,
            new_col,
        )
    return table


@register(domain="preprocessing", version="1.0.0")
class FeatureCalculate(PreprocessingMixin, Component, IServingExporter):
    '''
    Generate a new feature by performing calculations on an origin feature
    '''

    rules: CalculateOpRules = Field.custom_attr(desc="input CalculateOpRules rules")
    features: list[str] = Field.table_column_attr(
        "input_ds",
        desc="Feature(s) to operate on",
        limit=Interval.closed(1, None),
    )
    input_ds: Input = Field.input(
        desc="Input vertical table",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="output_dataset",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_rule: Output = Field.output(
        desc="feature calculate rule",
        types=[DistDataType.PREPROCESSING_RULE],
    )

    def evaluate(self, ctx: Context):
        str_rule = MessageToJson(self.rules)

        def _fit(df: sc.Table, info: FeatureInfo | None) -> sc.Table:
            import secretflow.spec.extend.calculate_rules_pb2 as pb

            rules = Parse(str_rule, pb.CalculateOpRules())
            out_df = apply_feature_calculate_rule(df, rules, self.features, info)
            return out_df

        need_extras = self.rules.op in [
            CalculateOpRules.OpType.STANDARDIZE,
            CalculateOpRules.OpType.NORMALIZATION,
        ]

        input_tbl = VTable.from_distdata(self.input_ds)
        trans_tbl = input_tbl.select(self.features)
        if need_extras:
            df = ctx.load_table(input_tbl)
            extras = self.get_feature_infos(df[self.features])
            input_df = df
        else:
            extras = None
            input_df = input_tbl

        rule = self.fit(ctx, self.output_rule, trans_tbl, _fit, extras)
        self.transform(ctx, self.output_ds, input_df, rule, streaming=not need_extras)

    def get_feature_infos(self, df: CompVDataFrame) -> dict[str, PYUObject]:
        op = self.rules.op

        def _apply(df: pa.Table) -> dict[str, FeatureInfo]:
            res = {}
            for name in df.column_names:
                col = df[name]
                nunique = len(pc.unique(col))
                info = FeatureInfo(nunique=nunique)
                if nunique != 1:
                    if op == CalculateOpRules.OpType.STANDARDIZE:
                        info.mean = pc.mean(col).as_py()
                        info.stddev = pc.stddev(col).as_py()
                    elif op == CalculateOpRules.OpType.NORMALIZATION:
                        info.min = pc.min(col).as_py()
                        info.max = pc.max(col).as_py()
                res[name] = info

            return res

        result: dict[str, PYUObject] = {}
        for pyu, p in df.partitions.items():
            info_obj = pyu(_apply)(p.data)
            result[pyu.party] = info_obj
        return result

    def export(self, ctx: Context, builder: ServingBuilder) -> None:
        self.do_export(ctx, builder, self.input_ds, self.output_rule.data)
