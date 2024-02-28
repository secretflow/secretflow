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

import pandas as pd
import pyarrow as pa
from google.protobuf.json_format import MessageToJson, Parse

import secretflow.compute as sc
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.core.table_utils import (
    v_preprocessing_transform,
)
from secretflow.spec.extend.calculate_rules_pb2 import CalculateOpRules

feature_calculate = Component(
    "feature_calculate",
    domain="preprocessing",
    version="0.0.1",
    desc="Generate a new feature by performing calculations on an origin feature",
)

feature_calculate.custom_pb_attr(
    name="rules",
    desc="input CalculateOpRules rules",
    pb_cls=CalculateOpRules,
)

feature_calculate.io(
    io_type=IoType.INPUT,
    name="in_ds",
    desc="Input vertical table",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="features",
            desc="Feature(s) to operate on",
            col_min_cnt_inclusive=1,
        )
    ],
)

feature_calculate.io(
    io_type=IoType.OUTPUT,
    name="out_ds",
    desc="output_dataset",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)

feature_calculate.io(
    io_type=IoType.OUTPUT,
    name="out_rules",
    desc="feature calculate rule",
    types=[DistDataType.PREPROCESSING_RULE],
    col_params=None,
)


def apply_feature_calcute_rule(
    table: sc.Table, rules: CalculateOpRules, in_ds_features
) -> sc.Table:
    def _check_numuric(type):
        assert pa.types.is_floating(type) or pa.types.is_integer(
            type
        ), f"operator only support float/int, but got {type}"

    def _check_text(type):
        assert pa.types.is_string(type), f"operator only support string, but got {type}"

    # std = (x-mean)/stde
    def _apply_standardize(col: sc.Array):
        _check_numuric(col.dtype)
        pd_col = col.to_pandas()
        # const column, set column elements to 0s
        if pd_col.nunique() == 1:
            new_col = sc.multiply(col, 0)
        else:
            mean = pd_col.mean()
            stde = pd_col.std(ddof=0)
            new_col = sc.divide(sc.subtract(col, mean), stde)
        return new_col

    # norm = (x-min)/(max-min)
    def _apply_normalize(col: sc.Array):
        _check_numuric(col.dtype)
        pd_col = col.to_pandas()
        # const column, set column elements to 0s
        if pd_col.nunique() == 1:
            new_col = sc.multiply(col, 0)
        else:
            max = pd_col.max()
            min = pd_col.min()
            new_col = sc.divide(sc.subtract(col, min), float(max - min))
        return new_col

    def _apply_range_limit(col: sc.Array):
        _check_numuric(col.dtype)
        op_cnt = len(rules.operands)
        assert op_cnt == 2, f"range limit operator need 2 operands, but got {op_cnt}"
        op0 = float(rules.operands[0])
        op1 = float(rules.operands[1])
        assert (
            op0 <= op1
        ), f"range limit operator expect min <= max, but get [{op0}, {op1}]"

        conds = [sc.less(col, op0), sc.greater(col, op1)]
        cases = [op0, op1, col]
        new_col = sc.case_when(sc.make_struct(*conds), *cases)
        return new_col

    def _apply_unary(col: sc.Array):
        _check_numuric(col.dtype)
        op_cnt = len(rules.operands)
        assert op_cnt == 3, f"unary operator needs 3 operands, but got {op_cnt}"
        op0 = rules.operands[0]
        assert op0 in ['+', '-'], f"unary op0 should be [+ - r], but get {op0}"
        op1 = rules.operands[1]
        assert op1 in [
            '+',
            '-',
            '*',
            '/',
        ], f"unary op1 should be [+ - * /], but get {op1}"
        op3 = float(rules.operands[2])
        if op1 == "+":
            new_col = sc.add(col, op3)
        elif op1 == "-":
            new_col = sc.subtract(col, op3) if op0 == "+" else sc.subtract(op3, col)
        elif op1 == "*":
            new_col = sc.multiply(col, op3)
        elif op1 == "/":
            if op0 == "+":
                assert op3 != 0, "unary operator divide zero"
                new_col = sc.divide(col, op3)
            else:
                new_col = sc.divide(op3, col)
        return new_col

    def _apply_reciprocal(col: sc.Array):
        _check_numuric(col.dtype)
        new_col = sc.divide(1.0, col)
        return new_col

    def _apply_round(col: sc.Array):
        _check_numuric(col.dtype)
        new_col = sc.round(col)
        return new_col

    def _apply_log_round(col: sc.Array):
        _check_numuric(col.dtype)
        op_cnt = len(rules.operands)
        assert op_cnt == 1, f"log operator needs 1 operands, but got {op_cnt}"
        op0 = float(rules.operands[0])
        new_col = sc.round(sc.log2(sc.add(col, op0)))
        return new_col

    def _apply_sqrt(col: sc.Array):
        _check_numuric(col.dtype)
        # TODO: whether check positive? sqrt will return a NaN when meets negative argument
        new_col = sc.sqrt(col)
        return new_col

    def _apply_log(col: sc.Array):
        _check_numuric(col.dtype)
        op_cnt = len(rules.operands)
        assert op_cnt == 2, f"log operator needs 2 operands, but got {op_cnt}"
        op0 = rules.operands[0]
        op1 = float(rules.operands[1])
        if op0 == "e":
            new_col = sc.multiply(sc.log2(sc.add(col, op1)), math.log(2, math.e))
        else:
            new_col = sc.logb(sc.add(col, op1), float(op0))
        return new_col

    def _apply_exp(col: sc.Array):
        _check_numuric(col.dtype)
        new_col = sc.exp(col)
        return new_col

    def _apply_lenth(col: sc.Array):
        _check_text(col.dtype)
        new_col = sc.utf8_length(col)
        return new_col

    def _apply_substr(col: sc.Array):
        _check_text(col.dtype)
        op_cnt = len(rules.operands)
        assert op_cnt == 2, f"substr operator need 2 oprands, but get {op_cnt}"
        start = int(rules.operands[0])
        lenth = int(rules.operands[1])
        new_col = sc.utf8_slice_codeunits(col, start, start + lenth)
        return new_col

    for feature in in_ds_features:
        if feature in table.column_names:
            col = table.column(feature)
            if rules.op == CalculateOpRules.OpType.STANDARDIZE:
                new_col = _apply_standardize(col)
            elif rules.op == CalculateOpRules.OpType.NORMALIZATION:
                new_col = _apply_normalize(col)
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
                new_col = _apply_lenth(col)
            elif rules.op == CalculateOpRules.OpType.SUBSTR:
                new_col = _apply_substr(col)
            else:
                raise AttributeError(f"unknown rules.op {rules.op}")
            table = table.set_column(
                table.column_names.index(feature),
                feature,
                new_col,
            )
    return table


@feature_calculate.eval_fn
def feature_calculate_eval_fn(
    *,
    ctx,
    rules: CalculateOpRules,
    in_ds,
    in_ds_features,
    out_rules,
    out_ds,
):
    assert in_ds.type == DistDataType.VERTICAL_TABLE, "only support vtable for now"
    str_rule = MessageToJson(rules)

    def _transform(data: pd.DataFrame):
        import secretflow.spec.extend.calculate_rules_pb2 as pb

        rules = Parse(str_rule, pb.CalculateOpRules())
        data = apply_feature_calcute_rule(
            sc.Table.from_pandas(data), rules, in_ds_features
        )
        return data, [], None

    (out_ds, model_dd, _) = v_preprocessing_transform(
        ctx,
        in_ds,
        in_ds_features,
        _transform,
        out_ds,
        out_rules,
        "Feature Calculate",
        assert_one_party=False,
    )

    return {"out_rules": model_dd, "out_ds": out_ds}
