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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.∏
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
from google.protobuf.json_format import MessageToJson, Parse

import secretflow.compute as sc
from secretflow.component.component import Component, IoType
from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.core.table_utils import (
    float_almost_equal,
    v_preprocessing_transform,
)
from secretflow.spec.extend.case_when_rules_pb2 import CaseWhenRule

case_when = Component(
    "case_when",
    domain="preprocessing",
    version="0.0.1",
    desc="case_when",
)

case_when.custom_pb_attr(
    name="rules",
    desc="input CaseWhen rules",
    pb_cls=CaseWhenRule,
)


case_when.io(
    io_type=IoType.INPUT,
    name="input_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
)

case_when.io(
    io_type=IoType.OUTPUT,
    name="output_dataset",
    desc="output_dataset",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)

case_when.io(
    io_type=IoType.OUTPUT,
    name="out_rules",
    desc="case when substitution rule",
    types=[DistDataType.PREPROCESSING_RULE],
    col_params=None,
)


def get_rule_features(rules: CaseWhenRule) -> List[str]:
    fs = []

    def _get_value_f(value: CaseWhenRule.ValueExpr):
        assert value.type != CaseWhenRule.ValueExpr.ValueType.INVAL
        if value.type == CaseWhenRule.ValueExpr.ValueType.COLUMN:
            fs.append(value.column_name)

    for when in rules.whens:
        for cond in when.conds:
            fs.append(cond.cond_column)
            _get_value_f(cond.cond_value)
        _get_value_f(when.then)

    _get_value_f(rules.else_value)

    return fs


def apply_case_when_rule(table: sc.Table, rules: CaseWhenRule) -> sc.Table:
    def _get_value(value: CaseWhenRule.ValueExpr, table: sc.Table):
        assert value.type != CaseWhenRule.ValueExpr.ValueType.INVAL
        if value.type == CaseWhenRule.ValueExpr.ValueType.CONST_INT:
            return value.i
        if value.type == CaseWhenRule.ValueExpr.ValueType.CONST_FLOAT:
            return value.f
        if value.type == CaseWhenRule.ValueExpr.ValueType.CONST_STR:
            return value.s
        if value.type == CaseWhenRule.ValueExpr.ValueType.COLUMN:
            return table.column(value.column_name)

    def _is_floating(value):
        if isinstance(value, sc.Array) and pa.types.is_floating(value.dtype):
            return True
        if isinstance(value, float) or isinstance(value, np.floating):
            return True
        return False

    def _apply_cond(cond: CaseWhenRule.Cond, table: sc.Table):
        assert cond.op != CaseWhenRule.Cond.CondOp.INVAL
        op1 = table.column(cond.cond_column)
        op2 = _get_value(cond.cond_value, table)
        if cond.op == CaseWhenRule.Cond.CondOp.EQ:
            if _is_floating(op2):
                return float_almost_equal(op1, op2, rules.float_epsilon)
            else:
                return sc.equal(op1, op2)
        elif cond.op == CaseWhenRule.Cond.CondOp.NE:
            return sc.not_equal(op1, op2)
        elif cond.op == CaseWhenRule.Cond.CondOp.LT:
            return sc.less(op1, op2)
        elif cond.op == CaseWhenRule.Cond.CondOp.LE:
            return sc.less_equal(op1, op2)
        elif cond.op == CaseWhenRule.Cond.CondOp.GT:
            return sc.greater(op1, op2)
        elif cond.op == CaseWhenRule.Cond.CondOp.GE:
            return sc.greater_equal(op1, op2)
        else:
            raise AttributeError(f"unknown cond.op {cond.op}")

    def _apply_when(when: CaseWhenRule.When, table: sc.Table):
        assert len(when.connections) == len(when.conds) - 1
        assert len(when.conds) > 0
        cond_stack = []
        cond_stack.append(_apply_cond(when.conds[0], table))
        for idx, conn in enumerate(when.connections):
            assert conn != CaseWhenRule.When.ConnectType.INVAL
            assert len(cond_stack) == 1 or len(cond_stack) == 2
            if conn == CaseWhenRule.When.ConnectType.AND:
                cond1 = cond_stack.pop()
                cond2 = _apply_cond(when.conds[idx + 1], table)
                cond_stack.append(sc.and_(cond1, cond2))
            if conn == CaseWhenRule.When.ConnectType.OR:
                if len(cond_stack) == 2:
                    cond1 = cond_stack.pop()
                    cond2 = cond_stack.pop()
                    cond_stack.append(sc.or_(cond1, cond2))
                cond_stack.append(_apply_cond(when.conds[idx + 1], table))

        assert len(cond_stack) == 1 or len(cond_stack) == 2
        if len(cond_stack) == 2:
            ret = sc.or_(cond_stack.pop(), cond_stack.pop())
        else:
            ret = cond_stack.pop()
        return ret, _get_value(when.then, table)

    conds = []
    cases = []
    for when in rules.whens:
        cond, case = _apply_when(when, table)
        conds.append(cond)
        cases.append(case)
    cases.append(_get_value(rules.else_value, table))
    conds = sc.make_struct(*conds)
    new_col = sc.case_when(conds, *cases)
    if rules.output_column in table.column_names:
        n = rules.output_column
        table = table.set_column(table.column_names.index(n), n, new_col)
    else:
        table = table.append_column(rules.output_column, new_col)

    return table


@case_when.eval_fn
def case_when_eval_fn(
    *,
    ctx,
    rules: CaseWhenRule,
    input_dataset,
    out_rules,
    output_dataset,
):
    assert (
        input_dataset.type == DistDataType.VERTICAL_TABLE
    ), "only support vtable for now"

    assert len(rules.output_column), "output_column can not be empty"
    assert (
        rules.float_epsilon > 0 and rules.float_epsilon < 1
    ), f"float_epsilon range (0, 1), got {rules.float_epsilon}"

    rule_features = get_rule_features(rules)

    str_rule = MessageToJson(rules)

    def _transform(data: pd.DataFrame):
        import secretflow.spec.extend.case_when_rules_pb2 as pb

        rules = Parse(str_rule, pb.CaseWhenRule())
        data = apply_case_when_rule(sc.Table.from_pandas(data), rules)
        rules_cols = [rules.output_column] if rules.as_label else []
        return data, rules_cols, None

    (output_dd, model_dd, _) = v_preprocessing_transform(
        ctx,
        input_dataset,
        rule_features,
        _transform,
        output_dataset,
        out_rules,
        "Case When",
    )

    return {"out_rules": model_dd, "output_dataset": output_dd}
