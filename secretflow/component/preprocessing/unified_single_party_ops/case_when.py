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


import numpy as np
import pyarrow as pa
from google.protobuf.json_format import MessageToJson, Parse

import secretflow.compute as sc
from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    ServingBuilder,
    VTable,
    VTableField,
    VTableFieldKind,
    float_almost_equal,
    register,
)
from secretflow.error_system.exceptions import (
    EvalParamError,
    SFTrainingHyperparameterError,
)
from secretflow.spec.extend.case_when_rules_pb2 import CaseWhenRule

from ..preprocessing import PreprocessingMixin


def apply_case_when_rule(table: sc.Table, rules: CaseWhenRule) -> sc.Table:  # type: ignore
    def _get_value(value: CaseWhenRule.ValueExpr, table: sc.Table):  # type: ignore
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

    def _apply_cond(cond: CaseWhenRule.Cond, table: sc.Table):  # type: ignore
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

    def _apply_when(when: CaseWhenRule.When, table: sc.Table):  # type: ignore
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
        kind = VTableFieldKind.LABEL if rules.as_label else VTableFieldKind.FEATURE
        field = VTableField.pa_field(rules.output_column, new_col.dtype, kind)
        table = table.append_column(field, new_col)

    return table


@register(domain='preprocessing', version='1.0.0')
class CaseWhen(PreprocessingMixin, Component):
    '''
    case_when
    '''

    rules: CaseWhenRule = Field.custom_attr(desc="input CaseWhen rules")  # type: ignore
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="output_dataset",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_rule: Output = Field.output(
        desc="case when substitution rule",
        types=[DistDataType.PREPROCESSING_RULE],
    )

    def evaluate(self, ctx: Context):
        if len(self.rules.output_column) == 0:
            raise EvalParamError.missing_or_none_param("output_column can not be empty")
        if not (self.rules.float_epsilon > 0 and self.rules.float_epsilon < 1):
            SFTrainingHyperparameterError.out_of_range(
                f"float_epsilon range (0, 1), got {self.rules.float_epsilon}"
            )

        rule_features = self.get_rule_features()

        str_rule = MessageToJson(self.rules, indent=0)

        def _fit(df: sc.Table) -> sc.Table:
            import secretflow.spec.extend.case_when_rules_pb2 as pb

            rules = Parse(str_rule, pb.CaseWhenRule())
            data = apply_case_when_rule(df, rules)
            return data

        input_tbl = VTable.from_distdata(self.input_ds)
        tran_tbl = input_tbl.select(rule_features)
        rule = self.fit(ctx, self.output_rule, tran_tbl, _fit)
        self.transform(ctx, self.output_ds, input_tbl, rule)

    def get_rule_features(self) -> list[str]:
        fs = []

        def _get_value_f(value: CaseWhenRule.ValueExpr):  # type: ignore
            assert value.type != CaseWhenRule.ValueExpr.ValueType.INVAL
            if value.type == CaseWhenRule.ValueExpr.ValueType.COLUMN:
                fs.append(value.column_name)

        for when in self.rules.whens:
            for cond in when.conds:
                fs.append(cond.cond_column)
                _get_value_f(cond.cond_value)
            _get_value_f(when.then)

        _get_value_f(self.rules.else_value)

        fs_map = {n: idx for idx, n in enumerate(fs)}
        return list(fs_map.keys())

    def export(self, ctx: Context, builder: ServingBuilder) -> None:
        self.do_export(ctx, builder, self.input_ds, self.output_rule.data)
