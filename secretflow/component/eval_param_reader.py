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

import importlib
import math

from google.protobuf import json_format

from secretflow.spec.v1.component_pb2 import (
    Attribute,
    AttributeDef,
    AttrType,
    ComponentDef,
    IoDef,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


class EvalParamError(Exception):
    ...


def check_allowed_values(value: Attribute, definition: AttributeDef):
    if definition.type == AttrType.AT_FLOAT:
        if len(definition.atomic.allowed_values.fs) == 0:
            return True
        for i in definition.atomic.allowed_values.fs:
            if math.isclose(i, value.f):
                return True
        return False
    if definition.type == AttrType.AT_INT:
        if len(definition.atomic.allowed_values.i64s) == 0:
            return True
        return value.i64 in definition.atomic.allowed_values.i64s
    if definition.type == AttrType.AT_STRING:
        if len(definition.atomic.allowed_values.ss) == 0:
            return True
        return value.s in definition.atomic.allowed_values.ss
    return True


def check_lower_bound(value: Attribute, definition: AttributeDef):
    if not definition.atomic.lower_bound_enabled:
        return True
    if definition.type == AttrType.AT_FLOAT:
        return value.f > definition.atomic.lower_bound.f or (
            definition.atomic.lower_bound_inclusive
            and math.isclose(value.f, definition.atomic.lower_bound.f)
        )
    if definition.type == AttrType.AT_INT:
        return value.i64 > definition.atomic.lower_bound.i64 or (
            definition.atomic.lower_bound_inclusive
            and value.i64 == definition.atomic.lower_bound.i64
        )
    return True


def check_upper_bound(value: Attribute, definition: AttributeDef):
    if not definition.atomic.upper_bound_enabled:
        return True
    if definition.type == AttrType.AT_FLOAT:
        return value.f < definition.atomic.upper_bound.f or (
            definition.atomic.upper_bound_inclusive
            and math.isclose(value.f, definition.atomic.upper_bound.f)
        )
    if definition.type == AttrType.AT_INT:
        return value.i64 < definition.atomic.upper_bound.i64 or (
            definition.atomic.upper_bound_inclusive
            and value.i64 == definition.atomic.upper_bound.i64
        )
    return True


def check_table_attr_col_cnt(value: Attribute, definition: IoDef.TableAttrDef):
    cnt = len(value.ss)

    if definition.col_min_cnt_inclusive and cnt < definition.col_min_cnt_inclusive:
        return False

    if definition.col_max_cnt_inclusive and cnt > definition.col_max_cnt_inclusive:
        return False

    return True


def get_value(value: Attribute, at: AttrType, pb_cls_name: str = None):
    if at == AttrType.ATTR_TYPE_UNSPECIFIED:
        raise EvalParamError("Type of Attribute is undefined.")
    elif at == AttrType.AT_FLOAT:
        return value.f
    elif at == AttrType.AT_INT:
        return value.i64
    elif at == AttrType.AT_STRING:
        return value.s
    elif at == AttrType.AT_BOOL:
        return value.b
    elif at == AttrType.AT_FLOATS:
        return list(value.fs)
    elif at == AttrType.AT_INTS:
        return list(value.i64s)
    elif at == AttrType.AT_STRINGS:
        return list(value.ss)
    elif at == AttrType.AT_BOOLS:
        return list(value.bs)
    elif at == AttrType.AT_CUSTOM_PROTOBUF:
        pb_cls = importlib.import_module("secretflow.spec.extend")
        for name in pb_cls_name.split("."):
            pb_cls = getattr(pb_cls, name)
        return json_format.Parse(value.s, pb_cls())
    else:
        raise EvalParamError(f"unsupported type: {at}.")


class EvalParamReader:
    def __init__(self, instance: NodeEvalParam, definition: ComponentDef) -> None:
        self._instance = instance
        self._definition = definition
        self._preprocess()

    def _preprocess(self):
        if self._instance.domain != self._definition.domain:
            raise EvalParamError(
                f"domain inst:'{self._instance.domain}' def:'{self._definition.domain}'  does not match."
            )

        if self._instance.name != self._definition.name:
            raise EvalParamError(
                f"name inst:'{self._instance.name}' def:'{self._definition.name}' does not match."
            )

        if self._instance.version != self._definition.version:
            raise EvalParamError(
                f"version inst:'{self._instance.version}' def:'{self._definition.version}' does not match."
            )

        # attrs
        self._instance_attrs = {}
        for path, attr in zip(
            list(self._instance.attr_paths), list(self._instance.attrs)
        ):
            if path in self._instance_attrs:
                raise EvalParamError(f"attr {path} is duplicate in node def.")

            if not attr.is_na:
                self._instance_attrs[path] = attr

        for attr in self._definition.attrs:
            if attr.type not in [
                AttrType.AT_FLOAT,
                AttrType.AT_FLOATS,
                AttrType.AT_INT,
                AttrType.AT_INTS,
                AttrType.AT_STRING,
                AttrType.AT_STRINGS,
                AttrType.AT_BOOL,
                AttrType.AT_BOOLS,
                AttrType.AT_CUSTOM_PROTOBUF,
            ]:
                raise EvalParamError(
                    "only support ATOMIC and CUSTOM_PROTOBUF at this moment."
                )

            full_name = "/".join(list(attr.prefixes) + [attr.name])

            if full_name not in self._instance_attrs:
                if attr.type is AttrType.AT_CUSTOM_PROTOBUF:
                    raise EvalParamError(f"CUSTOM_PROTOBUF attr {full_name} not set.")

                # use default value.
                if not attr.atomic.is_optional:
                    raise EvalParamError(
                        f"attr {full_name} is not optional and not set."
                    )
                self._instance_attrs[full_name] = attr.atomic.default_value

            # check allowed value
            if not check_allowed_values(self._instance_attrs[full_name], attr):
                raise EvalParamError(f"attr {full_name}: check_allowed_values failed.")
            if not check_lower_bound(self._instance_attrs[full_name], attr):
                raise EvalParamError(f"attr {full_name}: check_lower_bound failed.")
            if not check_upper_bound(self._instance_attrs[full_name], attr):
                raise EvalParamError(f"attr {full_name}: check_upper_bound failed.")

            self._instance_attrs[full_name] = get_value(
                self._instance_attrs[full_name], attr.type, attr.custom_protobuf_cls
            )

        # input
        if len(self._instance.inputs) != len(self._definition.inputs):
            raise EvalParamError("number of input does not match.")

        self._instance_inputs = {}
        for input_instance, input_def in zip(
            list(self._instance.inputs), list(self._definition.inputs)
        ):
            if input_def.name in self._instance_inputs:
                raise EvalParamError(f"input {input_def.name} is duplicate.")

            if input_def.types and input_instance.type not in input_def.types:
                raise EvalParamError(
                    f"type of input {input_def.name} is wrong, got {input_instance.type }, expect {input_def.types}"
                )

            self._instance_inputs[input_def.name] = input_instance

            for input_attr in input_def.attrs:
                if len(input_attr.extra_attrs):
                    raise EvalParamError(
                        "extra attribute is unsupported at this moment."
                    )

                full_name = "/".join(["input", input_def.name, input_attr.name])

                if full_name not in self._instance_attrs:
                    self._instance_attrs[full_name] = Attribute()

                if not check_table_attr_col_cnt(
                    self._instance_attrs[full_name], input_attr
                ):
                    raise EvalParamError(
                        f"input attr {full_name} check_table_attr_col_cnt fails."
                    )

                self._instance_attrs[full_name] = get_value(
                    self._instance_attrs[full_name], AttrType.AT_STRINGS
                )

        # output
        if len(self._instance.output_uris) != len(self._definition.outputs):
            raise EvalParamError("number of output does not match.")

        self._instance_outputs = {}
        for output_prefix, output_def in zip(
            list(self._instance.output_uris), list(self._definition.outputs)
        ):
            if output_def.name in self._instance_outputs:
                raise EvalParamError(f"output {output_def.name} is duplicate.")
            self._instance_outputs[output_def.name] = output_prefix

    def get_attr(self, name: str):
        if name not in self._instance_attrs:
            raise EvalParamError(f"attr {name} does not exist.")
        return self._instance_attrs[name]

    def get_input(self, name: str):
        if name not in self._instance_inputs:
            raise EvalParamError(f"input {name} does not exist.")
        return self._instance_inputs[name]

    def get_input_attrs(self, input_name: str, attr_name: str):
        full_name = "/".join(["input", input_name, attr_name])
        if full_name not in self._instance_attrs:
            raise EvalParamError(f"input attr {full_name} does not exist.")

        return self._instance_attrs[full_name]

    def get_output_uri(self, name: str):
        if name not in self._instance_outputs:
            raise EvalParamError(f"output {name} does not exist.")
        return self._instance_outputs[name]
