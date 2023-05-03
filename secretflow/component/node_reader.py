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

from secretflow.protos.component.comp_def_pb2 import (
    AtomicParameter,
    AtomicParameterDef,
    AtomicParameterType,
    ComponentDef,
    ParameterNodeType,
)
from secretflow.protos.component.node_def_pb2 import NodeDef


def check_allowed_values(value: AtomicParameter, definition: AtomicParameterDef):
    if definition.type == AtomicParameterType.APT_FLOAT:
        if len(definition.allowed_values.fs) == 0:
            return True
        for i in definition.allowed_values.fs:
            if math.isclose(i, value.f):
                return True
        return False
    if definition.type == AtomicParameterType.APT_INT:
        if len(definition.allowed_values.i64s) == 0:
            return True
        return value.i64 in definition.allowed_values.i64s
    if definition.type == AtomicParameterType.APT_STRING:
        if len(definition.allowed_values.ss) == 0:
            return True
        return value.s in definition.allowed_values.ss
    return True


def check_lower_bound(value: AtomicParameter, definition: AtomicParameterDef):
    if not definition.has_lower_bound:
        return True
    if definition.type == AtomicParameterType.APT_FLOAT:
        return value.f > definition.lower_bound.f or (
            definition.lower_bound_inclusive
            and math.isclose(value.f, definition.lower_bound.f)
        )
    if definition.type == AtomicParameterType.APT_INT:
        return value.i64 > definition.lower_bound.i64 or (
            definition.lower_bound_inclusive and value.i64 == definition.lower_bound.i64
        )
    return True


def check_upper_bound(value: AtomicParameter, definition: AtomicParameterDef):
    if not definition.has_upper_bound:
        return True
    if definition.type == AtomicParameterType.APT_FLOAT:
        return value.f < definition.upper_bound.f or (
            definition.upper_bound_inclusive
            and math.isclose(value.f, definition.upper_bound.f)
        )
    if definition.type == AtomicParameterType.APT_INT:
        return value.i64 < definition.upper_bound.i64 or (
            definition.upper_bound_inclusive and value.i64 == definition.upper_bound.i64
        )
    return True


def get_value(value: AtomicParameter, definition: AtomicParameterDef):
    assert definition.type != AtomicParameterType.APT_UNDEFINED

    if definition.type == AtomicParameterType.APT_FLOAT:
        return value.f
    if definition.type == AtomicParameterType.APT_INT:
        return value.i64
    if definition.type == AtomicParameterType.APT_STRING:
        return value.s
    if definition.type == AtomicParameterType.APT_BOOL:
        return value.b
    if definition.type == AtomicParameterType.APT_FLOATS:
        return value.fs
    if definition.type == AtomicParameterType.APT_INTS:
        return value.i64s
    if definition.type == AtomicParameterType.APT_STRINGS:
        return value.ss
    if definition.type == AtomicParameterType.APT_BOOLS:
        return value.bs


class NodeReader:
    def __init__(self, instance: NodeDef, definition: ComponentDef) -> None:
        self._instance = instance
        self._definition = definition
        self._preprocess()

    def _preprocess(self):
        assert (
            self._instance.domain == self._definition.domain
        ), 'domain does not match.'

        assert self._instance.name == self._definition.name, 'name does not match.'

        assert (
            self._instance.version == self._definition.version
        ), 'version does not match.'

        # param
        self._instance_params = {}

        for p in self._instance.params:
            prefixes = '/'.join(p.prefixes)
            if prefixes not in self._instance_params:
                self._instance_params[prefixes] = {}
            assert (
                p.name not in self._instance_params[prefixes]
            ), f'parameter[{prefixes} - {p.name}] is duplicate in node def.'

            self._instance_params[prefixes][p.name] = p.atomic

        for p in self._definition.params:
            assert (
                p.type == ParameterNodeType.ATOMIC
            ), 'only support ATOMIC at this moment.'

            prefixes = '/'.join(p.prefixes)
            if prefixes not in self._instance_params:
                self._instance_params[prefixes] = {}

            if p.name not in self._instance_params[prefixes]:
                # use default value.
                assert p.atomic.is_optional, f'{p.name} is not set.'
                self._instance_params[prefixes][p.name] = p.atomic.default_value

            # check allowed value
            assert check_allowed_values(
                self._instance_params[prefixes][p.name], p.atomic
            ), f'check_allowed_values failed.'
            assert check_lower_bound(
                self._instance_params[prefixes][p.name], p.atomic
            ), f'check_lower_bound failed.'
            assert check_upper_bound(
                self._instance_params[prefixes][p.name], p.atomic
            ), f'check_upper_bound failed.'

            self._instance_params[prefixes][p.name] = get_value(
                self._instance_params[prefixes][p.name], p.atomic
            )

        # input
        self._instance_input = {}
        for input in self._instance.inputs:
            if input.name in self._instance_input:
                raise RuntimeError(f'input {input.name} is duplicate.')
            self._instance_input[input.name] = input

        for input in self._definition.inputs:
            if input.name not in self._instance_input:
                raise RuntimeError(f'input {input.name} is not set.')

            if self._instance_input[input.name].type not in [
                d.type for d in input.data
            ]:
                raise RuntimeError(f'type of input {input.name} is wrong.')

        # output
        self._instance_output = {}
        for output in self._instance.outputs:
            if output.name in self._instance_output:
                raise RuntimeError(f'output {output.name} is duplicate.')
            self._instance_output[output.name] = output

        for output in self._definition.outputs:
            if output.name not in self._instance_output:
                raise RuntimeError(f'output {output.name} is not set.')

            if self._instance_output[output.name].type not in [
                d.type for d in output.data
            ]:
                raise RuntimeError(f'type of output {output.name} is wrong.')

    def get_param(self, prefixes: str, name: str):
        return self._instance_params[prefixes][name]

    def get_input(self, name: str):
        return self._instance_input[name]

    def get_output(self, name: str):
        return self._instance_output[name]
