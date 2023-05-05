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
from enum import Enum, unique
from typing import List, Union

from secretflow.component.node_reader import NodeReader
from secretflow.protos.component.comp_def_pb2 import (
    AtomicParameterDef,
    AtomicParameterType,
    ComponentDef,
    IoDef,
    ModelDef,
    ParameterNode,
    ParameterNodeType,
    SFDataType,
    TableDef,
)
from secretflow.protos.component.node_def_pb2 import NodeDef


@unique
class IoType(Enum):
    INPUT = 1
    OUTPUT = 2


@dataclass
class TableColParam:
    name: str
    desc: str
    col_list_min_cnt: int = None
    col_list_max_cnt: int = None


class Component:
    def __init__(self, name: str, domain='', version='', desc='') -> None:
        self.name = name
        self.domain = domain
        self.version = version
        self.desc = desc

        self.__definition = None
        self.__eval_callback = None
        self.__comp_param_decls = []
        self.__input_io_decls = []
        self.__output_io_decls = []
        self.__argnames = set()

    def float_param(
        self,
        name: str,
        desc: str,
        is_list: bool,
        is_optional: bool,
        default_value: Union[List[float], float] = None,
        allowed_values: List[float] = None,
        lower_bound: float = None,
        upper_bound: float = None,
        lower_bound_inclusive: bool = False,
        upper_bound_inclusive: bool = False,
        list_min_lenth_inclusive: int = None,
        list_max_length_inclusive: int = None,
    ):
        # sanity checks
        if not is_optional and default_value is None:
            raise RuntimeError('default_value must be provided if not optional ')

        if allowed_values is not None and (
            lower_bound is not None or upper_bound is not None
        ):
            raise RuntimeError(
                'allowed_values and bounds could not be set at the same time.'
            )

        if allowed_values is not None and default_value is not None:
            if is_list:
                for v in default_value:
                    if v not in allowed_values:
                        raise RuntimeError(
                            f'default_value {v} is not in allowed_values {allowed_values}'
                        )
            else:
                if default_value not in allowed_values:
                    raise RuntimeError(
                        f'default_value {default_value} is not in allowed_values {allowed_values}'
                    )

        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise RuntimeError(
                f'lower_bound {lower_bound} is greater than upper_bound {upper_bound}'
            )

        if default_value is not None:
            if lower_bound is not None:
                if is_list:
                    for v in default_value:
                        if not (
                            v > lower_bound
                            or (lower_bound_inclusive and math.isclose(v, lower_bound))
                        ):
                            raise RuntimeError(
                                f'default_value {v} fails bound check: lower_bound {lower_bound}, lower_bound_inclusive {lower_bound_inclusive}'
                            )
                else:
                    if not (
                        default_value > lower_bound
                        or (
                            lower_bound_inclusive
                            and math.isclose(default_value, lower_bound)
                        )
                    ):
                        raise RuntimeError(
                            f'default_value {default_value} fails bound check: lower_bound {lower_bound}, lower_bound_inclusive {lower_bound_inclusive}'
                        )

        if default_value is not None:
            if upper_bound is not None:
                if is_list:
                    for v in default_value:
                        if not (
                            v < upper_bound
                            or (upper_bound_inclusive and math.isclose(v, upper_bound))
                        ):
                            raise RuntimeError(
                                f'default_value {v} fails bound check: upper_bound {upper_bound}, upper_bound_inclusive {upper_bound_inclusive}'
                            )
                else:
                    if not (
                        default_value < upper_bound
                        or (
                            upper_bound_inclusive
                            and math.isclose(default_value, upper_bound)
                        )
                    ):
                        raise RuntimeError(
                            f'default_value {default_value} fails bound check: upper_bound {upper_bound}, upper_bound_inclusive {upper_bound_inclusive}'
                        )

        if (
            list_min_lenth_inclusive is not None
            and list_max_length_inclusive is not None
            and list_min_lenth_inclusive > list_max_length_inclusive
        ):
            raise RuntimeError(
                f'list_min_lenth_inclusive {list_min_lenth_inclusive} should not be greater than list_max_length_inclusive {list_max_length_inclusive}.'
            )

        # create pb
        node = ParameterNode(
            name=name,
            doc_string=desc,
            type=ParameterNodeType.ATOMIC,
            atomic=AtomicParameterDef(
                type=AtomicParameterType.APT_FLOATS
                if is_list
                else AtomicParameterType.APT_FLOAT,
                is_optional=is_optional,
            ),
        )

        if default_value is not None:
            if is_list:
                node.atomic.default_value.fs.extend(default_value)
            else:
                node.atomic.default_value.f = default_value

        if allowed_values is not None:
            node.atomic.allowed_values.fs.extend(allowed_values)

        if lower_bound is not None:
            node.atomic.has_lower_bound = True
            node.atomic.lower_bound_inclusive = lower_bound_inclusive
            node.atomic.lower_bound.f = lower_bound

        if upper_bound is not None:
            node.atomic.has_upper_bound = True
            node.atomic.upper_bound_inclusive = upper_bound_inclusive
            node.atomic.upper_bound.f = upper_bound

        if is_list:
            if list_min_lenth_inclusive is not None:
                node.atomic.list_min_lenth_inclusive = list_min_lenth_inclusive
            else:
                node.atomic.list_min_lenth_inclusive = 0

            if list_max_length_inclusive is not None:
                node.atomic.list_max_length_inclusive = list_max_length_inclusive
            else:
                node.atomic.list_max_length_inclusive = -1

        # append
        self.__comp_param_decls.append(node)

    def int_param(
        self,
        name: str,
        desc: str,
        is_list: bool,
        is_optional: bool,
        default_value: Union[List[int], int] = None,
        allowed_values: List[int] = None,
        lower_bound: int = None,
        upper_bound: int = None,
        lower_bound_inclusive: bool = False,
        upper_bound_inclusive: bool = False,
        list_min_lenth_inclusive: int = None,
        list_max_length_inclusive: int = None,
    ):
        # sanity checks
        if not is_optional and default_value is None:
            raise RuntimeError('default_value must be provided if not optional ')

        if allowed_values is not None and (
            lower_bound is not None or upper_bound is not None
        ):
            raise RuntimeError(
                'allowed_values and bounds could not be set at the same time.'
            )

        if allowed_values is not None and default_value is not None:
            if is_list:
                for v in default_value:
                    if v not in allowed_values:
                        raise RuntimeError(
                            f'default_value {v} is not in allowed_values {allowed_values}'
                        )
            else:
                if default_value not in allowed_values:
                    raise RuntimeError(
                        f'default_value {default_value} is not in allowed_values {allowed_values}'
                    )

        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise RuntimeError(
                f'lower_bound {lower_bound} is greater than upper_bound {upper_bound}'
            )

        if default_value is not None:
            if lower_bound is not None:
                if is_list:
                    for v in default_value:
                        if not (
                            v > lower_bound
                            or (lower_bound_inclusive and v == lower_bound)
                        ):
                            raise RuntimeError(
                                f'default_value {v} fails bound check: lower_bound {lower_bound}, lower_bound_inclusive {lower_bound_inclusive}'
                            )
                else:
                    if not (
                        default_value > lower_bound
                        or (lower_bound_inclusive and default_value == lower_bound)
                    ):
                        raise RuntimeError(
                            f'default_value {default_value} fails bound check: lower_bound {lower_bound}, lower_bound_inclusive {lower_bound_inclusive}'
                        )

        if default_value is not None:
            if upper_bound is not None:
                if is_list:
                    for v in default_value:
                        if not (
                            v < upper_bound
                            or (upper_bound_inclusive and v == upper_bound)
                        ):
                            raise RuntimeError(
                                f'default_value {v} fails bound check: upper_bound {upper_bound}, upper_bound_inclusive {upper_bound_inclusive}'
                            )
                else:
                    if not (
                        default_value < upper_bound
                        or (upper_bound_inclusive and default_value == upper_bound)
                    ):
                        raise RuntimeError(
                            f'default_value {default_value} fails bound check: upper_bound {upper_bound}, upper_bound_inclusive {upper_bound_inclusive}'
                        )

        if (
            list_min_lenth_inclusive is not None
            and list_max_length_inclusive is not None
            and list_min_lenth_inclusive > list_max_length_inclusive
        ):
            raise RuntimeError(
                f'list_min_lenth_inclusive {list_min_lenth_inclusive} should not be greater than list_max_length_inclusive {list_max_length_inclusive}.'
            )

        # create pb
        node = ParameterNode(
            name=name,
            doc_string=desc,
            type=ParameterNodeType.ATOMIC,
            atomic=AtomicParameterDef(
                type=AtomicParameterType.APT_INTS
                if is_list
                else AtomicParameterType.APT_INT,
                is_optional=is_optional,
            ),
        )

        if default_value is not None:
            if is_list:
                node.atomic.default_value.i64s.extend(default_value)
            else:
                node.atomic.default_value.i64 = default_value

        if allowed_values is not None:
            node.atomic.allowed_values.i64s.extend(allowed_values)

        if lower_bound is not None:
            node.atomic.has_lower_bound = True
            node.atomic.lower_bound_inclusive = lower_bound_inclusive
            node.atomic.lower_bound.i64 = lower_bound

        if upper_bound is not None:
            node.atomic.has_upper_bound = True
            node.atomic.upper_bound_inclusive = upper_bound_inclusive
            node.atomic.upper_bound.i64 = upper_bound

        if is_list:
            if list_min_lenth_inclusive is not None:
                node.atomic.list_min_lenth_inclusive = list_min_lenth_inclusive
            else:
                node.atomic.list_min_lenth_inclusive = 0

            if list_max_length_inclusive is not None:
                node.atomic.list_max_length_inclusive = list_max_length_inclusive
            else:
                node.atomic.list_max_length_inclusive = -1

        # append
        self.__comp_param_decls.append(node)

    def str_param(
        self,
        name: str,
        desc: str,
        is_list: bool,
        is_optional: bool,
        default_value: Union[List[str], str] = None,
        allowed_values: List[str] = None,
        list_min_lenth_inclusive: int = None,
        list_max_length_inclusive: int = None,
    ):
        # sanity checks
        if not is_optional and default_value is None:
            raise RuntimeError('default_value must be provided if not optional ')

        if allowed_values is not None and default_value is not None:
            if is_list:
                for v in default_value:
                    if v not in allowed_values:
                        raise RuntimeError(
                            f'default_value {v} is not in allowed_values {allowed_values}'
                        )
            else:
                if default_value not in allowed_values:
                    raise RuntimeError(
                        f'default_value {default_value} is not in allowed_values {allowed_values}'
                    )

        if (
            list_min_lenth_inclusive is not None
            and list_max_length_inclusive is not None
            and list_min_lenth_inclusive > list_max_length_inclusive
        ):
            raise RuntimeError(
                f'list_min_lenth_inclusive {list_min_lenth_inclusive} should not be greater than list_max_length_inclusive {list_max_length_inclusive}.'
            )

        # create pb
        node = ParameterNode(
            name=name,
            doc_string=desc,
            type=ParameterNodeType.ATOMIC,
            atomic=AtomicParameterDef(
                type=AtomicParameterType.APT_STRINGS
                if is_list
                else AtomicParameterType.APT_STRING,
                is_optional=is_optional,
            ),
        )

        if default_value is not None:
            if is_list:
                node.atomic.default_value.ss.extend(default_value)
            else:
                node.atomic.default_value.s = default_value

        if allowed_values is not None:
            node.atomic.allowed_values.ss.extend(allowed_values)

        if is_list:
            if list_min_lenth_inclusive is not None:
                node.atomic.list_min_lenth_inclusive = list_min_lenth_inclusive
            else:
                node.atomic.list_min_lenth_inclusive = 0

            if list_max_length_inclusive is not None:
                node.atomic.list_max_length_inclusive = list_max_length_inclusive
            else:
                node.atomic.list_max_length_inclusive = -1

        # append
        self.__comp_param_decls.append(node)

    def bool_param(
        self,
        name: str,
        desc: str,
        is_list: bool,
        is_optional: bool,
        default_value: Union[List[bool], bool] = None,
        list_min_lenth_inclusive: int = None,
        list_max_length_inclusive: int = None,
    ):
        # sanity checks
        if not is_optional and default_value is None:
            raise RuntimeError('default_value must be provided if not optional ')

        if (
            list_min_lenth_inclusive is not None
            and list_max_length_inclusive is not None
            and list_min_lenth_inclusive > list_max_length_inclusive
        ):
            raise RuntimeError(
                f'list_min_lenth_inclusive {list_min_lenth_inclusive} should not be greater than list_max_length_inclusive {list_max_length_inclusive}.'
            )

        # create pb
        node = ParameterNode(
            name=name,
            doc_string=desc,
            type=ParameterNodeType.ATOMIC,
            atomic=AtomicParameterDef(
                type=AtomicParameterType.APT_BOOLS
                if is_list
                else AtomicParameterType.APT_BOOL,
                is_optional=is_optional,
            ),
        )

        if default_value is not None:
            if is_list:
                node.atomic.default_value.bs.extend(default_value)
            else:
                node.atomic.default_value.b = default_value

        if is_list:
            if list_min_lenth_inclusive is not None:
                node.atomic.list_min_lenth_inclusive = list_min_lenth_inclusive
            else:
                node.atomic.list_min_lenth_inclusive = 0

            if list_max_length_inclusive is not None:
                node.atomic.list_max_length_inclusive = list_max_length_inclusive
            else:
                node.atomic.list_max_length_inclusive = -1

        # append
        self.__comp_param_decls.append(node)

    def table_io(
        self,
        io_type: IoType,
        name: str,
        desc: str,
        types: List['TableType'],
        col_params: List[TableColParam] = None,
    ):
        # create pb
        table = TableDef(types=types)

        if col_params is not None:
            for col_param in col_params:
                col = table.cols.add()
                col.name = col_param.name
                col.doc_string = col_param.desc
                if col_param.col_list_min_cnt is not None:
                    col.col_list_min_cnt = col_param.col_list_min_cnt
                if col_param.col_list_max_cnt is not None:
                    col.col_list_max_cnt = col_param.col_list_max_cnt

        io_def = IoDef(
            name=name,
            doc_string=desc,
            data=[IoDef.SFDataDef(type=SFDataType.TABLE, table=table)],
        )

        # append
        if io_type == IoType.INPUT:
            self.__input_io_decls.append(io_def)
        else:
            self.__output_io_decls.append(io_def)

    def model_io(
        self,
        io_type: IoType,
        name: str,
        desc: str,
        types: List[str],
    ):
        # create pb
        model = ModelDef(types=types)

        io_def = IoDef(
            name=name,
            doc_string=desc,
            data=[IoDef.SFDataDef(type=SFDataType.MODEL, model=model)],
        )

        # append
        if io_type == IoType.INPUT:
            self.__input_io_decls.append(io_def)
        else:
            self.__output_io_decls.append(io_def)

    def eval_fn(self, f):
        import functools

        @functools.wraps(f)
        def decorator(*args, **kwargs):
            return f(*args, **kwargs)

        self.__eval_callback = f
        return decorator

    def definition(self):
        if self.__definition is None:
            comp_def = ComponentDef(
                domain=self.domain,
                name=self.name,
                doc_string=self.desc,
                version=self.version,
            )

            for p in self.__comp_param_decls:
                if p.name in self.__argnames:
                    raise RuntimeError(f'param {p.name} is duplicate.')
                self.__argnames.add(p.name)
                new_p = comp_def.params.add()
                new_p.CopyFrom(p)

            for io in self.__input_io_decls:
                if io.name in self.__argnames:
                    raise RuntimeError(f'input {io.name} is duplicate.')
                self.__argnames.add(io.name)
                new_io = comp_def.inputs.add()
                new_io.CopyFrom(io)

            for io in self.__output_io_decls:
                if io.name in self.__argnames:
                    raise RuntimeError(f'input {io.name} is duplicate.')
                self.__argnames.add(io.name)
                new_io = comp_def.outputs.add()
                new_io.CopyFrom(io)

            self.__definition = comp_def

        return self.__definition

    def eval(self, instance: NodeDef, secretflow_cluster_config):
        definition = self.definition()

        # sanity check on __eval_callback
        from inspect import signature

        PREDEFIND_PARAM = ['ctx']

        sig = signature(self.__eval_callback)
        for param in sig.parameters.values():
            if param.kind != param.KEYWORD_ONLY:
                raise RuntimeError(f'param {param.name} must be KEYWORD_ONLY.')
            if param.name not in PREDEFIND_PARAM and param.name not in self.__argnames:
                raise RuntimeError(f'param {param.name} is not allowed.')

        reader = NodeReader(instance=instance, definition=definition)
        kwargs = {'ctx': secretflow_cluster_config}

        for p in definition.params:
            kwargs[p.name] = reader.get_param(prefixes='', name=p.name)

        for input in definition.inputs:
            kwargs[input.name] = reader.get_input(name=input.name)

        for output in definition.outputs:
            kwargs[output.name] = reader.get_output(name=output.name)

        return self.__eval_callback(**kwargs)
