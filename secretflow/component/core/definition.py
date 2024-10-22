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


import importlib
import re
from dataclasses import MISSING
from dataclasses import Field as DField
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Type, get_args, get_origin

from google.protobuf import json_format
from google.protobuf.message import Message as PbMessage

from secretflow.error_system.exceptions import (
    CompDeclError,
    EvalParamError,
    InvalidArgumentError,
    NotSupportedError,
)
from secretflow.spec.v1.component_pb2 import (
    Attribute,
    AttributeDef,
    AttrType,
    ComponentDef,
    IoDef,
)
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from .checkpoint import Checkpoint
from .common.types import Input, Output, UnionGroup, UnionSelection
from .component import Component
from .dist_data.base import DistDataType
from .utils import clean_text


class Interval:
    def __init__(
        self,
        lower: float | int | None = None,
        upper: float | int | None = None,
        lower_closed: bool = False,
        upper_closed: bool = False,
    ):
        if lower is not None and upper is not None:
            assert upper >= lower

        self.lower = lower
        self.upper = upper
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed

    @staticmethod
    def open(lower: float | int | None, upper: float | int | None) -> 'Interval':
        '''return (lower, upper)'''
        return Interval(
            lower=lower, upper=upper, lower_closed=False, upper_closed=False
        )

    @staticmethod
    def closed(lower: float | int | None, upper: float | int | None) -> 'Interval':
        '''return [lower, upper]'''
        return Interval(lower=lower, upper=upper, lower_closed=True, upper_closed=True)

    @staticmethod
    def open_closed(lower: float | int | None, upper: float | int | None) -> 'Interval':
        '''return (lower, upper]'''
        return Interval(lower=lower, upper=upper, lower_closed=False, upper_closed=True)

    @staticmethod
    def closed_open(lower: float | int | None, upper: float | int | None) -> 'Interval':
        '''return [lower, upper)'''
        return Interval(lower=lower, upper=upper, lower_closed=True, upper_closed=False)

    def astype(self, typ: type):
        assert typ in [float, int]
        if self.lower is not None:
            self.lower = typ(self.lower)
        if self.upper is not None:
            self.upper = typ(self.upper)

    def enforce_closed(self):
        if self.lower != None:
            if isinstance(self.lower, float) and not self.lower.is_integer():
                raise ValueError(f"Lower bound must be an integer, {self.lower}")
            self.lower = int(self.lower)
            if not self.lower_closed:
                self.lower += 1
                self.lower_closed = True

        if self.upper != None:
            if isinstance(self.upper, float) and not self.upper.is_integer():
                raise ValueError(f"Upper bound must be an integer, {self.upper}")
            self.upper = int(self.upper)
            if not self.upper_closed:
                self.upper -= 1
                self.upper_closed = True

    def check(self, v: float | int) -> tuple[bool, str]:
        if self.upper is not None:
            if self.upper_closed:
                if v > self.upper:
                    return (
                        False,
                        f"should be less than or equal {self.upper}, but got {v}",
                    )
            else:
                if v >= self.upper:
                    return (
                        False,
                        f"should be less than {self.upper}, but got {v}",
                    )
        if self.lower is not None:
            if self.lower_closed:
                if v < self.lower:
                    return (
                        False,
                        f"should be greater than or equal {self.lower}, but got {v}",
                    )
            else:
                if v <= self.lower:
                    return (
                        False,
                        f"should be greater than {self.lower}, but got {v}",
                    )
        return True, ""


class FieldKind(Enum):
    BasicAttr = auto()
    PartyAttr = auto()
    CustomAttr = auto()
    StructAttr = auto()
    UnionAttr = auto()
    SelectionAttr = auto()
    TableColumnAttr = auto()
    Input = auto()
    Output = auto()


def is_deprecated_minor(minor_max: int, minor: int) -> bool:
    return minor_max != -1 and minor > minor_max


@dataclass
class _Metadata:
    prefixes: list = None
    fullname: str = ""
    name: str = ""
    type: Type = None
    kind: FieldKind = None
    desc: str = None
    is_optional: bool = False
    choices: list = None
    bound_limit: Interval = None
    list_limit: Interval = None
    default: Any = None
    selections: dict[str, UnionSelection] = None  # only used in union_group
    input_name: str = None  # only used in table_column_attr
    custom_pb_cls: str = None  # only used in custom attr
    is_checkpoint: bool = False  # if true it will be save when dump checkpoint
    types: list[str] = None  # only used in input/output
    minor_min: int = 0  # the first supported minor version
    minor_max: int = -1  # the last supported minor version

    @property
    def is_deprecated(self) -> bool:
        return self.minor_max != -1


class Field:
    @staticmethod
    def _field(
        kind: FieldKind,
        minor_min: int,
        minor_max: int,
        desc: str,
        md: _Metadata | None = None,
        default: Any = None,
        init=True,
    ):
        assert minor_max is not None
        if minor_max != -1 and minor_min > minor_max:
            raise NotSupportedError.not_supported_version(
                f"invalid minor version, {minor_min}, {minor_max}"
            )
        if md is None:
            md = _Metadata()
        md.kind = kind
        md.desc = clean_text(desc)
        md.minor_min = minor_min
        md.minor_max = minor_max

        if isinstance(default, list):
            default = MISSING
            default_factory = lambda: default
        else:
            default_factory = MISSING
        return field(
            default=default,
            default_factory=default_factory,
            init=init,
            kw_only=True,
            metadata={"md": md},
        )

    @staticmethod
    def attr(
        desc: str = "",
        is_optional: bool | None = None,
        default: Any | None = None,
        choices: list | None = None,
        bound_limit: Interval | None = None,
        list_limit: Interval | None = None,
        is_checkpoint: bool = False,
        minor_min: int = 0,
        minor_max: int = -1,
    ):
        if is_optional is None:
            is_optional = default != MISSING and default is not None

        md = _Metadata(
            is_optional=is_optional,
            choices=choices,
            bound_limit=bound_limit,
            list_limit=list_limit,
            is_checkpoint=is_checkpoint,
            default=default if default != MISSING else None,
        )
        return Field._field(
            FieldKind.BasicAttr, minor_min, minor_max, desc, md, default
        )

    @staticmethod
    def party_attr(
        desc: str = "",
        list_limit: Interval | None = None,
        minor_min: int = 0,
        minor_max: int = -1,
    ):
        md = _Metadata(list_limit=list_limit)
        return Field._field(FieldKind.PartyAttr, minor_min, minor_max, desc, md)

    @staticmethod
    def struct_attr(desc: str = "", minor_min: int = 0, minor_max: int = -1):
        return Field._field(FieldKind.StructAttr, minor_min, minor_max, desc)

    @staticmethod
    def union_attr(
        desc: str = "",
        default: str = "",
        selections: list[UnionSelection] | None = None,  # only used when type is str
        minor_min: int = 0,
        minor_max: int = -1,
    ):
        if selections:
            selections = {s.name: s for s in selections}
        md = _Metadata(default=default, selections=selections)
        return Field._field(FieldKind.UnionAttr, minor_min, minor_max, desc, md)

    @staticmethod
    def selection_attr(desc: str = "", minor_min: int = 0, minor_max: int = -1):
        return Field._field(FieldKind.SelectionAttr, minor_min, minor_max, desc)

    @staticmethod
    def custom_attr(desc: str = "", minor_min: int = 0, minor_max: int = -1):
        return Field._field(FieldKind.CustomAttr, minor_min, minor_max, desc)

    @staticmethod
    def table_column_attr(
        input_name: str,
        desc: str = "",
        limit: Interval | None = None,
        is_checkpoint: bool = False,
        minor_min: int = 0,
        minor_max: int = -1,
    ):
        if input_name == "":
            raise EvalParamError.missing_or_none_param("input_name cannot be empty")
        md = _Metadata(
            input_name=input_name,
            list_limit=limit,
            is_checkpoint=is_checkpoint,
        )
        return Field._field(FieldKind.TableColumnAttr, minor_min, minor_max, desc, md)

    @staticmethod
    def input(
        desc: str = "",
        types: list[str] = [],
        minor_min: int = 0,
        minor_max: int = -1,
        is_checkpoint: bool = False,
    ):
        if not types:
            raise EvalParamError.missing_or_none_param("input types is none")
        types = [str(s) for s in types]
        md = _Metadata(types=types, is_checkpoint=is_checkpoint)
        return Field._field(FieldKind.Input, minor_min, minor_max, desc, md)

    @staticmethod
    def output(
        desc: str = "",
        types: list[str] = [],
        minor_min: int = 0,
        minor_max: int = -1,
    ):
        if not types:
            raise EvalParamError.missing_or_none_param("output types is none")
        types = [str(s) for s in types]
        md = _Metadata(types=types)
        return Field._field(FieldKind.Output, minor_min, minor_max, desc, md)


class Creator:
    def __init__(self, check_exist: bool) -> None:
        self._check_exist = check_exist

    def make(self, cls: Type, kwargs: dict, minor: int):
        args = {}
        for name, field in cls.__dataclass_fields__.items():
            if name == MINOR_NAME:
                continue
            args[name] = self._make_field(field, kwargs, minor)
        if len(kwargs) > 0:
            raise ValueError(f"unused fields {kwargs}")

        args[MINOR_NAME] = minor
        ins = cls(**args)
        setattr(ins, MINOR_NAME, minor)
        return ins

    def _make_field(self, field: DField, kwargs: dict, minor: int):
        md: _Metadata = field.metadata['md']
        if is_deprecated_minor(md.minor_max, minor):
            return None

        if md.kind == FieldKind.StructAttr:
            return self._make_struct(md, kwargs, minor)
        elif md.kind == FieldKind.UnionAttr:
            return self._make_union(md, kwargs, minor)

        if minor < md.minor_min:
            return md.default

        if md.fullname not in kwargs:
            if self._check_exist and not md.is_optional:
                raise ValueError(f"{md.fullname} is required")
            else:
                return md.default

        value = kwargs.pop(md.fullname, md.default)

        if md.kind == FieldKind.Input:
            if not isinstance(value, DistData):
                raise EvalParamError.wrong_param_type(
                    f"type of {md.name} should be DistData"
                )

            if md.types is not None:
                if str(value.type) not in md.types:
                    raise EvalParamError.wrong_param_type(
                        f"type of {md.name} must be in {md.types}"
                    )
            return value if value.type != str(DistDataType.NULL) else None
        elif md.kind == FieldKind.Output:
            if not isinstance(value, (Output, str)):
                raise EvalParamError.wrong_param_type(
                    f"type of {md.name} should be str or Output, but got {type(value)}"
                )
            return value if isinstance(value, Output) else Output(uri=value, data=None)
        elif md.kind == FieldKind.TableColumnAttr:
            return self._make_str_or_list(md, value)
        elif md.kind == FieldKind.PartyAttr:
            return self._make_str_or_list(md, value)
        elif md.kind == FieldKind.CustomAttr:
            pb_cls = importlib.import_module("secretflow.spec.extend")
            for name in md.custom_pb_cls.split("."):
                pb_cls = getattr(pb_cls, name)
            return json_format.Parse(value, pb_cls())
        elif md.kind == FieldKind.BasicAttr:
            return self._make_basic(md, value)
        else:
            raise ValueError(f"invalid field kind, {md.fullname}, {md.kind}")

    def _make_struct(self, md: _Metadata, kwargs: dict, minor: int):
        cls = md.type
        args = {}
        for name, field in cls.__dataclass_fields__.items():
            args[name] = self._make_field(field, kwargs, minor)

        return cls(**args)

    def _make_union(self, md: _Metadata, kwargs: dict, minor: int):
        union_type = md.type
        if minor < md.minor_min:
            selected_key = md.default
        else:
            selected_key = kwargs.pop(md.fullname, md.default)

        if not isinstance(selected_key, str):
            raise ValueError(
                f"{md.fullname} should be a str, but got {type(selected_key)}"
            )
        if union_type == str:
            if selected_key not in md.selections:
                raise ValueError(f'{selected_key} not in {md.selections.keys()}')
            selection = md.selections[selected_key]
            if is_deprecated_minor(selection.minor_max, minor):
                raise ValueError(f"{selected_key} is deprecated")
            return selected_key

        choices = union_type.__dataclass_fields__.keys()
        if selected_key not in choices:
            raise ValueError(f"{selected_key} should be one of {choices}")

        selected_field = md.type.__dataclass_fields__[selected_key]
        selected_md: _Metadata = selected_field.metadata["md"]
        if is_deprecated_minor(selected_md.minor_max, minor):
            raise ValueError(f"{selected_key} is deprecated")

        args = {}
        if selected_md.kind != FieldKind.SelectionAttr:
            value = self._make_field(selected_field, kwargs, minor)
            args = {selected_key: value}
        res: UnionGroup = md.type(**args)
        res.set_selected(selected_key)
        return res

    def _make_basic(self, md: _Metadata, value):
        is_list = isinstance(value, list)
        if is_list and md.list_limit:
            is_valid, err_str = md.list_limit.check(len(value))
            if not is_valid:
                raise ValueError(f"length of {md.fullname} is valid, {err_str}")

        check_list = value if is_list else [value]
        if md.bound_limit is not None:
            for v in check_list:
                is_valid, err_str = md.bound_limit.check(v)
                if not is_valid:
                    raise ValueError(f"value of {md.fullname} is valid, {err_str}")
        if md.choices is not None:
            for v in check_list:
                if v not in md.choices:
                    raise ValueError(
                        f"value {v} must be in {md.choices}, name is {md.fullname}"
                    )
        return value

    def _make_str_or_list(self, md: _Metadata, value):
        if value is None:
            raise EvalParamError.missing_or_none_param(f"{md.name} can not be none")
        is_list = get_origin(md.type) is list
        if not is_list:
            if isinstance(value, list):
                if len(value) != 1:
                    raise ValueError(f"{md.name} can only have one element")
                value = value[0]
            assert isinstance(
                value, str
            ), f"{md.name} must be str, but got {type(value)}"
            return value
        else:
            assert isinstance(
                value, list
            ), f"{md.name} must be list[str], but got {type(value)}"
            if md.list_limit is not None:
                is_valid, err_str = md.list_limit.check(len(value))
                if not is_valid:
                    raise ValueError(f"length of {md.name} is invalid, {err_str}")

            return value


MINOR_NAME = "_minor"
RESERVED = ["input", "output"]


@dataclass
class _IoDef:
    io: IoDef  # type: ignore
    minor_min: int
    minor_max: int

    @property
    def name(self) -> str:
        return self.io.name

    @property
    def is_deprecated(self) -> bool:
        return self.minor_max != -1


@dataclass
class _AttrDef:
    attr: AttributeDef  # type: ignore
    minor_min: int
    minor_max: int

    @property
    def is_deprecated(self) -> bool:
        return self.minor_max != -1


class Reflector:
    def __init__(self, cls, name: str, minor: int):
        self._name = name
        self._minor = minor
        self._inputs: list[_IoDef] = []
        self._outputs: list[_IoDef] = []
        self._attrs: list[_AttrDef] = []
        self._attr_types: dict[str, AttrType] = {}  # type: ignore
        self.reflect(cls)

    def reflect(self, cls):  # type: ignore
        """
        Reflect dataclass to ComponentDef.
        """
        self._force_dataclass(cls)

        attrs: list[_Metadata] = []
        for field in cls.__dataclass_fields__.values():
            if field.name == MINOR_NAME:
                continue
            md = self._build_metadata(field, [])
            if md.kind == FieldKind.Input:
                io_def = self._reflect_io(md, Input)
                self._inputs.append(io_def)
            elif md.kind == FieldKind.Output:
                io_def = self._reflect_io(md, Output)
                self._outputs.append(io_def)
            else:
                attrs.append(md)

        for md in attrs:
            self._reflect_attr_field(md)

    def build_inputs(self) -> list[IoDef]:  # type: ignore
        return self._build_comp_io_defs(self._inputs, "input")

    def build_outputs(self) -> list[IoDef]:  # type: ignore
        return self._build_comp_io_defs(self._outputs, "output")

    def build_attrs(self) -> list[AttributeDef]:  # type: ignore
        return self._build_comp_attr_defs(self._attrs)

    def _build_comp_io_defs(self, io_defs: list[_IoDef], io_name: str) -> list[IoDef]:  # type: ignore
        if len(io_defs) == 0:
            return None

        return [d.io for d in io_defs if not d.is_deprecated]

    def _build_comp_attr_defs(self, attrs: list[_AttrDef]):
        result = []
        for attr in attrs:
            raw = attr.attr
            if attr.is_deprecated:
                if attr.minor_max >= self._minor:
                    raise NotSupportedError.not_supported_version(
                        f"minor_max of {raw.name} should be less than {self._minor}"
                    )
                continue
            result.append(raw)

        return result

    def _reflect_io(self, md: _Metadata, excepted_type: type):
        if md.type != excepted_type:
            raise CompDeclError(f"type of {md.name} must be {excepted_type}")
        is_optional = str(DistDataType.NULL) in md.types
        return _IoDef(
            io=IoDef(
                name=md.name, desc=md.desc, types=md.types, is_optional=is_optional
            ),
            minor_min=md.minor_min,
            minor_max=md.minor_max,
        )

    def _reflect_party_attr(self, md: _Metadata):
        is_list, org_type = self._check_list(md.type)
        if org_type != str:
            raise EvalParamError.wrong_param_type(
                f"the type of party attr should be str or list[str]"
            )
        list_min_length_inclusive, list_max_length_inclusive = self._build_list_limit(
            is_list, md.list_limit
        )
        if list_min_length_inclusive <= 0:
            md.is_optional = True
        atomic = AttributeDef.AtomicAttrDesc(
            list_min_length_inclusive=list_min_length_inclusive,
            list_max_length_inclusive=list_max_length_inclusive,
        )
        self._append_attr(AttrType.AT_PARTY, md, atomic=atomic)

    def _reflect_table_column_attr(self, md: _Metadata):
        is_list, prim_type = self._check_list(md.type)
        if prim_type != str:
            raise CompDeclError(
                f"input_table_attr's type must be str or list[str], but got {md.type}]"
            )

        input_name = md.input_name
        io_def = next((io.io for io in self._inputs if io.name == input_name), None)
        if io_def is None:
            raise CompDeclError(f"cannot find input io, {input_name}")
        for t in io_def.types:
            if t not in [
                str(DistDataType.VERTICAL_TABLE),
                str(DistDataType.INDIVIDUAL_TABLE),
            ]:
                raise CompDeclError(f"{input_name} is not defined correctly in input.")

        col_min_cnt_inclusive, col_max_cnt_inclusive = self._build_list_limit(
            is_list, md.list_limit
        )
        if col_min_cnt_inclusive <= 0:
            md.is_optional = True
        if md.prefixes:
            atomic = AttributeDef.AtomicAttrDesc(
                list_min_length_inclusive=col_min_cnt_inclusive,
                list_max_length_inclusive=col_max_cnt_inclusive,
            )
            self._append_attr(
                AttrType.AT_COL_PARAMS,
                md,
                atomic=atomic,
                col_params_binded_table=md.input_name,
            )
        else:
            if col_max_cnt_inclusive < 0:
                col_max_cnt_inclusive = 0
            preifx = md.input_name + "_"
            if md.name.startswith(preifx):
                name = md.name[len(preifx) :]
            else:
                name = md.name
            tbl_attr = IoDef.TableAttrDef(
                name=name,
                desc=md.desc,
                col_min_cnt_inclusive=col_min_cnt_inclusive,
                col_max_cnt_inclusive=col_max_cnt_inclusive,
            )
            io_def.attrs.append(tbl_attr)
            self._attr_types[md.fullname] = AttrType.AT_STRINGS

    def _reflect_attr_field(self, md: _Metadata):
        if md.kind == FieldKind.StructAttr:
            self._reflect_struct_attr(md)
        elif md.kind == FieldKind.UnionAttr:
            self._reflect_union_attr(md)
        elif md.kind == FieldKind.BasicAttr:
            self._reflect_basic_attr(md)
        elif md.kind == FieldKind.CustomAttr:
            self._reflect_custom_attr(md)
        elif md.kind == FieldKind.TableColumnAttr:
            self._reflect_table_column_attr(md)
        elif md.kind == FieldKind.PartyAttr:
            self._reflect_party_attr(md)
        else:
            raise CompDeclError(f"{md.kind} not supported, metadata={md}.")

    def _reflect_struct_attr(self, md: _Metadata):
        self._force_dataclass(md.type)

        self._append_attr(AttrType.AT_STRUCT_GROUP, md)

        prefixes = md.prefixes + [md.name]
        for field in md.type.__dataclass_fields__.values():
            sub_md = self._build_metadata(field, prefixes, md)
            self._reflect_attr_field(sub_md)

    def _reflect_union_attr(self, md: _Metadata):
        sub_mds = []
        prefixes = md.prefixes + [md.name]

        if md.type == str:
            if not md.selections:
                raise CompDeclError(f"no selections in {md.name}")
            prefix = '/'.join(prefixes)
            for s in md.selections.values():
                fullname = f"{prefix}/{s.name}"
                sub_md: _Metadata = _Metadata(
                    kind=FieldKind.SelectionAttr,
                    type=str,
                    prefixes=prefixes,
                    fullname=fullname,
                    name=s.name,
                    desc=clean_text(s.desc),
                    minor_min=s.minor_min,
                    minor_max=s.minor_max,
                )
                sub_mds.append(sub_md)
        else:
            if md.selections:
                raise CompDeclError(
                    f"cannot assign selections when type is not str, {md.name}"
                )
            if not issubclass(md.type, UnionGroup):
                raise CompDeclError(
                    f"type<{md.type}> of {md.name} must be subclass of UnionGroup."
                )

            self._force_dataclass(md.type)

            for field in md.type.__dataclass_fields__.values():
                sub_md: _Metadata = self._build_metadata(field, prefixes, parent=md)
                sub_mds.append(sub_md)

        md.choices = []
        for sub_md in sub_mds:
            if not sub_md.is_deprecated:
                md.choices.append(sub_md.name)

        if len(md.choices) == 0:
            raise CompDeclError(f"union {md.name} must have at least one choice.")

        if md.default == "":
            md.default = md.choices[0]
        elif md.default not in md.choices:
            raise CompDeclError(
                f"{md.default} not in {md.choices}, union name is {md.name}"
            )

        union_desc = AttributeDef.UnionAttrGroupDesc(default_selection=md.default)
        self._append_attr(AttrType.AT_UNION_GROUP, md, union=union_desc)

        for sub_md in sub_mds:
            if sub_md.kind == FieldKind.SelectionAttr:
                self._append_attr(AttrType.ATTR_TYPE_UNSPECIFIED, sub_md)
            else:
                self._reflect_attr_field(sub_md)

    def _reflect_custom_attr(self, md: _Metadata):
        pb_cls = md.type
        assert issubclass(
            pb_cls, PbMessage
        ), f"support protobuf class only, got {pb_cls}"

        extend_path = "secretflow.spec.extend."
        assert pb_cls.__module__.startswith(
            extend_path
        ), f"only support protobuf defined under {extend_path} path, got {pb_cls.__module__}"

        cls_name = ".".join(
            pb_cls.__module__[len(extend_path) :].split(".") + [pb_cls.__name__]
        )
        md.custom_pb_cls = cls_name
        self._append_attr(AttrType.AT_CUSTOM_PROTOBUF, md, pb_cls=md.custom_pb_cls)

    def _reflect_basic_attr(self, md: _Metadata):
        is_list, prim_type = self._check_list(md.type)
        attr_type = self._to_attr_type(prim_type, is_list)
        if attr_type == AttrType.ATTR_TYPE_UNSPECIFIED:
            raise CompDeclError(
                f"invalid primative type {prim_type}, name is {md.name}."
            )

        if is_list:
            list_min_length_inclusive, list_max_length_inclusive = (
                self._build_list_limit(True, md.list_limit)
            )
        else:
            list_min_length_inclusive, list_max_length_inclusive = None, None

        # check bound
        lower_bound_enabled = False
        lower_bound_inclusive = False
        lower_bound = None
        upper_bound_enabled = False
        upper_bound_inclusive = False
        upper_bound = None

        if md.bound_limit is not None:
            if prim_type not in [int, float]:
                raise CompDeclError(
                    f"bound limit is not supported for {prim_type}, name is {md.name}."
                )
            md.bound_limit.astype(prim_type)
            if md.choices is not None:
                for v in md.choices:
                    is_valid, err_str = md.bound_limit.check(v)
                    if not is_valid:
                        raise CompDeclError(
                            f"choices of {md.fullname} is valid, {err_str}"
                        )
            if md.bound_limit.lower is not None:
                lower_bound_enabled = True
                lower_bound_inclusive = md.bound_limit.lower_closed
                lower_bound = self._to_attr(prim_type(md.bound_limit.lower))
            if md.bound_limit.upper is not None:
                upper_bound_enabled = True
                upper_bound_inclusive = md.bound_limit.upper_closed
                upper_bound = self._to_attr(prim_type(md.bound_limit.upper))

        default_value = None
        allowed_values = None
        if md.is_optional and md.default is None:
            raise CompDeclError(f"no default value for optional field, {md.name}")
        if md.default is not None:
            if is_list and not isinstance(md.default, list):
                raise CompDeclError("Default value for list must be a list")

            # make sure the default type is correct
            if not isinstance(md.default, list):
                md.default = md.type(md.default)
            else:
                for idx, v in enumerate(md.default):
                    md.default[idx] = prim_type(md.default[idx])
            if md.choices is not None:
                values = md.default if is_list else [md.default]
                for v in values:
                    if v not in md.choices:
                        raise CompDeclError(
                            f"Default value for {v} must be one of {md.choices}"
                        )
            default_value = self._to_attr(md.default, prim_type)

        if md.choices is not None:
            allowed_values = self._to_attr(md.choices, prim_type)

        atomic = AttributeDef.AtomicAttrDesc(
            default_value=default_value,
            allowed_values=allowed_values,
            is_optional=md.is_optional,
            list_min_length_inclusive=list_min_length_inclusive,
            list_max_length_inclusive=list_max_length_inclusive,
            lower_bound_enabled=lower_bound_enabled,
            lower_bound_inclusive=lower_bound_inclusive,
            lower_bound=lower_bound,
            upper_bound_enabled=upper_bound_enabled,
            upper_bound_inclusive=upper_bound_inclusive,
            upper_bound=upper_bound,
        )
        self._append_attr(attr_type, md, atomic=atomic)

    def _append_attr(
        self,
        typ: str,
        md: _Metadata,
        atomic=None,
        union=None,
        pb_cls=None,
        col_params_binded_table=None,
    ):
        attr = AttributeDef(
            type=typ,
            name=md.name,
            desc=md.desc,
            prefixes=md.prefixes,
            atomic=atomic,
            union=union,
            custom_protobuf_cls=pb_cls,
            col_params_binded_table=col_params_binded_table,
        )
        self._attrs.append(
            _AttrDef(attr=attr, minor_min=md.minor_min, minor_max=md.minor_max)
        )
        if typ not in [AttrType.ATTR_TYPE_UNSPECIFIED, AttrType.AT_STRUCT_GROUP]:
            self._attr_types[md.fullname] = typ

    @staticmethod
    def _check_list(field_type) -> tuple[bool, type]:
        origin = get_origin(field_type)
        if origin is list:
            args = get_args(field_type)
            if not args:
                raise CompDeclError("list must have type.")
            return (True, args[0])
        else:
            return (False, field_type)

    @staticmethod
    def _build_metadata(
        field: DField, prefixes: list[str], parent: _Metadata = None
    ) -> _Metadata:
        if field.name in RESERVED:
            raise CompDeclError(f"{field.name} is a reserved word.")

        if "md" not in field.metadata:
            raise CompDeclError(f"md not exist in {field.name}, {field.metadata}")
        md: _Metadata = field.metadata["md"]
        md.name = field.name
        md.type = field.type
        md.prefixes = prefixes
        md.fullname = Reflector._to_fullname(prefixes, field.name)

        if parent != None:
            # inherit parent‘s minor_min version if it is zero
            if md.minor_min == 0:
                md.minor_min = parent.minor_min
            elif md.minor_min < parent.minor_min:
                raise CompDeclError(
                    f"minor version of {md.name} must be greater than or equal to {parent.minor_min}"
                )
        return md

    @staticmethod
    def _build_list_limit(is_list: bool, limit: Interval | None) -> tuple[int, int]:
        if not is_list:
            # limit must be 1 if target type is not list
            return (1, 1)
        if limit is None:
            return (0, -1)

        limit.enforce_closed()
        list_min_length_inclusive = 0
        list_max_length_inclusive = -1
        if limit.lower != None:
            assert limit.lower >= 0, f"list min size should be 1"
            list_min_length_inclusive = int(limit.lower)
        if limit.upper != None:
            list_max_length_inclusive = int(limit.upper)
        return (list_min_length_inclusive, list_max_length_inclusive)

    @staticmethod
    def _to_attr_type(prim_type, is_list) -> str:
        if prim_type is float:
            return AttrType.AT_FLOATS if is_list else AttrType.AT_FLOAT
        elif prim_type is int:
            return AttrType.AT_INTS if is_list else AttrType.AT_INT
        elif prim_type is str:
            return AttrType.AT_STRINGS if is_list else AttrType.AT_STRING
        elif prim_type is bool:
            return AttrType.AT_BOOLS if is_list else AttrType.AT_BOOL
        else:
            return AttrType.ATTR_TYPE_UNSPECIFIED

    @staticmethod
    def _to_attr(v: Any, prim_type: type | None = None) -> Attribute:  # type: ignore
        is_list = isinstance(v, list)
        if prim_type == None:
            if is_list:
                raise CompDeclError(f"unknown list primitive type for {v}")
            prim_type = type(v)

        if prim_type == bool:
            return Attribute(bs=v) if is_list else Attribute(b=v)
        elif prim_type == int:
            return Attribute(i64s=v) if is_list else Attribute(i64=v)
        elif prim_type == float:
            return Attribute(fs=v) if is_list else Attribute(f=v)
        elif prim_type == str:
            return Attribute(ss=v) if is_list else Attribute(s=v)
        else:
            raise CompDeclError(f"unsupported primitive type {prim_type}")

    @staticmethod
    def _to_fullname(prefixes: list, name: str) -> str:
        if prefixes is not None and len(prefixes) > 0:
            return '/'.join(prefixes) + '/' + name
        else:
            return name

    @staticmethod
    def _force_dataclass(cls):
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(cls)


class Definition:
    def __init__(
        self,
        cls: type[Component],
        domain: str,
        version: str,
        name: str = "",
        desc: str | None = None,
    ):
        if not issubclass(cls, Component):
            raise CompDeclError(f"{cls} must be subclass of Component")

        if name == "":
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

        if desc is None:
            desc = cls.__doc__ if cls.__doc__ is not None else ""

        versions = version.split(".")
        if len(versions) != 3:
            raise NotSupportedError.not_supported_version(
                f"version of Definition must be in format of x.y.z, but got {version}"
            )

        self.name = name
        self.domain = domain
        self.version = version

        self._minor = int(versions[1])

        r = Reflector(cls, self.name, self._minor)
        self._comp_def = ComponentDef(
            name=self.name,
            desc=clean_text(desc, no_line_breaks=False),
            domain=self.domain,
            version=self.version,
            inputs=r.build_inputs(),
            outputs=r.build_outputs(),
            attrs=r.build_attrs(),
        )
        self._inputs: list[_IoDef] = r._inputs
        self._inputs_map = {io.name: io for io in r._inputs}
        self._outputs: list[_IoDef] = r._outputs
        self._attrs: list[_AttrDef] = r._attrs
        self._attr_types = r._attr_types
        self._comp_cls = cls
        self._comp_id = f"{domain}/{name}:{version}"
        self._class_id = self.to_class_id(cls)

    def __str__(self) -> str:
        return json_format.MessageToJson(self._comp_def, indent=0)

    @staticmethod
    def to_class_id(cls) -> str:
        return f"{cls.__module__}:{cls.__qualname__}"

    @staticmethod
    def parse_id(comp_id: str) -> tuple[str, str, str]:
        pattern = r"(?P<domain>[^/]+)/(?P<name>[^:]+):(?P<version>.+)"
        match = re.match(pattern, comp_id)

        if match:
            return match.group("domain"), match.group("name"), match.group("version")
        else:
            raise ValueError(f"comp_id<{comp_id}> format is incorrect")

    @staticmethod
    def parse_minor(version: str) -> int:
        versions = version.split(".")
        assert len(versions) == 3, f"version<{version}> must be in format of x.y.z"
        minor = int(versions[1])
        assert minor >= 0, f"invalid minor<{minor}>"
        return minor

    @property
    def component_id(self) -> str:
        return self._comp_id

    @property
    def class_id(self) -> str:
        return self._class_id

    @property
    def component_def(self) -> ComponentDef:  # type: ignore
        return self._comp_def

    @property
    def attrs(self) -> list[AttributeDef]:  # type: ignore
        return self._comp_def.attrs

    @staticmethod
    def _get_io(io_defs: list[_IoDef], minor: int) -> list[IoDef]:  # type: ignore
        result = []
        for io in io_defs:
            if minor < io.minor_min:
                continue
            if is_deprecated_minor(io.minor_max, minor):
                continue
            result.append(io.io)

        return result

    def get_input_defs(self, minor: int) -> list[IoDef]:  # type: ignore
        return self._get_io(self._inputs, minor)

    def get_output_defs(self, minor: int) -> list[IoDef]:  # type: ignore
        return self._get_io(self._outputs, minor)

    def make_checkpoint(self, param: NodeEvalParam | dict, checkpoint_uri: str) -> Checkpoint:  # type: ignore
        if not checkpoint_uri:
            return None

        if isinstance(param, NodeEvalParam):
            kwargs = self.parse_param(param)
        elif isinstance(param, dict):
            kwargs = {}
            for k, v in param.items():
                k = self._trim_input_prefix(k)
                kwargs[k] = v
        else:
            raise ValueError(f"unsupported param type {type(param)}")

        if MINOR_NAME not in kwargs:
            raise KeyError(f"param kwargs must contain {MINOR_NAME}")
        minor = kwargs.pop(MINOR_NAME)

        parties = set()
        for v in kwargs.values():
            if not isinstance(v, DistData):
                continue
            for dr in v.data_refs:
                parties.add(dr.party)

        assert len(parties) > 0

        args = {}
        cls = self._comp_cls
        for name, field in cls.__dataclass_fields__.items():
            if name == MINOR_NAME:
                continue
            md: _Metadata = field.metadata['md']
            if md.kind not in [
                FieldKind.BasicAttr,
                FieldKind.TableColumnAttr,
                FieldKind.Input,
            ]:
                continue
            if is_deprecated_minor(md.minor_max, minor) or not md.is_checkpoint:
                continue

            args[md.fullname] = (
                kwargs[md.fullname] if md.fullname in kwargs else md.default
            )

        return Checkpoint(checkpoint_uri, args, sorted(list(parties)))

    def make_component(self, param: NodeEvalParam | dict, check_exist: bool = True) -> type[Component]:  # type: ignore
        if isinstance(param, NodeEvalParam):
            kwargs = self.parse_param(param)
        elif isinstance(param, dict):
            kwargs = {}
            for k, v in param.items():
                k = self._trim_input_prefix(k)
                kwargs[k] = v
        else:
            raise ValueError(f"unsupported param type {type(param)}")

        if MINOR_NAME not in kwargs:
            raise KeyError(f"param kwargs must contain {MINOR_NAME}")
        minor = kwargs.pop(MINOR_NAME)

        creator = Creator(check_exist=check_exist)
        ins = creator.make(self._comp_cls, kwargs, minor)
        return ins

    def parse_param(
        self,
        param: NodeEvalParam,  # type: ignore
        input_params: list[DistData] | None = None,
        output_params: list[str] | list[DistData] | None = None,
    ) -> dict:
        version = param.version.split('.')
        minor = int(version[1])

        attrs = self._parse_attrs(param)

        assert all(
            isinstance(item, DistData) for item in param.inputs
        ), f"type of inputs must be DistData"
        assert all(
            isinstance(item, str) for item in param.output_uris
        ), f"type of output_uris must be str"

        if input_params is None:
            input_params = param.inputs
        input_defs = self.get_input_defs(minor)
        assert len(input_defs) == len(
            input_params
        ), f"input size<{len(param.inputs)}> mismatch, expect {input_defs} but got {input_params}"
        inputs = {}
        for idx, io_def in enumerate(input_defs):
            in_param = input_params[idx]
            assert (
                in_param.type in io_def.types
            ), f"input type<{in_param.type}> mismatch, expect {io_def.types}"
            inputs[io_def.name] = in_param

        if output_params is None:
            output_params = param.output_uris

        output_defs = self.get_output_defs(minor)
        assert len(output_defs) == len(
            output_params
        ), f"input size<{len(output_params)}> mismatch, expect {output_defs} but got {output_params}"

        outputs = {}
        for idx, io_def in enumerate(output_defs):
            output_data = output_params[idx]
            if isinstance(output_data, str):
                output = Output(output_data, None)
            elif isinstance(output_data, DistData):
                output = Output("", output_data)
            else:
                raise ValueError(
                    f"unsupport output type<{type(output_data)}>, name={io_def.name}"
                )
            outputs[io_def.name] = output

        return {**attrs, **inputs, **outputs, MINOR_NAME: minor}

    def _parse_attrs(self, param: NodeEvalParam) -> dict:  # type: ignore
        attrs = {}
        for path, attr in zip(list(param.attr_paths), list(param.attrs)):
            path = self._trim_input_prefix(path)
            if path not in self._attr_types:
                raise KeyError(f"unknown attr key {path}")
            at = self._attr_types[path]
            attrs[path] = self._from_attr(attr, at)
        return attrs

    def _trim_input_prefix(self, p: str) -> str:
        if p.startswith("input/"):
            tokens = p.split('/', maxsplit=3)
            if len(tokens) != 3:
                raise InvalidArgumentError(f"invalid input, {p}")
            assert (
                tokens[1] in self._inputs_map
            ), f"unknown input table name<{p}> in {self.component_id}"
            key = "_".join(tokens[1:])
            if key in self._attr_types:
                return key
            return tokens[2]
        return p

    @staticmethod
    def _from_attr(value: Attribute, at: AttrType) -> Any:  # type: ignore
        if at == AttrType.ATTR_TYPE_UNSPECIFIED:
            raise ValueError("Type of Attribute is undefined.")
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
        elif at == AttrType.AT_BOOLS:
            return list(value.bs)
        elif at == AttrType.AT_CUSTOM_PROTOBUF:
            return value.s
        elif at == AttrType.AT_UNION_GROUP:
            return value.s
        elif at in [AttrType.AT_STRINGS, AttrType.AT_PARTY, AttrType.AT_COL_PARAMS]:
            return list(value.ss)
        elif at == AttrType.AT_STRUCT_GROUP:
            raise ValueError(f"AT_STRUCT_GROUP should be ignore")
        else:
            raise ValueError(f"unsupported type: {at}.")
