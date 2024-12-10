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

import enum
from dataclasses import dataclass
from enum import IntFlag

import numpy as np
import pyarrow as pa

from secretflow.error_system.exceptions import (
    CompDeclError,
    CompEvalError,
    NotSupportedError,
)
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    SystemInfo,
    TableSchema,
    VerticalTable,
)

from ..common.types import BaseEnum
from .base import DistDataType

NP_DTYPE_TO_PA_DTYPE = {
    np.int8: pa.int8(),
    np.int16: pa.int16(),
    np.int32: pa.int32(),
    np.int64: pa.int64(),
    np.uint8: pa.uint8(),
    np.uint16: pa.uint16(),
    np.uint32: pa.uint32(),
    np.uint64: pa.uint64(),
    np.float32: pa.float32(),
    np.float64: pa.float64(),
    np.bool_: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    np.object_: pa.string(),
    np.str_: pa.string(),
}

_str_to_pa_dtype = {
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
    "float16": pa.float16(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
}

_pa_dtype_to_str = dict((v, k) for k, v in _str_to_pa_dtype.items())

_str_to_np_dtype = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
    "int": int,
    "float": float,
    "str": np.object_,
}

_np_dtype_to_str = dict((v, k) for k, v in _str_to_np_dtype.items())

_str_to_serving_type = {
    "int8": "DT_INT8",
    "int16": "DT_INT16",
    "int32": "DT_INT32",
    "int64": "DT_INT64",
    "uint8": "DT_UINT8",
    "uint16": "DT_UINT16",
    "uint32": "DT_UINT32",
    "uint64": "DT_UINT64",
    "float32": "DT_FLOAT",
    "float64": "DT_DOUBLE",
    "bool": "DT_BOOL",
    "int": 'DT_INT64',
    "float": 'DT_DOUBLE',
    "str": 'DT_STRING',
}

_str_to_duckdb_dtype = {
    "int8": "TINYINT",
    "int16": "SMALLINT",
    "int32": "INTEGER",
    "int64": "BIGINT",
    "uint8": "UTINYINT",
    "uint16": "USMALLINT",
    "uint32": "UINTEGER",
    "uint64": "UBIGINT",
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "bool": "BOOLEAN",
    "int": "BIGINT",
    "float": "DOUBLE",
    "str": "VARCHAR",
}

_same_type = {
    "int": "int64",
    "float": "float64",
    "int64": "int",
    "float64": "float",
}


def is_same_type(t1: str, t2: str) -> bool:
    return t1 == t2 or (t1 in _same_type and _same_type[t1] == t2)


@enum.unique
class VTableFormat(BaseEnum):
    CSV = "csv"
    ORC = "orc"


class VTableFieldType(BaseEnum):
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"

    @staticmethod
    def from_dtype(dtype: pa.DataType | np.dtype | str) -> "VTableFieldType":
        if isinstance(dtype, pa.DataType):
            assert dtype in _pa_dtype_to_str
            value = _pa_dtype_to_str[dtype]
        elif isinstance(dtype, np.dtype):
            assert dtype.type in _np_dtype_to_str, f"{dtype.type} not support"
            value = _np_dtype_to_str[dtype.type]
        elif isinstance(dtype, str):
            value = dtype
        else:
            raise ValueError(f"invalid dtype, {type(dtype)}, {dtype}")
        return VTableFieldType(value)

    def to_pa_dtype(self) -> pa.DataType:
        return _str_to_pa_dtype[self.value]

    def to_np_dtype(self) -> np.dtype:
        return _str_to_np_dtype[self.value]

    def to_serving_dtype(self) -> str:
        return _str_to_serving_type[self.value]

    def to_duckdb_dtype(self) -> str:
        return _str_to_duckdb_dtype[self.value]

    def is_float(self) -> bool:
        return self.value in ["float", "float16", "float32", "float64"]

    def is_string(self) -> bool:
        return self.value == "str"

    def is_bool(self) -> bool:
        return self.value == "bool"

    def is_signed_integer(self) -> bool:
        return self.value in ["int8", "int16", "int32", "int64"]

    def is_unsigned_integer(self) -> bool:
        return self.value in ["uint8", "uint16", "uint32", "uint64"]

    def is_integer(self) -> bool:
        return (
            self.value == "int"
            or self.is_signed_integer()
            or self.is_unsigned_integer()
        )


class VTableFieldKind(IntFlag):
    UNKNOWN = 0
    FEATURE = 1 << 0
    LABEL = 1 << 1
    ID = 1 << 2

    FEATURE_LABEL = FEATURE | LABEL
    ALL = FEATURE | LABEL | ID

    @staticmethod
    def from_str(s: str) -> "VTableFieldKind":
        return getattr(VTableFieldKind, s)

    def __str__(self):
        if self == VTableFieldKind.UNKNOWN:
            return "UNKNOWN"

        fields = []
        if self & VTableFieldKind.FEATURE:
            fields.append("FEATURE")
        if self & VTableFieldKind.LABEL:
            fields.append("LABEL")
        if self & VTableFieldKind.ID:
            fields.append("ID")

        return " | ".join(fields)


class VTableField:
    def __init__(self, name: str, ftype: str | VTableFieldType, kind: VTableFieldKind):
        self.name = name
        self.ftype = VTableFieldType(ftype)
        self.kind = kind

    def __eq__(self, other: 'VTableField') -> bool:
        return (
            self.name == other.name
            and self.ftype == other.ftype
            and self.kind == other.kind
        )

    def __str__(self) -> str:
        return f"VTableField(name: {self.name}, type: {str(self.ftype)}, kind: {self.kind})"

    @staticmethod
    def from_arrow(f: pa.Field, check_kind: bool = True) -> 'VTableField':
        f_type = _pa_dtype_to_str[f.type]
        kind = VTableFieldKind.UNKNOWN

        if f.metadata:
            if b'kind' in f.metadata:
                kind = VTableFieldKind.from_str(f.metadata[b'kind'].decode('utf-8'))
            if b'type' in f.metadata:
                raw_type = f.metadata[b'type'].decode('utf-8')
                f_type = raw_type if is_same_type(f_type, raw_type) else f_type

        if kind == VTableFieldKind.UNKNOWN:
            if check_kind:
                raise f"kind should be in metadata of pa.Field {f}"
            kind = VTableFieldKind.FEATURE

        return VTableField(f.name, f_type, kind)

    def to_arrow(self) -> pa.Field:
        return self.pa_field(self.name, self.ftype.to_pa_dtype(), self.kind, self.ftype)

    @staticmethod
    def pa_field(
        name: str, data_type: pa.DataType, kind: VTableFieldKind, raw_type: str = ""
    ) -> pa.Field:
        metadata = {"kind": kind.name}
        if raw_type != "":
            metadata["type"] = str(raw_type)
        return pa.field(name, data_type, metadata=metadata)

    @staticmethod
    def pa_field_from(name: str, dtype: pa.DataType, old: pa.Field) -> pa.Field:
        assert (
            old.metadata and b'kind' in old.metadata
        ), f"kind not in metadata, {old.name}, {old.metadata}"
        if dtype == old.type:
            metadata = old.metadata
        else:
            metadata = {b'kind': old.metadata[b'kind']}
        return pa.field(name, dtype, metadata=metadata)


class VTableSchema:
    def __init__(self, fields: list[VTableField] | dict[str, VTableField]) -> None:
        if isinstance(fields, list):
            fields = {f.name: f for f in fields}
        self.fields: dict[str, VTableField] = fields

    def __getitem__(self, key: int | str) -> VTableField:
        return self.field(key)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, VTableSchema):
            return self.fields == value.fields

        return False

    def __contains__(self, keys: list[str] | str) -> bool:
        if isinstance(keys, list):
            for k in keys:
                if k not in self.fields:
                    return False
            return True
        return keys in self.fields

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields.values()]

    @property
    def kinds(self) -> dict[str, VTableFieldKind]:
        return {f.name: f.kind for f in self.fields.values()}

    @property
    def types(self) -> dict[str, str]:
        return {f.name: f.ftype for f in self.fields.values()}

    def field(self, key: int | str) -> VTableField:
        if isinstance(key, int):
            keys = self.fields.keys()
            key = next(iter(keys)) if key == 0 else list(keys)[key]

        return self.fields[key]

    def select(self, columns: list[str]) -> 'VTableSchema':
        fields = {n: self.fields[n] for n in columns}
        return VTableSchema(fields)

    @staticmethod
    def from_dict(
        features: dict[str, str] = None,
        labels: dict[str, str] = None,
        ids: dict[str, str] = None,
    ) -> 'VTableSchema':
        kinds = [VTableFieldKind.FEATURE, VTableFieldKind.LABEL, VTableFieldKind.ID]
        values = [features, labels, ids]
        fields = []
        for kind, value in zip(kinds, values):
            if not value:
                continue
            fields.extend([VTableField(name, typ, kind) for name, typ in value.items()])

        return VTableSchema(fields)

    @staticmethod
    def from_pb_str(pb_str: str) -> "VTableSchema":
        pb = TableSchema()
        pb.ParseFromString(pb_str)
        return VTableSchema.from_pb(pb)

    @staticmethod
    def from_pb(schema: TableSchema) -> "VTableSchema":  # type: ignore
        return VTableSchema.from_dict(
            features={f: t for f, t in zip(schema.features, schema.feature_types)},
            labels={f: t for f, t in zip(schema.labels, schema.label_types)},
            ids={f: t for f, t in zip(schema.ids, schema.id_types)},
        )

    @staticmethod
    def from_arrow(schema: pa.Schema, check_kind: bool = True) -> 'VTableSchema':
        fields = [VTableField.from_arrow(f, check_kind=check_kind) for f in schema]
        return VTableSchema(fields)

    def to_arrow(self) -> pa.Schema:
        return pa.schema([p.to_arrow() for p in self.fields.values()])

    def to_pb(self) -> TableSchema:  # type: ignore
        feature_types = []
        features = []
        label_types = []
        labels = []
        id_types = []
        ids = []

        for f in self.fields.values():
            if f.kind == VTableFieldKind.FEATURE:
                feature_types.append(str(f.ftype))
                features.append(f.name)
            elif f.kind == VTableFieldKind.LABEL:
                label_types.append(str(f.ftype))
                labels.append(f.name)
            elif f.kind == VTableFieldKind.ID:
                id_types.append(str(f.ftype))
                ids.append(f.name)
            else:
                raise ValueError(f"invalid vtable field kind: {f}")

        return TableSchema(
            features=features,
            feature_types=feature_types,
            labels=labels,
            label_types=label_types,
            ids=ids,
            id_types=id_types,
        )


@dataclass
class VTableParty:
    party: str = ""
    uri: str = ""
    format: str = ""
    null_strs: list = None
    schema: VTableSchema = None

    @property
    def columns(self) -> list[str]:
        return self.schema.names

    @property
    def kinds(self) -> dict[str, VTableFieldKind]:
        return self.schema.kinds

    @property
    def types(self) -> dict[str, str]:
        return self.schema.types

    def copy(self, schema: VTableSchema) -> 'VTableParty':
        return VTableParty(self.party, self.uri, self.format, self.null_strs, schema)

    @staticmethod
    def from_dict(
        party: str = "",
        format: str = "",
        uri: str = "",
        null_strs: list = None,
        features: dict[str, str] = None,
        labels: dict[str, str] = None,
        ids: dict[str, str] = None,
    ) -> 'VTableParty':
        return VTableParty(
            party=party,
            uri=uri,
            format=format,
            null_strs=null_strs,
            schema=VTableSchema.from_dict(features, labels, ids),
        )


class VTable:
    def __init__(
        self,
        name: str,
        parties: dict[str, VTableParty] | list[VTableParty],
        line_count: int = 0,
        system_info: SystemInfo = None,
    ):
        if len(parties) == 0:
            raise CompEvalError.party_check_failed("empty parties when init VTable")
        if isinstance(parties, list):
            parties: dict[str, VTableParty] = {p.party: p for p in parties}

        self.name = name
        self.parties = parties
        self.line_count = line_count
        self.system_info = system_info

    @property
    def schemas(self) -> dict[str, VTableSchema]:
        ret: dict[str, VTableSchema] = {}
        for k, v in self.parties.items():
            ret[k] = v.schema

        return ret

    @property
    def flatten_schema(self) -> VTableSchema:
        if len(self.parties) == 1:
            return next(iter(self.parties.values())).schema
        else:
            fields = []
            for p in self.parties.values():
                fields.extend(p.schema.fields.values())
            return VTableSchema(fields)

    @property
    def columns(self) -> list[str]:
        ret = []
        for p in self.parties.values():
            ret.extend(p.schema.names)
        return ret

    def party(self, key: int | str) -> VTableParty:
        if isinstance(key, int):
            keys = self.parties.keys()
            key = next(iter(keys)) if key == 0 else list(keys)[key]

        return self.parties[key]

    def schema(self, key: int | str) -> VTableSchema:
        return self.party(key).schema

    def sort_partitions(self, order: list[str]) -> 'VTable':
        set_order = set(order)
        set_keys = set(self.parties.keys())
        if set_order != set_keys:
            raise CompEvalError.party_check_failed(
                f"unknown parties, {set_order}, {set_keys}"
            )
        parties = {p: self.parties[p] for p in order}
        return VTable(self.name, parties, self.line_count, self.system_info)

    def drop(self, excludes: list[str]) -> 'VTable':
        '''
        drop some columns, return new VTable
        '''
        if not excludes:
            raise ValueError(f"empty exclude columns set")

        excludes_set = set(excludes)
        parties = {}
        for p in self.parties.values():
            if len(excludes_set) == 0:
                parties[p.party] = p
                break

            fields = {}
            for f in p.schema.fields.values():
                if f.name in excludes_set:
                    excludes_set.remove(f.name)
                    continue
                fields[f.name] = f

            if len(fields) == 0:
                continue

            parties[p.party] = VTableParty(
                p.party, p.uri, p.format, p.null_strs, VTableSchema(fields)
            )

        if len(excludes_set) > 0:
            raise ValueError(f'unknowns columns, {excludes_set}')

        return VTable(self.name, parties, self.line_count, self.system_info)

    def select(self, columns: list[str]) -> 'VTable':
        '''
        select some columns by names, return new VTable
        '''
        if not columns:
            raise ValueError(f"columns cannot be empty")

        seen = set()
        duplicates = set(x for x in columns if x in seen or seen.add(x))
        if duplicates:
            raise f"has duplicate items<{duplicates}> in {columns}"

        columns_map = {name: idx for idx, name in enumerate(columns)}

        parties = {}
        for p in self.parties.values():
            fields = {}
            for f in p.schema.fields.values():
                if f.name not in columns_map:
                    continue
                fields[f.name] = f

            if len(fields) == 0:
                continue

            fields = {n: fields[n] for n in columns_map.keys() if n in fields}

            for n in fields.keys():
                del columns_map[n]

            parties[p.party] = VTableParty(
                p.party, p.uri, p.format, p.null_strs, VTableSchema(fields)
            )
            if len(columns_map) == 0:
                break

        if len(columns_map) > 0:
            raise ValueError(f'unknowns columns, {columns_map.keys()}')

        return VTable(self.name, parties, self.line_count, self.system_info)

    def select_by_kinds(self, kinds: VTableFieldKind) -> 'VTable':
        if kinds == VTableFieldKind.ALL:
            return self

        parties = {}
        for p in self.parties.values():
            fields = {}
            for f in p.schema.fields.values():
                if f.kind & kinds:
                    fields[f.name] = f

            parties[p.party] = p.copy(VTableSchema(fields))

        return VTable(self.name, parties, self.line_count, self.system_info)

    def check_kinds(self, kinds: VTableFieldKind):
        assert kinds != 0 and kinds != VTableFieldKind.ALL
        mismatch = {}
        for p in self.parties.values():
            for f in p.schema.fields.values():
                if not (kinds & f.kind):
                    mismatch[f.name] = str(f.kind)

        if len(mismatch) > 0:
            raise ValueError(f"kind of {mismatch} mismatch, expected {kinds}")

    @staticmethod
    def from_distdata(
        dd: DistData, columns: list[str] | None = None  # type: ignore
    ) -> 'VTable':
        dd_type = dd.type.lower()
        if dd_type not in [
            DistDataType.VERTICAL_TABLE,
            DistDataType.INDIVIDUAL_TABLE,
        ]:
            raise NotSupportedError.not_supported_data_type(
                f"Unsupported DistData type {dd_type}"
            )

        is_individual = dd_type == DistDataType.INDIVIDUAL_TABLE

        meta = IndividualTable() if is_individual else VerticalTable()
        dd.meta.Unpack(meta)

        pb_schemas = [meta.schema] if is_individual else meta.schemas
        if len(pb_schemas) == 0:
            raise CompDeclError.vtable_meta_schema_error(f"invliad schema {dd}")
        if len(dd.data_refs) != len(pb_schemas):
            raise CompDeclError.vtable_meta_schema_error(
                f"schemas<{len(pb_schemas)}> and data_refs<{len(dd.data_refs)}> mismatch"
            )

        if is_individual:
            if len(pb_schemas) != 1:
                raise CompDeclError.vtable_meta_schema_error(
                    f"invalid individual schema size<{len(pb_schemas)}> != 1 for INDIVIDUAL_TABLE"
                )
        else:
            if len(pb_schemas) <= 1:
                raise CompDeclError.vtable_meta_schema_error(
                    f"invalid vertical schema size<{len(pb_schemas)}> <= 1 for VERTICAL_TABLE"
                )

        def _parse_fields(schema: TableSchema) -> dict[str, VTableField]:  # type: ignore
            fields: list[VTableField] = []
            kind_list = [
                VTableFieldKind.FEATURE,
                VTableFieldKind.LABEL,
                VTableFieldKind.ID,
            ]
            name_list = [schema.features, schema.labels, schema.ids]
            type_list = [schema.feature_types, schema.label_types, schema.id_types]
            for kind, names, types in zip(kind_list, name_list, type_list):
                res = [VTableField(n, t, kind) for n, t in zip(names, types)]
                fields.extend(res)
            return fields

        parties: dict[str, VTableParty] = {}
        for idx, pb_schema in enumerate(pb_schemas):
            dr = dd.data_refs[idx]
            fields = _parse_fields(pb_schema)
            if len(fields) == 0:
                continue
            parties[dr.party] = VTableParty(
                party=dr.party,
                format=dr.format,
                uri=dr.uri,
                null_strs=list(dr.null_strs),
                schema=VTableSchema(fields),
            )

        res = VTable(
            name=dd.name,
            parties=parties,
            line_count=meta.line_count,
            system_info=dd.system_info,
        )
        if columns:
            res = res.select(columns)
        return res

    def to_distdata(self) -> DistData:
        assert len(self.parties) > 0
        parties = list(self.parties.values())
        is_individual = len(self.parties) == 1
        if is_individual:
            pb_schema = parties[0].schema.to_pb()
            meta = IndividualTable(schema=pb_schema, line_count=self.line_count)
            dtype = DistDataType.INDIVIDUAL_TABLE
        else:
            pb_schemas = [p.schema.to_pb() for p in parties]
            meta = VerticalTable(schemas=pb_schemas, line_count=self.line_count)
            dtype = DistDataType.VERTICAL_TABLE

        dd = DistData(
            name=self.name,
            type=str(dtype),
            system_info=self.system_info,
            data_refs=[
                DistData.DataRef(
                    uri=p.uri,
                    party=p.party,
                    format=str(p.format),
                    null_strs=p.null_strs,
                )
                for p in parties
            ],
        )
        dd.meta.Pack(meta)
        return dd
