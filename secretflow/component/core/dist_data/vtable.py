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

import copy
import enum
import inspect
import math
import traceback
from collections import defaultdict
from dataclasses import dataclass
from enum import IntFlag
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from secretflow.data import FedNdarray, partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import PYU, PYUObject, reveal, wait
from secretflow.device.device.base import Device
from secretflow.error_system.exceptions import (
    CompDeclError,
    CompEvalError,
    DataFormatError,
    NotSupportedError,
)
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    SystemInfo,
    TableSchema,
    VerticalTable,
)

from ..common.types import Output, TimeTracer
from ..storage import Storage
from .base import BaseEnum, DistDataType, IDumper
from .vtable_io import IReader, IWriter, new_reader, new_writer

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
    def from_arrow(f: pa.Field) -> 'VTableField':
        assert (
            f.metadata is not None and b'kind' in f.metadata
        ), f"kind should be in metadata of pa.Field {f}"
        f_type = _pa_dtype_to_str[f.type]
        kind = VTableFieldKind.from_str(f.metadata[b'kind'].decode('utf-8'))

        if b'type' in f.metadata:
            raw_type = f.metadata[b'type'].decode('utf-8')
            f_type = raw_type if is_same_type(f_type, raw_type) else f_type

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

    def __contains__(self, key) -> bool:
        return key in self.fields

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
    def from_arrow(schema: pa.Schema) -> 'VTableSchema':
        fields = [VTableField.from_arrow(f) for f in schema]
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
            if len(excludes_set) == 0:
                break

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


def _to_distdata_by_uri(
    schemas: dict[str, VTableSchema],
    uri: str,
    format: VTableFormat = VTableFormat.ORC,
    line_count: int = 0,
    system_info: SystemInfo = None,
) -> DistData:
    parties = list(schemas.keys())
    is_individual = len(schemas) == 1
    if is_individual:
        dd_type = DistDataType.INDIVIDUAL_TABLE
        pb_schema = schemas[parties[0]].to_pb()
        meta = IndividualTable(schema=pb_schema, line_count=line_count)
    else:
        dd_type = DistDataType.VERTICAL_TABLE
        pb_schemas = [schemas[p].to_pb() for p in parties]
        meta = VerticalTable(schemas=pb_schemas, line_count=line_count)

    dd = DistData(
        name=uri,
        type=str(dd_type),
        system_info=system_info,
        data_refs=[
            DistData.DataRef(uri=uri, party=p, format=str(format)) for p in parties
        ],
    )
    dd.meta.Pack(meta)
    return dd


class CompPartition:
    """
    Partition is PYUObject of pa.Table
    """

    def __init__(self, obj: PYUObject) -> None:
        self.obj = obj

    def to(self, device: Device, *args, **kwargs):
        return self.obj.to(device, *args, **kwargs)

    @property
    def device(self):
        return self.obj.device

    @property
    def device_type(self):
        return self.obj.device_type

    @property
    def data(self):
        return self.obj.data

    @property
    def shape(self) -> Tuple[int, int]:
        pyu = self.device
        return reveal(pyu(lambda t: t.shape)(self.data))

    @property
    def columns(self) -> List[str]:
        pyu = self.device
        return reveal(pyu(lambda t: t.column_names)(self.data))

    @property
    def types(self) -> Dict[str, pa.DataType]:
        pyu = self.device
        return reveal(pyu(lambda t: {c.name: c.type for c in t.schema})(self.data))

    @property
    def pandas_dtypes(self) -> Dict[str, np.dtype]:
        pyu = self.data.device
        dtypes = pyu(lambda t: {c.name: c.type.to_pandas_dtype() for c in t.schema})(
            self.data
        )
        return reveal(dtypes)

    @property
    def schema(self) -> pa.Schema:
        pyu = self.device
        return reveal(pyu(lambda t: t.schema))(self.data)

    @property
    def num_rows(self) -> int:
        pyu = self.device
        return reveal(pyu(lambda t: t.num_rows)(self.data))

    def __getitem__(self, items) -> "CompPartition":
        def _select(t, i):
            ret = t.select(i)
            return ret

        new_data = self.device(_select)(self.data, items)
        return CompPartition(new_data)

    def drop(self, items) -> "CompPartition":
        def _drop(t, i):
            return t.drop(i)

        new_data = self.device(_drop)(self.data, items)

        return CompPartition(new_data)

    def concat(self, other: "CompPartition", axis: int) -> "CompPartition":
        assert axis in [0, 1]
        assert self.device == other.device

        if axis == 0:
            assert self.shape[1] == other.shape[1]
            assert self.columns == other.columns

            def _concat(t1: pa.Table, t2: pa.Table):
                return pa.concat_tables([t1, t2])

        else:
            assert self.shape[0] == other.shape[0]
            assert len(set(self.columns).intersection(set(other.columns))) == 0

            def _concat(dest: pa.Table, source: pa.Table):
                for i in range(source.shape[1]):
                    dest = dest.append_column(source.field(i), source.column(i))
                return dest

        new_data = self.device(_concat)(self.data, other.data)

        return CompPartition(new_data)


class CompVDataFrame(IDumper):
    def __init__(
        self, partitions: dict[PYU, CompPartition], system_info: SystemInfo = None
    ) -> None:
        self.partitions = partitions
        self.system_info = system_info

    def data(self, pyu: PYU) -> PYUObject:
        if pyu not in self.partitions:
            raise CompEvalError.party_check_failed(f"pyu {pyu} not exist in comp data")
        return self.partitions[pyu].data

    def set_data(self, data: PYUObject) -> None:
        assert isinstance(data, PYUObject)
        pyu = data.device
        self.partitions[pyu] = CompPartition(data)

    def has_data(self, pyu: PYU) -> bool:
        return pyu in self.partitions

    def partition(self, key: int | str | PYU) -> CompPartition:
        if isinstance(key, int):
            keys = self.partitions.keys()
            key = next(iter(keys)) if key == 0 else list(keys)[key]
        elif isinstance(key, str):
            key = PYU(key)

        return self.partitions[key]

    def to_pandas(self, check_null=True) -> VDataFrame:
        def _to_pandas(t: pa.Table):
            if not check_null:
                return t.to_pandas()

            # pyarrow's to_pandas() will change col type if col contains NULL
            # and trainning comp cannot handle NULL too,
            # so, failed if table contains NULL.
            for c in t.column_names:
                if pc.any(pc.is_null(t[c], nan_is_null=True)).as_py():
                    raise DataFormatError.none_filed_in_column(column=c)

            return t.to_pandas()

        pandas = {}
        for pyu, pa_table in self.partitions.items():
            pandas[pyu] = pyu(_to_pandas)(pa_table.data)
        wait(pandas)
        return VDataFrame({p: partition(pandas[p]) for p in pandas})

    @staticmethod
    def from_pandas(
        v_data: VDataFrame,
        schemas: dict[str, VTableSchema],
        system_info: SystemInfo = None,
    ) -> "CompVDataFrame":
        assert isinstance(v_data, VDataFrame)

        def _from_pandas(df: pd.DataFrame, ts: VTableSchema) -> pa.Table:
            kinds = ts.kinds
            fields = []
            for name, dtype in zip(df.columns, df.dtypes):
                pa_dtype = NP_DTYPE_TO_PA_DTYPE.get(dtype.type)
                if pa_dtype is None:
                    raise ValueError(f"unsupport type: {dtype}")
                field = VTableField.pa_field(name, pa_dtype, kinds[name])
                fields.append(field)
            schema = pa.schema(fields)
            res = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            return res

        partitions = {}
        for pyu, partition in v_data.partitions.items():
            data = pyu(_from_pandas)(partition.data, schemas[pyu.party])
            partitions[pyu] = CompPartition(data)

        return CompVDataFrame(partitions, system_info)

    @staticmethod
    def from_values(
        f_nd: FedNdarray,
        schemas: dict[str, VTableSchema],
        system_info: SystemInfo = None,
    ) -> "CompVDataFrame":
        assert isinstance(f_nd, FedNdarray)

        def _from_values(data: np.ndarray, ts: VTableSchema) -> pa.Table:
            pa_dtype = NP_DTYPE_TO_PA_DTYPE.get(data.dtype.type)
            if pa_dtype is None:
                raise NotSupportedError.not_supported_data_type(
                    f"unsupported type: {data.dtype}"
                )

            if data.shape[1] != len(ts.fields):
                raise DataFormatError.feature_not_matched(
                    f"columns size mismatch, {data.shape}, {ts.fields.keys()}"
                )

            fields = []
            for f in ts.fields.values():
                field = VTableField.pa_field(f.name, pa_dtype, f.kind)
                fields.append(field)
            schema = pa.schema(fields)
            df = pd.DataFrame(data, columns=schema.names)
            res = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            # res = pa.Table.from_arrays(data, schema=schema)
            return res

        partitions = {}
        for pyu, partition in f_nd.partitions.items():
            if pyu.party not in schemas:
                raise CompEvalError.party_check_failed(
                    f"pyu.party {pyu.party} not in schemas"
                )
            data = pyu(_from_values)(partition, schemas[pyu.party])
            partitions[pyu] = CompPartition(data)

        return CompVDataFrame(partitions, system_info)

    def _col_index(self, items) -> Dict[PYU, List[str]]:
        if not self.partitions:
            raise DataFormatError.empty_dataset(f"can not get index on empty DataFrame")

        if hasattr(items, "tolist"):
            items = items.tolist()
        if not isinstance(items, (list, tuple)):
            items = [items]

        columns_to_party = {}
        for pyu, p in self.partitions.items():
            columns_to_party.update({c: pyu for c in p.columns})

        ret = defaultdict(list)
        for item in items:
            pyu = columns_to_party.get(item, None)
            if pyu is None:
                raise CompEvalError(f'Item {item} does not exist in columns_to_party.')
            ret[pyu].append(item)

        # keep party order
        return {p: ret[p] for p in self.partitions.keys() if p in ret}

    def __getitem__(self, items) -> "CompVDataFrame":
        items = self._col_index(items)

        partitions = {}
        for p, i in items.items():
            t = self.partitions[p]
            partitions[p] = t[i]

        wait([t.data for t in partitions.values()])
        return CompVDataFrame(partitions)

    def drop(self, items) -> "CompVDataFrame":
        items = self._col_index(items)

        partitions = copy.deepcopy(self.partitions)
        for p, i in items.items():
            t = self.partitions[p].drop(i)
            if len(t.columns) > 0:
                partitions[p] = t
            else:
                del partitions[p]
        return CompVDataFrame(partitions)

    def concat(self, other: "CompVDataFrame", axis: int) -> "CompVDataFrame":
        assert axis in [0, 1]
        if axis == 1:
            if not set(other.partitions.keys()).issubset(self.partitions.keys()):
                raise CompEvalError(
                    f"other DataFrame's partition {other.partitions.keys()} not in self partition {self.partitions.keys()}"
                )
        else:
            if set(other.partitions.keys()) != set(self.partitions.keys()):
                raise CompEvalError(
                    f"other DataFrame's partition {other.partitions.keys()} not equal to self partition {self.partitions.keys()}"
                )

        ret_partitions = copy.deepcopy(self.partitions)
        for pyu, t2 in other.partitions.items():
            t1 = self.partitions[pyu]
            ret_partitions[pyu] = t1.concat(t2, axis)
        wait([t.data for t in ret_partitions.values()])
        return CompVDataFrame(ret_partitions)

    @property
    def num_rows(self) -> int:
        return self.partition(0).num_rows

    @property
    def columns(self) -> List[str]:
        result = []
        for p in self.partitions.values():
            result.extend(p.columns)
        return result

    @property
    def shape(self) -> Tuple[int, int]:
        shapes = []
        for p in self.partitions.values():
            shapes.append(p.shape)

        return shapes[0][0], sum([shape[1] for shape in shapes])

    def apply(
        self,
        fn: Callable[[pa.Table, list[str]], pa.Table],
        columns: list[str] | set[str] = None,
    ) -> 'CompVDataFrame':
        signature = inspect.signature(fn)
        is_one_param = len(signature.parameters) == 1

        def _handle(
            in_tbl: pa.Table, column_set: set[str]
        ) -> Tuple[pa.Table, Exception]:
            try:
                party_columns = None
                if column_set is not None:
                    name_set = set(in_tbl.column_names)
                    party_columns = list(column_set.intersection(name_set))

                if is_one_param:
                    out_tbl = fn(in_tbl)
                else:
                    out_tbl = fn(in_tbl, party_columns)
                return out_tbl, None
            except Exception as e:
                traceback.print_exc()
                return None, e

        column_set = set(columns) if columns is not None else None

        out_partitions = {}
        for pyu, obj in self.partitions.items():
            data_obj, err_obj = pyu(_handle)(obj.data, column_set)
            out_partitions[pyu] = CompPartition(data_obj)
            err = reveal(err_obj)
            if err is not None:
                raise err

        return CompVDataFrame(out_partitions, system_info=self.system_info)

    @staticmethod
    def load(storage: Storage, dd: DistData | VTable) -> "CompVDataFrame":
        vtbl = dd if isinstance(dd, VTable) else VTable.from_distdata(dd)
        partitions = {}
        for party, p in vtbl.parties.items():
            pa_schema = p.schema.to_arrow()
            r = new_reader(
                PYU(party), storage, p.format, p.uri, p.null_strs, pa_schema, 0
            )
            data_obj = r.read_all()
            p = CompPartition(data_obj)
            partitions[PYU(party)] = CompPartition(data_obj)
        ret = CompVDataFrame(partitions=partitions, system_info=dd.system_info)
        if not math.prod(ret.shape):
            raise DataFormatError.empty_dataset(
                f"empty dataset {ret.shape} is not allowed"
            )
        return ret

    def dump(self, storage: Storage, uri: str, format: VTableFormat = VTableFormat.ORC) -> DistData:  # type: ignore
        if not self.partitions:
            raise DataFormatError.empty_dataset("can not dump empty dataframe")

        closes = []
        lines = []
        for pyu, party in self.partitions.items():
            w = new_writer(pyu, storage, format, uri)
            lines.append(w.write(party.data))
            closes.append(w.close())
        lines = reveal(lines)
        if len(set(lines)) != 1:
            raise DataFormatError.dataset_not_aligned(
                f"DataFrame is not aligned, {lines}"
            )
        if lines[0] <= 0:
            raise DataFormatError.empty_dataset(
                f"empty dataset is not allowed, line_count={lines[0]}"
            )
        wait(closes)

        schemas = {}
        for pyu, obj in self.partitions.items():
            schemas[pyu.party] = VTableSchema.from_arrow(obj.schema)

        return _to_distdata_by_uri(
            schemas, uri, format, line_count=lines[0], system_info=self.system_info
        )


class CompVDataFrameReader:
    def __init__(
        self,
        storage: Storage,
        tracer: TimeTracer,
        dd: DistData | VTable,
        batch_size: int = -1,
    ) -> None:
        vtbl = dd if isinstance(dd, VTable) else VTable.from_distdata(dd)
        readers: dict[str, IReader] = {}
        for p in vtbl.parties.values():
            pyu = PYU(p.party)
            pa_schema = p.schema.to_arrow()
            reader = new_reader(
                pyu, storage, p.format, p.uri, p.null_strs, pa_schema, batch_size
            )
            readers[pyu] = reader
        self._readers = readers
        self._storage = storage
        self._tracer = tracer
        self._system_info = dd.system_info

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        df = self.read_next()
        if df is None:
            raise StopIteration
        return df

    def read_next(self) -> CompVDataFrame:
        datas = {}
        eofs = []
        for p, r in self._readers.items():
            (data_obj, eof_obj) = r.read_next()
            datas[p] = CompPartition(data_obj)
            eofs.append(eof_obj)

        eofs = set(reveal(eofs))
        if len(eofs) != 1:
            raise DataFormatError.dataset_not_aligned(
                f"DataFrame is not aligned, {eofs}"
            )
        eof = next(iter(eofs)) == True

        if eof:
            return None

        ret = CompVDataFrame(datas, system_info=self._system_info)
        if not math.prod(ret.shape):
            raise DataFormatError.empty_dataset(
                f"empty dataset {ret.shape} is not allowed"
            )
        return ret

    def close(self) -> None:
        for r in self._readers.values():
            r.close()
        self._readers = {}


class CompVDataFrameWriter:
    def __init__(
        self,
        storage: Storage,
        tracer: TimeTracer,
        uri: str,
        format: VTableFormat = VTableFormat.ORC,
    ) -> None:
        self.line_count = 0
        self._uri = uri
        self._format = format
        self._tracer = tracer
        self._storage = storage
        self._writers: dict[PYU, IWriter] = None
        self._schemas: dict[str, VTableSchema] = None
        self._system_info: SystemInfo = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _init_writers(self, df: CompVDataFrame) -> None:
        if self._writers is not None:
            return
        if not df.partitions:
            raise DataFormatError.empty_dataset("cannot write empty DataFrame")
        self._writers = {}
        self._schemas = {}
        for pyu, obj in df.partitions.items():
            party = pyu.party
            w = new_writer(pyu, self._storage, self._format, self._uri)
            self._schemas[party] = VTableSchema.from_arrow(obj.schema)
            self._writers[pyu] = w

        self._system_info = df.system_info

    def write(self, df: CompVDataFrame):
        self._init_writers(df)
        partitions = df.partitions
        write_lines = []
        for p, w in self._writers.items():
            assert p in partitions
            write_lines.append(w.write(partitions[p].data))
        write_lines = reveal(write_lines)
        if len(set(write_lines)) != 1:
            raise DataFormatError.dataset_not_aligned(
                f"DataFrame is not aligned, lines: {write_lines}"
            )
        self.line_count += write_lines[0]

    def close(self) -> None:
        if self._writers is not None:
            wait([w.close() for w in self._writers.values()])
            self._writers = None

    def dump(self) -> DistData:
        if self._schemas is None or self.line_count <= 0:
            raise DataFormatError.empty_dataset(
                f"empty dataset is not allowed, schema={self._schemas}, line_count={self.line_count}"
            )
        return _to_distdata_by_uri(
            self._schemas,
            self._uri,
            format=self._format,
            line_count=self.line_count,
            system_info=self._system_info,
        )

    def dump_to(self, out: Output):
        out.data = self.dump()


def save_prediction(
    storage: Storage,
    tracer: TimeTracer,
    uri: str,
    pyu: PYU,
    batch_pred: Callable,
    pred_name: str,
    pred_features: list[str],
    pred_partitions_order: list[str],
    feature_dataset: DistData,
    saved_features: list[str],
    saved_labels: list[str],
    save_ids: bool,
    check_null: bool = True,
) -> DistData:
    tbl = VTable.from_distdata(feature_dataset)

    addition_cols = []
    if save_ids:
        t = tbl.select_by_kinds(kinds=VTableFieldKind.ID)
        addition_cols.extend(t.parties[pyu.party].columns)
    if saved_features:
        addition_cols.extend(saved_features)
    if saved_labels:
        addition_cols.extend(saved_labels)

    # all columns must belong be receiver
    receiver_cols_set = set(tbl.parties[pyu.party].columns)
    addition_cols_set = set()
    for f in addition_cols:
        if f in addition_cols_set or f == pred_name:
            raise DataFormatError.feature_intersection_with_label(
                f"do not select {f} as saved feature, repeated with id or label"
            )
        if f not in receiver_cols_set:
            raise CompEvalError.party_check_failed(
                f"The saved feature {addition_cols} can only belong to receiver party {pyu.party}, but {f} is not"
            )
        addition_cols_set.add(f)

    if addition_cols:
        reader_features = pred_features + [
            a for a in addition_cols if a not in pred_features
        ]
    else:
        reader_features = pred_features

    tbl = tbl.select(reader_features)
    if pred_partitions_order is not None:
        if pyu.party not in pred_partitions_order:
            pred_partitions_order.append(pyu.party)
        tbl = tbl.sort_partitions(pred_partitions_order)

    reader = CompVDataFrameReader(storage, tracer, tbl)
    writer = CompVDataFrameWriter(storage, tracer, uri)
    pred_schemas = {
        pyu.party: VTableSchema(
            [VTableField(pred_name, "float", VTableFieldKind.LABEL)]
        )
    }
    with writer:
        for batch in reader:
            pred_batch = batch[pred_features].to_pandas(check_null)
            pred = batch_pred(pred_batch)
            assert len(pred.partitions) == 1
            if pyu not in pred.partitions:
                raise CompEvalError.party_check_failed(
                    f"pyu [{pyu}] not in pred.partitions [{pred.partitions}]"
                )
            if isinstance(pred, FedNdarray):
                pred_df = CompVDataFrame.from_values(
                    pred, pred_schemas, feature_dataset.system_info
                )
            elif isinstance(pred, VDataFrame):
                pred_df = CompVDataFrame.from_pandas(
                    pred, pred_schemas, feature_dataset.system_info
                )

            if addition_cols:
                addition_df = batch[addition_cols]
                assert len(addition_df.partitions) == 1
                if pyu not in addition_df.partitions:
                    raise CompEvalError.party_check_failed(
                        f"pyu [{pyu}] not in addition_df.partitions [{addition_df.partitions}]"
                    )
                out_df = pred_df.concat(addition_df, axis=1)
            else:
                out_df = pred_df
            writer.write(out_df)

    return writer.dump()
