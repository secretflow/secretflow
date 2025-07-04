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


import numpy as np
import pyarrow as pa
from secretflow_spec import VTableField, VTableFieldKind, VTableFieldType, VTableSchema

from secretflow.data.vertical import VDataFrame

_np_dtype_to_pa_dtype = {
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


class VTableUtils:
    @staticmethod
    def from_dtype(dtype: pa.DataType | np.dtype | str) -> VTableFieldType:
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

    @staticmethod
    def np_dtype_to_pa_dtype(dt: np.generic | np.dtype) -> pa.DataType:
        if isinstance(dt, np.dtype):
            dt = dt.type
        return _np_dtype_to_pa_dtype[dt]

    @staticmethod
    def to_pa_dtype(ftype: VTableFieldType) -> pa.DataType:
        return _str_to_pa_dtype[str(ftype)]

    @staticmethod
    def to_np_dtype(ftype: VTableFieldType) -> np.dtype:
        return _str_to_np_dtype[str(ftype)]

    @staticmethod
    def to_serving_dtype(ftype: VTableFieldType) -> str:
        return _str_to_serving_type[str(ftype)]

    @staticmethod
    def to_duckdb_dtype(ftype: VTableFieldType) -> str:
        return _str_to_duckdb_dtype[str(ftype)]

    @staticmethod
    def from_arrow_schema(schema: pa.Schema, check_kind: bool = True) -> VTableSchema:
        fields = [
            VTableUtils.from_arrow_field(f, check_kind=check_kind) for f in schema
        ]
        return VTableSchema(fields)

    @staticmethod
    def to_arrow_schema(s: VTableSchema) -> pa.Schema:
        return pa.schema([VTableUtils.to_arrow_field(p) for p in s.fields.values()])

    @staticmethod
    def from_arrow_field(f: pa.Field, check_kind: bool = True) -> VTableField:
        f_type = _pa_dtype_to_str[f.type]
        kind = VTableFieldKind.UNKNOWN

        if f.metadata:
            if b'kind' in f.metadata:
                kind = VTableFieldKind.from_str(f.metadata[b'kind'].decode('utf-8'))
            if b'type' in f.metadata:
                raw_type = f.metadata[b'type'].decode('utf-8')
                is_same = VTableFieldType.is_same_type(f_type, raw_type)
                f_type = raw_type if is_same else f_type

        if kind == VTableFieldKind.UNKNOWN:
            if check_kind:
                raise f"kind should be in metadata of pa.Field {f}"
            kind = VTableFieldKind.FEATURE

        return VTableField(f.name, f_type, kind)

    @staticmethod
    def to_arrow_field(f: VTableField) -> pa.Field:
        pa_dtype = VTableUtils.to_pa_dtype(f.type)
        return VTableUtils.pa_field(f.name, pa_dtype, f.kind, f.type)

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


def build_schema(df: VDataFrame, labels: set = {"y"}) -> dict[str, VTableSchema]:
    res = {}
    for pyu, p in df.partitions.items():
        fields = []
        for name, dtype in p.dtypes.items():
            dt = VTableUtils.from_dtype(dtype)
            kind = VTableFieldKind.LABEL if name in labels else VTableFieldKind.FEATURE
            fields.append(VTableField(name, dt, kind))

        res[pyu.party] = VTableSchema(fields)

    return res
