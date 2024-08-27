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
from secretflow.device.device.pyu import PYU
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

_str_to_dtype = {
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

_dtype_to_str = dict((v, k) for k, v in _str_to_dtype.items())

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


@dataclass
class VTableField:
    name: str
    type: str
    kind: VTableFieldKind

    @property
    def dtype(self) -> pa.DataType:
        return _str_to_dtype[self.type]

    @staticmethod
    def from_arrow(f: pa.Field):
        assert (
            f.metadata is not None and b'kind' in f.metadata
        ), f"kind should be in metadata of pa.Field {f}"
        ftype = _dtype_to_str[f.type]
        kind = VTableFieldKind.from_str(f.metadata[b'kind'].decode('utf-8'))

        if b'type' in f.metadata:
            raw_type = f.metadata[b'type'].decode('utf-8')
            ftype = raw_type if is_same_type(ftype, raw_type) else ftype

        return VTableField(f.name, ftype, kind)

    def to_arrow(self) -> pa.Field:
        return self.pa_field(self.name, self.dtype, self.kind, self.type)

    @staticmethod
    def pa_field(
        name: str, data_type: pa.DataType, kind: VTableFieldKind, raw_type: str = ""
    ) -> pa.Field:
        metadata = metadata = {"kind": kind.name}
        if raw_type != "":
            metadata["type"] = raw_type
        return pa.field(name, data_type, metadata=metadata)


@dataclass
class VTableSchema:
    party: str = ""
    format: str = ""
    uri: str = ""
    null_strs: list = None
    fields: list[VTableField] = None

    def to_arrow(self) -> pa.Schema:
        return pa.schema([p.to_arrow() for p in self.fields])

    def to_pb(self) -> TableSchema:  # type: ignore
        feature_types = []
        features = []
        label_types = []
        labels = []
        id_types = []
        ids = []

        for col in self.fields:
            if col.kind == VTableFieldKind.FEATURE:
                feature_types.append(col.type)
                features.append(col.name)
            elif col.kind == VTableFieldKind.LABEL:
                label_types.append(col.type)
                labels.append(col.name)
            elif col.kind == VTableFieldKind.ID:
                id_types.append(col.type)
                ids.append(col.name)
            else:
                raise ValueError(f"invalid vtable field kind: {self}")

        return TableSchema(
            features=features,
            feature_types=feature_types,
            labels=labels,
            label_types=label_types,
            ids=ids,
            id_types=id_types,
        )


def _parse_fields(
    schema: TableSchema,  # type: ignore
    kinds: VTableFieldKind = VTableFieldKind.ALL,
    col_selects: list[str] = None,
) -> list[VTableField]:
    col_selects_set = set(col_selects) if col_selects is not None else None

    def _to_fields(
        kind: VTableFieldKind, names: list, types: list
    ) -> list[VTableField]:
        result = []
        for n, t in zip(names, types):
            if col_selects_set is not None and n not in col_selects_set:
                continue
            result.append(VTableField(name=n, type=t, kind=kind))
        return result

    columns: list[VTableField] = []
    kind_list = [VTableFieldKind.FEATURE, VTableFieldKind.LABEL, VTableFieldKind.ID]
    name_list = [schema.features, schema.labels, schema.ids]
    type_list = [schema.feature_types, schema.label_types, schema.id_types]
    for k, n, t in zip(kind_list, name_list, type_list):
        if not (kinds & k):
            continue
        res = _to_fields(k, n, t)
        columns.extend(res)
    if col_selects is not None and len(col_selects) > 0:
        columns_map = {c.name: c for c in columns}
        columns = [columns_map[name] for name in col_selects]

    return columns


def _parse_schemas(
    dd: DistData,  # type: ignore
    kinds: VTableFieldKind = VTableFieldKind.ALL,
    col_selects: List[str] = None,  # if None, load all cols
    partitions_order: List[str] = None,
) -> dict[str, VTableSchema]:
    assert kinds != 0, "column kind could not be 0"
    dd_type = dd.type.lower()
    assert dd_type in [
        DistDataType.VERTICAL_TABLE,
        DistDataType.INDIVIDUAL_TABLE,
    ], f"Unsupported DistData type {dd_type}"
    is_individual = dd_type == DistDataType.INDIVIDUAL_TABLE

    meta = IndividualTable() if is_individual else VerticalTable()
    dd.meta.Unpack(meta)

    pb_schemas = [meta.schema] if is_individual else meta.schemas
    assert len(pb_schemas) > 0, f"invliad schema {dd}"
    assert len(dd.data_refs) == len(pb_schemas)

    schemas: dict[str, VTableSchema] = {}
    for idx, pb_schema in enumerate(pb_schemas):
        dr = dd.data_refs[idx]
        fields = _parse_fields(pb_schema, kinds, col_selects)
        if len(fields) == 0:
            continue
        schemas[dr.party] = VTableSchema(
            party=dr.party,
            format=dr.format,
            uri=dr.uri,
            null_strs=list(dr.null_strs),
            fields=fields,
        )

    if partitions_order is not None:
        schemas = {k: schemas[k] for k in partitions_order}

    return schemas


def _to_distdata(parities: list[VTableSchema], line_count: int, system_info: SystemInfo) -> DistData:  # type: ignore
    is_individual = len(parities) == 1
    if is_individual:
        dd_type = DistDataType.INDIVIDUAL_TABLE
        pb_schema = parities[0].to_pb()
        meta = IndividualTable(schema=pb_schema, line_count=line_count)
    else:
        dd_type = DistDataType.VERTICAL_TABLE
        pb_schemas = [p.to_pb() for p in parities]
        meta = VerticalTable(schemas=pb_schemas, line_count=line_count)

    uri = parities[0].uri
    dd = DistData(
        name=uri,
        type=str(dd_type),
        system_info=system_info,
        data_refs=[
            DistData.DataRef(uri=p.uri, party=p.party, format=p.format)
            for p in parities
        ],
    )
    dd.meta.Pack(meta)
    return dd


class CompPartition:
    """
    Partition is PYUObject of pa.Table
    """

    def __init__(self, obj: PYUObject) -> None:
        self.object = obj

    @property
    def device(self):
        return self.object.device

    @property
    def data(self):
        return self.object.data

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

    def partition(self, key: int | str | PYU) -> CompPartition:
        if isinstance(key, int):
            keys = self.partitions.keys()
            key = next(iter(keys)) if key == 0 else list(keys)[key]
        elif isinstance(key, str):
            key = PYU(key)

        return self.partitions[key]

    def to_pandas(self, check_null=True) -> VDataFrame:
        def _to_pandas(t: pa.Table):
            if check_null:
                for c in t.column_names:
                    assert not pc.any(pc.is_null(t[c], nan_is_null=True)).as_py(), (
                        f"None or NaN contains in column {c},"
                        "pls fillna before use in training."
                    )
            # pyarrow's to_pandas() will change col type if col contains NULL
            # and trainning comp cannot handle NULL too,
            # so, failed if table contains NULL.
            return t.to_pandas()

        pandas = {}
        for pyu, pa_table in self.partitions.items():
            pandas[pyu] = pyu(_to_pandas)(pa_table.data)
        wait(pandas)
        return VDataFrame({p: partition(pandas[p]) for p in pandas})

    @staticmethod
    def from_pandas(
        v_data: VDataFrame,
    ) -> "CompVDataFrame":
        assert isinstance(v_data, VDataFrame)

        def _from_pandas(df: pd.DataFrame) -> pa.Table:
            fields = []
            for name, dtype in zip(df.columns, df.dtypes):
                pa_dtype = NP_DTYPE_TO_PA_DTYPE.get(dtype.type)
                if pa_dtype is None:
                    raise ValueError(f"unsupport type: {dtype}")
                fields.append((name, pa_dtype))
            schema = pa.schema(fields)
            res = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            return res

        partitions = {}
        for pyu, partition in v_data.partitions.items():
            data = pyu(_from_pandas)(partition.data)
            partitions[pyu] = CompPartition(data)

        return CompVDataFrame(partitions)

    @staticmethod
    def from_values(
        f_nd: FedNdarray,
        partition_columns: Dict[PYU, List[str]],
    ) -> "CompVDataFrame":
        assert isinstance(f_nd, FedNdarray)

        def _from_values(data: np.ndarray, columns: List[str]):
            pa_dtype = NP_DTYPE_TO_PA_DTYPE.get(data.dtype.type)
            if pa_dtype is None:
                raise ValueError(f"unsupport type: {data.dtype}")
            fields = [(c, pa_dtype) for c in columns]
            pandas = pd.DataFrame(data, columns=columns)
            return pa.Table.from_pandas(pandas, schema=pa.schema(fields))

        partitions = {}
        for pyu, partition in f_nd.partitions.items():
            assert pyu in partition_columns
            columns = partition_columns[pyu]
            data = pyu(_from_values)(partition, columns)
            partitions[pyu] = CompPartition(data)

        return CompVDataFrame(partitions)

    def _col_index(self, items) -> Dict[PYU, List[str]]:
        assert self.partitions, f"can not get index on empty dataframe"

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
            assert pyu is not None, f'Item {item} does not exist.'
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
            assert set(other.partitions.keys()).issubset(self.partitions.keys())
        else:
            assert set(other.partitions.keys()) == set(self.partitions.keys())

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
    def load(
        storage: Storage,
        dd: DistData,
        kinds: VTableFieldKind = VTableFieldKind.ALL,
        col_selects: list[str] = None,
        partitions_order: list[str] = None,
    ) -> "CompVDataFrame":
        schemas = _parse_schemas(dd, kinds, col_selects, partitions_order)
        partitions = {}
        for party, s in schemas.items():
            r = new_reader(
                PYU(party), storage, s.format, s.uri, s.null_strs, s.to_arrow(), 0
            )
            data_obj = r.read_all()
            p = CompPartition(data_obj)
            partitions[PYU(party)] = CompPartition(data_obj)
        return CompVDataFrame(partitions=partitions, system_info=dd.system_info)

    def dump(self, storage: Storage, uri: str, format: VTableFormat = VTableFormat.ORC) -> DistData:  # type: ignore
        format = format.value
        closes = []
        lines = []
        for pyu, party in self.partitions.items():
            w = new_writer(pyu, storage, format, uri)
            lines.append(w.write(party.data))
            closes.append(w.close())
        lines = reveal(lines)
        assert len(set(lines)) == 1, f"dataframe is not aligned, {lines}"
        wait(closes)

        schemas = []
        for pyu, obj in self.partitions.items():
            party = pyu.party
            fields = [VTableField.from_arrow(f) for f in obj.schema]
            schema = VTableSchema(party=party, format=format, uri=uri, fields=fields)
            schemas.append(schema)

        return _to_distdata(schemas, lines[0], self.system_info)


class CompVDataFrameReader:
    def __init__(
        self,
        storage: Storage,
        tracer: TimeTracer,
        dd: DistData,
        *,
        kinds: VTableFieldKind = VTableFieldKind.ALL,
        col_selects: list[str] = None,
        partitions_order: list[str] = None,
        batch_size: int = -1,
    ) -> None:
        schemas = _parse_schemas(dd, kinds, col_selects, partitions_order)
        readers: dict[str, IReader] = {}
        for p in schemas.values():
            pyu = PYU(p.party)
            reader = new_reader(
                pyu, storage, p.format, p.uri, p.null_strs, p.to_arrow(), batch_size
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
        assert len(eofs) == 1, f"dataframe not aligned"
        eof = next(iter(eofs)) == True

        return CompVDataFrame(datas, system_info=self._system_info) if not eof else None

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
        self._format = format.value
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
        assert df.partitions, "cannot write empty dataframe"
        self._writers = {}
        self._schemas = {}
        for pyu, obj in df.partitions.items():
            party = pyu.party
            w = new_writer(pyu, self._storage, self._format, self._uri)
            fields = [VTableField.from_arrow(f) for f in obj.schema]
            schema = VTableSchema(
                party=party, format=self._format, uri=self._uri, fields=fields
            )
            self._schemas[party] = schema
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
        assert (
            len(set(write_lines)) == 1
        ), f"dataframe is not aligned, lines: {write_lines}"
        self.line_count += write_lines[0]

    def close(self) -> None:
        if self._writers is not None:
            wait([w.close() for w in self._writers.values()])
            self._writers = None

    def dump_to(self, out: Output):
        assert self._schemas is not None
        schemas = list(self._schemas.values())
        dd = _to_distdata(schemas, self.line_count, self._system_info)
        out.data = dd
