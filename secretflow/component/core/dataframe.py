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
import inspect
import math
import traceback
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from secretflow_spec import (
    Output,
    Storage,
    VTable,
    VTableField,
    VTableFieldKind,
    VTableFormat,
    VTableSchema,
)
from secretflow_spec.v1.data_pb2 import DistData, SystemInfo

from secretflow.component.core.dist_data.vtable_utils import VTableUtils
from secretflow.data import FedNdarray, partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import PYU, PYUObject, reveal, wait
from secretflow.device.device.base import Device
from secretflow.utils.errors import InvalidArgumentError, InvalidStateError

from .dataframe_io import new_reader_proxy, new_writer_proxy
from .dist_data.base import IDumper
from .io import IReader, IWriter
from .types import TimeTracer


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
            raise InvalidArgumentError(f"pyu {pyu} not exist in comp data")
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
            # and training comp cannot handle NULL too,
            # so, failed if table contains NULL.
            for c in t.column_names:
                if pc.any(pc.is_null(t[c], nan_is_null=True)).as_py():
                    raise InvalidStateError(
                        message="None or NaN contains in column, pls fillna before use in training.",
                        detail={"column": c},
                    )

            return t.to_pandas()

        pandas_data = {}
        for pyu, pa_table in self.partitions.items():
            pandas_data[pyu] = pyu(_to_pandas)(pa_table.data)
        wait(pandas_data)
        return VDataFrame({p: partition(pandas_data[p]) for p in pandas_data})

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
                pa_dtype = VTableUtils.np_dtype_to_pa_dtype(dtype)
                field = VTableUtils.pa_field(name, pa_dtype, kinds[name])
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
            pa_dtype = VTableUtils.np_dtype_to_pa_dtype(data.dtype)
            if data.ndim == 1 and len(ts.fields) != 1:
                raise InvalidArgumentError(
                    f"1d array schema field should be one, but got {data.shape}"
                )
            elif data.ndim == 2 and data.shape[1] != len(ts.fields):
                raise InvalidArgumentError(
                    f"columns size mismatch, {data.shape}, {ts.fields.keys()}"
                )

            fields = []
            for f in ts.fields.values():
                field = VTableUtils.pa_field(f.name, pa_dtype, f.kind)
                fields.append(field)
            schema = pa.schema(fields)
            df = pd.DataFrame(data, columns=schema.names)
            res = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            return res

        partitions = {}
        for pyu, partition in f_nd.partitions.items():
            if pyu.party not in schemas:
                raise InvalidArgumentError(f"pyu.party {pyu.party} not in schemas")
            data = pyu(_from_values)(partition, schemas[pyu.party])

            partitions[pyu] = CompPartition(data)

        return CompVDataFrame(partitions, system_info)

    def _col_index(self, items) -> Dict[PYU, List[str]]:
        if not self.partitions:
            raise RuntimeError(f"can not get index on empty DataFrame")

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
                raise InvalidArgumentError(
                    f'Item {item} does not exist in columns_to_party.'
                )
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
                raise InvalidArgumentError(
                    f"other DataFrame's partition {other.partitions.keys()} not in self partition {self.partitions.keys()}"
                )
        else:
            if set(other.partitions.keys()) != set(self.partitions.keys()):
                raise InvalidArgumentError(
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
            pa_schema = VTableUtils.to_arrow_schema(p.schema)
            r = new_reader_proxy(
                PYU(party), storage, p.format, p.uri, pa_schema, null_strs=p.null_strs
            )
            data_obj = r.read_all()
            p = CompPartition(data_obj)
            partitions[PYU(party)] = CompPartition(data_obj)
        ret = CompVDataFrame(partitions=partitions, system_info=dd.system_info)
        if not math.prod(ret.shape):
            raise InvalidStateError("empty dataset is not allowed")
        return ret

    def dump(
        self, storage: Storage, uri: str, format: VTableFormat = VTableFormat.ORC
    ) -> DistData:
        if not self.partitions:
            raise InvalidStateError("can not dump empty dataframe")

        closes = []
        lines = []
        for pyu, party in self.partitions.items():
            w = new_writer_proxy(pyu, storage, format, uri, schema=party.schema)
            lines.append(w.write(party.data))
            closes.append(w.close())
        lines = reveal(lines)
        if len(set(lines)) != 1:
            raise InvalidStateError(
                message="DataFrame is not aligned", detail={"lines": lines}
            )
        if lines[0] <= 0:
            raise InvalidStateError("empty dataset is not allowed")
        wait(closes)

        schemas = {}
        for pyu, obj in self.partitions.items():
            schemas[pyu.party] = VTableUtils.from_arrow_schema(obj.schema)

        vtbl = VTable.from_output_uri(
            uri, schemas, line_count=lines[0], system_info=self.system_info
        )
        return vtbl.to_distdata()


class CompVDataFrameReader:
    def __init__(
        self,
        storage: Storage,
        tracer: TimeTracer,
        dd: DistData | VTable,
        batch_size: int | None = None,
    ) -> None:
        vtbl = dd if isinstance(dd, VTable) else VTable.from_distdata(dd)
        readers: dict[str, IReader] = {}
        for p in vtbl.parties.values():
            pyu = PYU(p.party)
            pa_schema = VTableUtils.to_arrow_schema(p.schema)
            reader = new_reader_proxy(
                pyu,
                storage,
                p.format,
                p.uri,
                pa_schema,
                null_strs=p.null_strs,
                batch_size=batch_size,
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
        lines = []
        for p, r in self._readers.items():
            (data_obj, line_obj) = r.read_next()
            datas[p] = CompPartition(data_obj)
            lines.append(line_obj)

        lines = set(reveal(lines))
        if len(lines) != 1:
            raise InvalidStateError(
                message="DataFrame is not aligned", detail={"lines": lines}
            )
        eof = next(iter(lines)) == True

        if eof:
            return None

        ret = CompVDataFrame(datas, system_info=self._system_info)
        if not math.prod(ret.shape):
            raise InvalidStateError("empty dataset is not allowed")
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
            raise InvalidStateError("cannot write empty DataFrame")
        self._writers = {}
        self._schemas = {}
        for pyu, obj in df.partitions.items():
            party = pyu.party
            w = new_writer_proxy(
                pyu, self._storage, self._format, self._uri, schema=obj.schema
            )
            self._schemas[party] = VTableUtils.from_arrow_schema(obj.schema)
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
            raise InvalidStateError(
                message="DataFrame is not aligned", detail={"lines": write_lines}
            )
        self.line_count += write_lines[0]

    def close(self) -> None:
        if self._writers is not None:
            wait([w.close() for w in self._writers.values()])
            self._writers = None

    def dump(self) -> DistData:
        if self._schemas is None or self.line_count <= 0:
            raise InvalidStateError("empty dataset is not allowed")

        vtbl = VTable.from_output_uri(
            self._uri,
            self._schemas,
            line_count=self.line_count,
            format=self._format,
            system_info=self._system_info,
        )
        return vtbl.to_distdata()

    def dump_to(self, out: Output):
        out.data = self.dump()


import logging


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

    # all columns must belong to receiver
    receiver_cols_set = set(tbl.parties[pyu.party].columns)
    addition_cols_set = set()
    for f in addition_cols:
        if f in addition_cols_set or f == pred_name:
            raise InvalidStateError(
                message="The column cannot be used as both label and feature",
                detail={"column": f},
            )
        if f not in receiver_cols_set:
            raise InvalidStateError(
                message="The saved feature can only belong to receiver",
                detail={
                    "column": f,
                    "receiver": pyu.party,
                    "receiver_columns": receiver_cols_set,
                },
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
                raise InvalidStateError(
                    message="pyu not in pred.partitions",
                    detail={"pyu": pyu.party, "partitions": pred.partitions},
                )
            if isinstance(pred, FedNdarray):
                logging.info('from value')
                pred_df = CompVDataFrame.from_values(
                    pred, pred_schemas, feature_dataset.system_info
                )
            elif isinstance(pred, VDataFrame):
                logging.info('from pandas')
                pred_df = CompVDataFrame.from_pandas(
                    pred, pred_schemas, feature_dataset.system_info
                )

            if addition_cols:
                addition_df = batch[addition_cols]
                assert len(addition_df.partitions) == 1
                if pyu not in addition_df.partitions:
                    raise InvalidStateError(
                        message="pyu not in addition_df.partitions",
                        detail={"pyu": pyu.party, "partitions": addition_df.partitions},
                    )

                logging.info(f'pred_df: {type(pred_df)}')
                logging.info(f'pred_df: {pred_df.shape}')
                logging.info(f'addition_df: {addition_df.shape}')

                out_df = pred_df.concat(addition_df, axis=1)
            else:
                out_df = pred_df
            writer.write(out_df)

    return writer.dump()
