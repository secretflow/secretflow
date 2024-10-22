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
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import orc

from secretflow.component.component import CompEvalContext
from secretflow.component.storage import ComponentStorage
from secretflow.data import FedNdarray, partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import PYU, PYUObject, proxy
from secretflow.device.driver import reveal, wait
from secretflow.error_system.exceptions import (
    CompEvalError,
    DataFormatError,
    NotSupportedError,
)
from secretflow.spec.v1.data_pb2 import DistData, SystemInfo, TableSchema

from .data_utils import (
    REVERSE_DATA_TYPE_MAP,
    DataSetFormatSupported,
    DistDataInfo,
    DistDataType,
    TableMetaWrapper,
    extract_data_infos,
    is_same_np_type,
)

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


def _get_datainfo(
    ctx,
    db: DistData,
    *,
    partitions_order: List[str] = None,
    load_features: bool = False,
    load_labels: bool = False,
    load_ids: bool = False,
    col_selects: List[str] = None,  # if None, load all cols
    col_excludes: List[str] = None,
) -> Dict[PYU, DistDataInfo]:
    data_infos = extract_data_infos(
        db,
        partitions_order=partitions_order,
        load_features=load_features,
        load_labels=load_labels,
        load_ids=load_ids,
        col_selects=col_selects,
        col_excludes=col_excludes,
    )

    data_infos = {PYU(p): data_infos[p] for p in data_infos}

    file_metas = {}
    with ctx.tracer.trace_io():
        for pyu in data_infos:
            uri = data_infos[pyu].uri
            file_metas[pyu] = pyu(lambda uri=uri: ctx.comp_storage.get_file_meta(uri))()
        file_metas = reveal(file_metas)
    logging.info(
        f"try load VDataFrame, file infos {data_infos}, file meta {file_metas}"
    )

    return data_infos


@proxy(device_object_type=PYUObject)
class OrcWriter:
    def __init__(
        self,
        comp_storage: ComponentStorage,
        uri: str,
        stripe_size=64 * 1024 * 1024,
    ) -> None:
        self._io_writer = comp_storage.get_writer(uri)
        self._writer = orc.ORCWriter(
            self._io_writer,
            stripe_size=stripe_size,
        )
        self._schema = None

    def write(self, data: pa.Table) -> int:
        if self._schema is None:
            self._schema = data.schema

        # has same columns & types
        assert set(self._schema) == set(data.schema)
        if self._schema != data.schema:
            # but with different order. reorder input.
            data = data.select([s.name for s in self._schema])
        self._writer.write(data)
        return data.shape[0]

    def close(self) -> None:
        self._writer.close()
        self._io_writer.close()


@proxy(device_object_type=PYUObject)
class DistDataReader:
    def __init__(
        self,
        comp_storage: ComponentStorage,
        info: DistDataInfo,
        batch_size: int = 50000,
    ) -> None:
        if info.format not in DataSetFormatSupported:
            raise NotSupportedError.not_supported_file_format(
                f"not support file format {info.format}"
            )
        if info.format == "csv":
            self._init_csv(comp_storage, info, batch_size)
        elif info.format == "orc":
            self._init_orc(comp_storage, info, batch_size)
        else:
            raise AttributeError(f"not support file format {info.format}")

    def _init_csv(
        self, comp_storage: ComponentStorage, info: DistDataInfo, batch_size: int
    ):
        from .data_utils import NP_DTYPE_TO_DUCKDB_DTYPE

        duck_dtype = {c: NP_DTYPE_TO_DUCKDB_DTYPE[info.dtypes[c]] for c in info.dtypes}
        na_values = info.null_strs if info.null_strs else []
        self._io_reader = comp_storage.get_reader(info.uri)
        conn = duckdb.connect(":memory:")
        csv_db = conn.read_csv(self._io_reader, dtype=duck_dtype, na_values=na_values)
        col_list = [duckdb.ColumnExpression(c) for c in info.dtypes]
        csv_select = csv_db.select(*col_list)
        self._reader = csv_select.fetch_arrow_reader(batch_size=batch_size)
        self._columns = list(info.dtypes.keys())

    def _init_orc(
        self, comp_storage: ComponentStorage, info: DistDataInfo, batch_size: int
    ):
        class OrcFileReader:
            def __init__(
                self, orc_file: orc.ORCFile, columns: list[str], batch_size: int
            ) -> None:
                if orc_file.nstripes <= 0:
                    raise DataFormatError.empty_dataset("empty orc file is not allowed")
                self.nstripes = orc_file.nstripes
                self.columns = columns
                self.current_stripe = 0
                self.current_idx = 0
                self.orc_file = orc_file
                self.batch_size = batch_size

            def read_all(self) -> pa.Table:
                return self.orc_file.read(columns=self.columns)

            def read_next_batch(self) -> pa.Table:
                if self.current_stripe == self.nstripes:
                    raise StopIteration
                batches = []
                batches_lines = 0
                while (
                    batches_lines < self.batch_size
                    and self.current_stripe < self.nstripes
                ):
                    batch = self.orc_file.read_stripe(
                        self.current_stripe, columns=self.columns
                    )
                    read_lines = min(
                        self.batch_size - batches_lines,
                        batch.num_rows - self.current_idx,
                    )
                    batches.append(
                        batch[self.current_idx : self.current_idx + read_lines]
                    )
                    batches_lines += read_lines
                    if self.current_idx + read_lines == batch.num_rows:
                        self.current_stripe += 1
                        self.current_idx = 0
                    else:
                        self.current_idx += read_lines

                return pa.Table.from_batches(batches)

        self._io_reader = comp_storage.get_reader(info.uri)
        orc_file = orc.ORCFile(self._io_reader)
        orc_schema = orc_file.schema
        for c, t in info.dtypes.items():
            pa_type = orc_schema.field(c)
            if pa_type is None:
                raise DataFormatError.header_field_not_found(
                    f"can not find {c} column in file {info.uri}"
                )
            np_type = pa_type.type.to_pandas_dtype()
            if not is_same_np_type(np_type, t):
                raise DataFormatError.data_field_mismatch(
                    f"type missmatch, type in distdata: {t}, "
                    f"type in file {info.uri}: pa_type {pa_type} -> np_type {np_type}"
                )

        self._columns = list(info.dtypes.keys())
        self._reader = OrcFileReader(orc_file, self._columns, batch_size)

    def fetch_all(self) -> pa.Table:
        table = self._reader.read_all()
        # reorder columns
        return table.select(self._columns)

    def fetch_next_batch(self) -> Tuple[pa.Table, bool]:
        try:
            batch = self._reader.read_next_batch()
            if isinstance(batch, pa.Table):
                table = batch
            else:
                table = pa.Table.from_batches([batch])
            # reorder columns
            return table.select(self._columns), False
        except StopIteration:
            return None, True


@dataclass
class CompTable:
    # pa.Table
    data: PYUObject
    # classification of columns
    id_cols: List[str]
    feature_cols: List[str]
    label_cols: List[str]

    def _assert_cols(self):
        columns = self.columns
        all_cols = set(self.id_cols + self.feature_cols + self.label_cols)
        if set(columns) != all_cols:
            raise DataFormatError.data_field_mismatch(
                f"classification of columns {all_cols} is mismatch to data {columns}, "
                "pls update classification first"
            )

    @property
    def shape(self) -> Tuple[int, int]:
        pyu = self.data.device
        return reveal(pyu(lambda t: t.shape)(self.data))

    @property
    def dtypes(self) -> Dict[str, np.dtype]:
        pyu = self.data.device
        dtypes = pyu(lambda t: {c.name: c.type.to_pandas_dtype() for c in t.schema})(
            self.data
        )
        return reveal(dtypes)

    @property
    def columns(self) -> List[str]:
        pyu = self.data.device
        return reveal(pyu(lambda t: t.column_names)(self.data))

    def __getitem__(self, items) -> "CompTable":
        self._assert_cols()
        set_items = set(items)
        assert set_items.issubset(
            set(self.id_cols + self.feature_cols + self.label_cols)
        )

        def _select(t, i):
            ret = t.select(i)
            return ret

        new_data = self.data.device(_select)(self.data.data, items)
        new_ids = [c for c in self.id_cols if c in set_items]
        new_features = [c for c in self.feature_cols if c in set_items]
        new_label = [c for c in self.label_cols if c in set_items]

        return CompTable(new_data, new_ids, new_features, new_label)

    def drop(self, items) -> "CompTable":
        self._assert_cols()
        items = set(items)
        assert items.issubset(set(self.id_cols + self.feature_cols + self.label_cols))

        def _drop(t, i):
            return t.drop(i)

        new_data = self.data.device(_drop)(self.data.data, items)
        new_ids = [c for c in self.id_cols if c not in items]
        new_features = [c for c in self.feature_cols if c not in items]
        new_label = [c for c in self.label_cols if c not in items]

        return CompTable(new_data, new_ids, new_features, new_label)

    def concat(self, other: "CompTable", axis: int) -> "CompTable":
        self._assert_cols()
        other._assert_cols()
        assert axis in [0, 1]
        assert self.data.device == other.data.device

        if axis == 0:
            assert self.shape[1] == other.shape[1]
            assert self.columns == other.columns

            def _concat(t1: pa.Table, t2: pa.Table):
                return pa.concat_tables([t1, t2])

            new_ids = self.id_cols
            new_features = self.feature_cols
            new_label = self.label_cols

        else:
            assert self.shape[0] == other.shape[0]
            assert len(set(self.columns).intersection(set(other.columns))) == 0

            def _concat(dest: pa.Table, source: pa.Table):
                for i in range(source.shape[1]):
                    dest = dest.append_column(source.field(i), source.column(i))
                return dest

            new_ids = self.id_cols + other.id_cols
            new_features = self.feature_cols + other.feature_cols
            new_label = self.label_cols + other.label_cols

        new_data = self.data.device(_concat)(self.data.data, other.data.data)

        return CompTable(new_data, new_ids, new_features, new_label)

    def to_table_schema(self) -> TableSchema:
        self._assert_cols()
        dtypes = self.dtypes
        ids = [k for k in dtypes if k in set(self.id_cols)]
        labels = [k for k in dtypes if k in set(self.label_cols)]
        features = [k for k in dtypes if k in set(self.feature_cols)]

        return TableSchema(
            ids=ids,
            id_types=[REVERSE_DATA_TYPE_MAP[dtypes[k]] for k in ids],
            labels=labels,
            label_types=[REVERSE_DATA_TYPE_MAP[dtypes[k]] for k in labels],
            features=features,
            feature_types=[REVERSE_DATA_TYPE_MAP[dtypes[k]] for k in features],
        )


@dataclass
class CompDataFrame:
    partitions: Dict[PYU, CompTable]
    system_info: SystemInfo

    @staticmethod
    def from_distdata(
        ctx,
        db: DistData,
        *,
        partitions_order: List[str] = None,
        load_features: bool = False,
        load_labels: bool = False,
        load_ids: bool = False,
        col_selects: List[str] = None,  # if None, load all cols
        col_excludes: List[str] = None,
    ) -> "CompDataFrame":
        data_infos = _get_datainfo(
            ctx,
            db,
            partitions_order=partitions_order,
            load_features=load_features,
            load_labels=load_labels,
            load_ids=load_ids,
            col_selects=col_selects,
            col_excludes=col_excludes,
        )
        partitions = {}
        waits = []
        with ctx.tracer.trace_io():
            for pyu, info in data_infos.items():
                data = DistDataReader(ctx.comp_storage, info, device=pyu).fetch_all()
                waits.append(data)
                table = CompTable(
                    data, info.id_cols, info.feature_cols, info.label_cols
                )
                partitions[pyu] = table
            wait(waits)

        ret = CompDataFrame(partitions, db.system_info)
        shape = ret.shape
        logging.info(f"loaded VDataFrame, shape {shape}")
        if not math.prod(shape):
            raise DataFormatError.empty_dataset(f"empty dataset {shape} is not allowed")
        return ret

    def to_distdata(
        self,
        ctx,
        uri: str,
        stripe_size=64 * 1024 * 1024,
    ) -> DistData:
        if not self.partitions:
            raise DataFormatError.empty_dataset("can not dump empty DataFrame")

        dist_type = (
            str(DistDataType.VERTICAL_TABLE)
            if len(self.partitions) > 1
            else str(DistDataType.INDIVIDUAL_TABLE)
        )
        ret = DistData(
            name=uri,
            type=dist_type,
            system_info=self.system_info,
            data_refs=[
                DistData.DataRef(uri=uri, party=p.party, format="orc")
                for p in self.partitions
            ],
        )
        meta = TableMetaWrapper(
            self.shape[0],
            {p.party: t.to_table_schema() for p, t in self.partitions.items()},
        )
        if len(self.partitions) > 1:
            ret.meta.Pack(meta.to_vertical_table())
        else:
            ret.meta.Pack(meta.to_individual_table())

        def _to_orc(comp_storage: ComponentStorage, uri: str, table: pa.Table) -> None:
            with comp_storage.get_writer(uri) as w:
                orc.write_table(
                    table,
                    w,
                    stripe_size=stripe_size,
                )

        with ctx.tracer.trace_io():
            parties_shapes = self.partition_shapes

            waits = []
            for pyu, table in self.partitions.items():
                waits.append(pyu(_to_orc)(ctx.comp_storage, uri, table.data.data))
            wait(waits)

        file_metas = {}
        for pyu in self.partitions.keys():
            file_metas[pyu] = pyu(lambda: ctx.comp_storage.get_file_meta(uri))()
        file_metas = reveal(file_metas)
        logging.info(
            f"dumped VDataFrame, file uri {uri}, shape {parties_shapes}, file meta {file_metas}"
        )

        return ret

    def copy(self) -> "CompDataFrame":
        return CompDataFrame(copy.deepcopy(self.partitions), self.system_info)

    # for training comp
    def to_pandas(self, check_null=True) -> VDataFrame:
        pandas = {}

        def _to_pandas(t):
            if not check_null:
                return t.to_pandas()

            # pyarrow's to_pandas() will change col type if col contains NULL
            # and training comp cannot handle NULL too,
            # so, failed if table contains NULL.
            for c in t.column_names:
                if pc.any(pc.is_null(t[c], nan_is_null=True)).as_py():
                    raise DataFormatError.none_filed_in_column(
                        f"None or NaN contains in column {c},"
                        "pls fill na before use in training."
                    )

            return t.to_pandas()

        for pyu, pa_table in self.partitions.items():
            pandas[pyu] = pyu(_to_pandas)(pa_table.data)
        wait(pandas)
        return VDataFrame({p: partition(pandas[p]) for p in pandas})

    @staticmethod
    def from_pandas(
        v_data: VDataFrame,
        system_info: SystemInfo,
        id_cols: List[str] = None,
        label_cols: List[str] = None,
    ) -> "CompDataFrame":
        assert isinstance(v_data, VDataFrame)
        partitions = {}
        id_cols = set(id_cols) if id_cols else {}
        label_cols = set(label_cols) if label_cols else {}
        partition_columns = v_data.partition_columns

        def _from_pandas(df: pd.DataFrame) -> pa.Table:
            fields = []
            for name, dtype in zip(df.columns, df.dtypes):
                pa_dtype = NP_DTYPE_TO_PA_DTYPE.get(dtype.type)
                if pa_dtype is None:
                    raise ValueError(f"unsupported type: {dtype}")
                fields.append((name, pa_dtype))
            schema = pa.schema(fields)
            res = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            return res

        for pyu, partition in v_data.partitions.items():
            data = pyu(_from_pandas)(partition.data)
            columns = partition_columns[pyu]
            partitions[pyu] = CompTable(
                data,
                [c for c in columns if c in id_cols],
                [c for c in columns if c not in id_cols and c not in label_cols],
                [c for c in columns if c in label_cols],
            )

        return CompDataFrame(partitions, system_info)

    @staticmethod
    def from_values(
        f_nd: FedNdarray,
        system_info: SystemInfo,
        partition_columns: Dict[PYU, List[str]],
        id_cols: List[str] = None,
        label_cols: List[str] = None,
    ) -> "CompDataFrame":
        assert isinstance(f_nd, FedNdarray)
        partitions = {}
        id_cols = set(id_cols) if id_cols else {}
        label_cols = set(label_cols) if label_cols else {}

        def _from_values(data: np.ndarray, columns: List[str]):
            pa_dtype = NP_DTYPE_TO_PA_DTYPE.get(data.dtype.type)
            if pa_dtype is None:
                raise ValueError(f"unsupport type: {data.dtype}")
            fields = [(c, pa_dtype) for c in columns]
            pandas = pd.DataFrame(data, columns=columns)
            return pa.Table.from_pandas(pandas, schema=pa.schema(fields))

        for pyu, partition in f_nd.partitions.items():
            if pyu not in partition_columns:
                raise CompEvalError.party_check_failed(
                    f"partition_columns does not contain pyu {pyu}"
                )
            columns = partition_columns[pyu]
            data = pyu(_from_values)(partition, columns)
            partitions[pyu] = CompTable(
                data,
                [c for c in columns if c in id_cols],
                [c for c in columns if c not in id_cols and c not in label_cols],
                [c for c in columns if c in label_cols],
            )

        return CompDataFrame(partitions, system_info)

    def _col_index(self, items) -> Dict[PYU, List[str]]:
        if not self.partitions:
            raise DataFormatError.empty_dataset(f"can not get index on empty DataFrame")

        if hasattr(items, "tolist"):
            items = items.tolist()
        if not isinstance(items, (list, tuple)):
            items = [items]

        columns_to_party = {}
        for p, cols in self.partition_columns.items():
            columns_to_party.update({c: p for c in cols})

        ret = defaultdict(list)
        for item in items:
            pyu = columns_to_party.get(item, None)
            if pyu is None:
                raise CompEvalError(f'Item {item} does not exist in columns_to_party.')
            ret[pyu].append(item)

        # keep party order
        return {p: ret[p] for p in self.partitions.keys() if p in ret}

    def __getitem__(self, items) -> "CompDataFrame":
        items = self._col_index(items)

        partitions = {}
        for p, i in items.items():
            t = self.partitions[p]
            partitions[p] = t[i]

        wait([t.data for t in partitions.values()])
        return CompDataFrame(partitions, self.system_info)

    def drop(self, items) -> "CompDataFrame":
        items = self._col_index(items)

        partitions = copy.deepcopy(self.partitions)
        for p, i in items.items():
            t = self.partitions[p].drop(i)
            if len(t.columns) > 0:
                partitions[p] = t
            else:
                del partitions[p]
        return CompDataFrame(partitions, self.system_info)

    def data(self, pyu: PYU) -> PYUObject:
        if pyu not in self.partitions:
            raise CompEvalError(f"pyu {pyu} not exist in comp data")
        return self.partitions[pyu].data

    def set_data(self, data: PYUObject) -> None:
        assert isinstance(data, PYUObject)
        pyu = data.device
        if pyu not in self.partitions:
            raise CompEvalError(f"pyu {pyu} not exist in comp data")
        self.partitions[pyu].data = data

    def has_data(self, pyu: PYU) -> bool:
        return pyu in self.partitions

    def concat(self, other: "CompDataFrame", axis: int) -> "CompDataFrame":
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
        return CompDataFrame(ret_partitions, self.system_info)

    @property
    def columns(self) -> List[str]:
        partition_columns = self.partition_columns
        return [c for pc in partition_columns.values() for c in pc]

    @property
    def partition_columns(self) -> Dict[PYU, List[str]]:
        partition_columns = {}
        for pyu, table in self.partitions.items():
            partition_columns[pyu] = table.columns
        return partition_columns

    @property
    def dtypes(self) -> Dict[str, np.dtype]:
        partition_dtypes = self.partition_dtypes
        return {
            col: dtype for pt in partition_dtypes.values() for col, dtype in pt.items()
        }

    @property
    def partition_dtypes(self) -> Dict[PYU, Dict[str, np.dtype]]:
        partition_dtypes = {}
        for pyu, table in self.partitions.items():
            partition_dtypes[pyu] = table.dtypes
        return partition_dtypes

    @property
    def shape(self) -> Tuple[int, int]:
        partition_shapes = self.partition_shapes
        partition_shapes = list(partition_shapes.values())
        return partition_shapes[0][0], sum([shape[1] for shape in partition_shapes])

    @property
    def partition_shapes(self) -> Dict[PYU, Tuple[int, int]]:
        partition_shapes = {}
        for pyu, table in self.partitions.items():
            partition_shapes[pyu] = table.shape
        if len(set([s[0] for s in partition_shapes.values()])) != 1:
            raise CompEvalError(
                f"number of samples must be equal across all devices, got {partition_shapes}"
            )
        return partition_shapes


@dataclass
class StreamingReader:
    ctx: CompEvalContext
    readers: Dict[PYU, DistDataReader]
    data_infos: Dict[PYU, DistDataInfo]
    system_info: SystemInfo

    def fetch_next(self) -> CompDataFrame:
        assert self.readers

        partitions = {}
        with self.ctx.tracer.trace_io():
            eofs = []
            for p, r in self.readers.items():
                data, eof = r.fetch_next_batch()
                eofs.append(eof)
                data_info = self.data_infos[p]
                partitions[p] = CompTable(
                    data,
                    data_info.id_cols,
                    data_info.feature_cols,
                    data_info.label_cols,
                )
            eofs = set(reveal(eofs))
            if len(eofs) != 1:
                raise DataFormatError.dataset_not_aligned(
                    f"streaming read err, dataset not aligned"
                )
            if eofs.pop():
                raise StopIteration

            ret = CompDataFrame(partitions, self.system_info)
            logging.info(f"streaming loaded VDataFrame, shape {ret.shape}")
            if not math.prod(ret.shape):
                raise DataFormatError.empty_dataset(
                    f"empty dataset {ret.shape} is not allowed"
                )

        return ret

    def __iter__(self) -> "StreamingReader":
        return self

    def __next__(self) -> CompDataFrame:
        return self.fetch_next()

    """
    @staticmethod
    def streaming_by_column() -> "StreamingReader":
        # TODO: col streaming, but is this necessary ??
        raise NotImplemented
    """

    @staticmethod
    def from_distdata(
        ctx,
        db: DistData,
        *,
        partitions_order: List[str] = None,
        load_features: bool = False,
        load_labels: bool = False,
        load_ids: bool = False,
        col_selects: List[str] = None,  # if None, load all cols
        col_excludes: List[str] = None,
        batch_size: int = 50000,
    ) -> "StreamingReader":
        data_infos = _get_datainfo(
            ctx,
            db,
            partitions_order=partitions_order,
            load_features=load_features,
            load_labels=load_labels,
            load_ids=load_ids,
            col_selects=col_selects,
            col_excludes=col_excludes,
        )

        readers = {}
        for p, i in data_infos.items():
            readers[p] = DistDataReader(
                ctx.comp_storage,
                info=i,
                batch_size=batch_size,
                device=p,
            )

        return StreamingReader(ctx, readers, data_infos, db.system_info)

    @staticmethod
    def from_data_infos(
        ctx,
        data_infos: Dict[str, DistDataInfo],
        system_info: Any,
        *,
        batch_size: int = 50000,
    ) -> "StreamingReader":
        readers = {}
        data_infos = {PYU(p): data_infos[p] for p in data_infos}
        for p, i in data_infos.items():
            readers[p] = DistDataReader(
                ctx.comp_storage,
                info=i,
                batch_size=batch_size,
                device=p,
            )

        return StreamingReader(ctx, readers, data_infos, system_info)


class StreamingWriter:
    def __init__(
        self,
        ctx,
        uri: str,
        data_infos: Dict[PYU, DistDataInfo] = None,
    ) -> None:
        self.ctx = ctx
        self.writers = None
        self.uri = uri
        self.meta = None
        self.system_info = None
        self.line_count = 0
        self.closed = False
        self.meta = None
        self.write_data = False
        self.data_infos = data_infos
        if data_infos is not None:
            self.meta = TableMetaWrapper.from_dist_info(data_infos)
            self._init_writer()

    def _init_writer(self) -> None:
        self.writers = {}
        for party in self.meta.schema_map.keys():
            self.writers[party] = OrcWriter(
                self.ctx.comp_storage, self.uri, device=PYU(party)
            )

    def write(self, data: CompDataFrame) -> None:
        if not data.partitions:
            raise DataFormatError.empty_dataset(f"cannot write empty dataframe")

        assert not self.closed
        if self.writers is None:
            self.meta = TableMetaWrapper(
                0,
                {p.party: t.to_table_schema() for p, t in data.partitions.items()},
            )
            self._init_writer()

        with self.ctx.tracer.trace_io():
            write_lines = []
            for pyu, table in data.partitions.items():
                write_lines.append(self.writers[pyu.party].write(table.data.data))
            write_lines = reveal(write_lines)
            if len(set(write_lines)) != 1:
                raise DataFormatError.dataset_not_aligned(
                    f"input data is not aligned, lines: {write_lines}"
                )
            self.line_count += write_lines[0]
            self.write_data = True

        frame_schema = {
            p.party: t.to_table_schema() for p, t in data.partitions.items()
        }
        assert set(frame_schema.keys()) == set(self.meta.schema_map.keys())
        for party, schema in frame_schema.items():
            meta_schema = self.meta.schema_map[party]
            assert set(schema.ids) == set(meta_schema.ids)
            assert set(schema.labels) == set(meta_schema.labels)
            assert set(schema.features) == set(meta_schema.features)

    def close(self) -> None:
        if not self.write_data:
            for p, w in self.writers.items():
                pyu = PYU(p)
                w.write(
                    pa.Table.from_pylist(
                        [],
                        schema=pa.schema(
                            [
                                pa.field(
                                    name,
                                    NP_DTYPE_TO_PA_DTYPE[
                                        self.data_infos[pyu].dtypes[name]
                                    ],
                                )
                                for name in self.data_infos[pyu].feature_cols
                                + self.data_infos[pyu].label_cols
                                + self.data_infos[pyu].id_cols
                            ]
                        ),
                    )
                )

        if self.writers is not None:
            wait([w.close() for w in self.writers.values()])
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def to_distdata(self) -> DistData:
        assert self.closed and (self.writers is not None or self.line_count == 0)
        ret = DistData(
            name=self.uri,
            type=self.meta.dist_table_type(),
            system_info=self.system_info,
            data_refs=[
                DistData.DataRef(uri=self.uri, party=p, format="orc")
                for p in self.meta.schema_map.keys()
            ],
        )
        order = [p for p in self.meta.schema_map]
        self.meta.line_count = self.line_count
        ret.meta.Pack(self.meta.to_dist_data(order))

        return ret


def load_table_select_and_exclude_pair(
    ctx,
    db: DistData,
    load_features: bool = False,
    load_labels: bool = False,
    load_ids: bool = False,
    col_selects: List[str] = None,  # if None, load all cols
):
    """
    Load two tables, one is the selected, another is the complement.
    """

    if not col_selects:
        raise DataFormatError.empty_dataset(f"cannot load empty train set")
    # Only used to verify col_selects
    _ = extract_data_infos(
        db,
        load_features=load_features,
        load_labels=load_labels,
        load_ids=load_ids,
        col_selects=col_selects,
    )
    remain_infos = extract_data_infos(
        db,
        load_features=True,
        load_labels=True,
        load_ids=True,
        col_excludes=col_selects,
    )
    x = CompDataFrame.from_distdata(
        ctx,
        db,
        load_features=True,
        load_labels=True,
        load_ids=True,
    )
    remain_cols = [c for pi in remain_infos.values() for c in pi.dtypes.keys()]
    if remain_cols:
        trans_x = x[col_selects]
        remain_x = x[remain_cols]
    else:
        trans_x = x
        remain_x = None

    return trans_x, remain_x


def save_prediction_dd(
    ctx,
    uri: str,
    pyu: PYU,
    batch_pred: Callable,
    pred_name: str,
    pred_features: List[str],
    pred_partitions_order: List[str],
    feature_dataset: DistData,
    saved_features: List[str],
    saved_labels: List[str],
    save_ids: bool,
    check_null: bool = True,
) -> DistData:
    addition_cols = []

    def _named_features(features_name: List[str]):
        for f in features_name:
            if f in addition_cols or f == pred_name:
                raise DataFormatError.feature_intersection_with_label(
                    f"do not select {f} as saved feature, repeated with id or label"
                )

        infos = extract_data_infos(
            feature_dataset,
            load_ids=True,
            load_features=True,
            load_labels=True,
            col_selects=features_name,
        )
        if len(infos) != 1 or pyu.party not in infos:
            raise CompEvalError.party_check_failed(
                f"The saved feature {features_name} can only belong to receiver party {pyu.party}, got {infos.keys()}"
            )

        addition_cols.extend(features_name)

    if save_ids:
        infos = extract_data_infos(feature_dataset, load_ids=True)
        if pyu.party not in infos:
            raise CompEvalError.party_check_failed(
                f"can not find id col for receiver party {pyu.party}, {infos}"
            )
        saved_ids = list(infos[pyu.party].dtypes.keys())
        _named_features(saved_ids)

    if saved_labels:
        _named_features(saved_labels)

    if saved_features:
        _named_features(saved_features)

    if addition_cols:
        reader_features = pred_features + [
            a for a in addition_cols if a not in pred_features
        ]
    else:
        reader_features = pred_features

    if pred_partitions_order is not None and pyu.party not in pred_partitions_order:
        pred_partitions_order.append(pyu.party)

    reader = StreamingReader.from_distdata(
        ctx,
        feature_dataset,
        partitions_order=pred_partitions_order,
        load_features=True,
        load_ids=True,
        load_labels=True,
        col_selects=reader_features,
    )

    writer = StreamingWriter(ctx, uri)
    with writer:
        for batch in reader:
            pred_batch = batch[pred_features].to_pandas(check_null)
            pred = batch_pred(pred_batch)
            assert len(pred.partitions) == 1
            if pyu not in pred.partitions:
                raise CompEvalError.party_check_failed(
                    f"pyu [{pyu}] is not in pred.partitions [{pred.partitions}]"
                )
            if isinstance(pred, FedNdarray):
                pred = CompDataFrame.from_values(
                    pred,
                    feature_dataset.system_info,
                    {pyu: [pred_name]},
                    label_cols=[pred_name],
                )
            elif isinstance(pred, VDataFrame):
                pred = CompDataFrame.from_pandas(
                    pred,
                    feature_dataset.system_info,
                    label_cols=[pred_name],
                )

            if addition_cols:
                addition_df = batch[addition_cols]
                assert len(addition_df.partitions) == 1
                if pyu not in addition_df.partitions:
                    raise CompEvalError.party_check_failed(
                        f"pyu [{pyu}] is not in pred.partitions [{addition_df.partitions}]"
                    )
                out_df = pred.concat(addition_df, axis=1)
            else:
                out_df = pred
            writer.write(out_df)

    return writer.to_distdata()
