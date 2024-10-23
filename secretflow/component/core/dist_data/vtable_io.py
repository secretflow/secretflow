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


import abc
from typing import List, Tuple

import duckdb
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.orc as orc

from secretflow.device import PYU, PYUObject, proxy

from ..storage import Storage


class IReader(abc.ABC):
    @abc.abstractmethod
    def read_all(self) -> pa.Table: ...

    @abc.abstractmethod
    def read_next(self) -> Tuple[pa.Table, bool]: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class IWriter(abc.ABC):
    @abc.abstractmethod
    def write(self, data: pa.Table | pa.RecordBatch) -> int: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class BatchReader(abc.ABC):
    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self._remain: pa.RecordBatch = None

    def read_next_batches(self) -> list[pa.RecordBatch]:
        batches: list[pa.RecordBatch] = []
        num_rows: int = 0
        if self._remain is not None:
            num_rows = self._remain.num_rows
            batches.append(self._remain)
            self._remain = None

        while num_rows < self._batch_size:
            block = self.read_next_block()
            if block is None:
                if num_rows == 0:
                    return None
                break
            batches.append(block)
            num_rows += block.num_rows

        if num_rows > self._batch_size:
            last = batches[-1]
            remain_size = num_rows - self._batch_size
            last_size = last.num_rows - remain_size
            self._remain = last.slice(last_size, remain_size)
            last = last.slice(0, last_size)
            batches[-1] = last

        return batches if num_rows > 0 else None

    @abc.abstractmethod
    def read_next_block(self) -> pa.RecordBatch:
        pass


@proxy(device_object_type=PYUObject)
class CSVReader(BatchReader, IReader):
    def __init__(
        self,
        storage: Storage,
        uri: str,
        schema: pa.Schema,
        batch_size,
        null_values: List[str],
    ):
        super().__init__(batch_size=batch_size)
        buffer = storage.get_reader(uri)
        include_columns = [field.name for field in schema]
        read_options = csv.ReadOptions(column_names=None)
        convert_options = csv.ConvertOptions(
            include_columns=include_columns,
            null_values=null_values,
            strings_can_be_null=True,
        )
        self._reader = csv.open_csv(
            buffer,
            read_options=read_options,
            convert_options=convert_options,
        )
        self._schema = schema

    def close(self) -> None:
        self._reader.close()

    def read_all(self) -> pa.Table:
        result = self._reader.read_all()
        return result.cast(self._schema)

    def read_next(self) -> Tuple[pa.Table, bool]:
        batches = self.read_next_batches()
        if batches is None:
            return None, True
        result = pa.Table.from_batches(batches)
        return result.cast(self._schema), False

    def read_next_block(self) -> pa.RecordBatch:
        try:
            return self._reader.read_next_batch()
        except StopIteration:
            return None


@proxy(device_object_type=PYUObject)
class DuckDbCSVReader(IReader):
    def __init__(
        self,
        storage: Storage,
        uri: str,
        schema: pa.Schema,
        batch_size,
        null_values: List[str],
    ):
        buffer = storage.get_reader(uri)
        self._buffer = buffer
        self._reader = self.new_reader(buffer, schema, null_values, batch_size)
        self._schema = schema

    def new_reader(
        self, buffer, schema: pa.Schema, null_values: List[str], batch_size: int
    ):
        PA_TYPE_TO_DUCK_TYPE = {
            pa.int8(): duckdb.typing.TINYINT,
            pa.int16(): duckdb.typing.SMALLINT,
            pa.int32(): duckdb.typing.INTEGER,
            pa.int64(): duckdb.typing.BIGINT,
            pa.uint8(): duckdb.typing.UTINYINT,
            pa.uint16(): duckdb.typing.USMALLINT,
            pa.uint32(): duckdb.typing.UINTEGER,
            pa.uint64(): duckdb.typing.UBIGINT,
            pa.float32(): duckdb.typing.FLOAT,
            pa.float64(): duckdb.typing.DOUBLE,
            pa.bool_(): duckdb.typing.BOOLEAN,
            pa.string(): duckdb.typing.VARCHAR,
        }
        duck_dtype = {f.name: PA_TYPE_TO_DUCK_TYPE[f.type] for f in schema}
        na_values = null_values if null_values else []
        conn = duckdb.connect(":memory:")
        csv_db = conn.read_csv(buffer, dtype=duck_dtype, na_values=na_values)
        col_list = [duckdb.ColumnExpression(f.name) for f in schema]
        csv_select = csv_db.select(*col_list)
        return csv_select.fetch_arrow_reader(batch_size=batch_size)

    def close(self) -> None:
        self._reader.close()
        self._buffer.close()

    def read_all(self) -> pa.Table:
        result = self._reader.read_all()
        return result.cast(self._schema)

    def read_next(self) -> Tuple[pa.Table, bool]:
        try:
            batch = self._reader.read_next_batch()
            result = pa.Table.from_batches([batch])
            return result.cast(self._schema), False
        except StopIteration:
            return None, True


@proxy(device_object_type=PYUObject)
class CSVWriter(IWriter):
    def __init__(self, storage: Storage, uri: str) -> None:
        buffer = storage.get_writer(uri)
        self._writer = csv.CSVWriter(buffer)

    def write(self, data: pa.Table | pa.RecordBatch) -> int:
        self._writer.write(data)
        return data.num_rows

    def close(self) -> None:
        self._writer.close()


@proxy(device_object_type=PYUObject)
class ORCReader(BatchReader, IReader):
    def __init__(
        self, storage: Storage, uri: str, schema: pa.Schema, batch_size=0
    ) -> None:
        super().__init__(batch_size)
        self._buffer = storage.get_reader(uri)
        self._file = orc.ORCFile(self._buffer)
        self._nstripes = self._file.nstripes
        self._cur_stripe = 0
        self._schema = schema
        self._columns = schema.names

    def read_all(self) -> pa.Table:
        result = self._file.read(columns=self._columns)
        # sort by columns and reset schema
        result = result.select(self._columns).cast(self._schema)
        return result

    def read_next(self) -> Tuple[pa.Table, bool]:
        batches = self.read_next_batches()
        if batches is None:
            return None, True
        result = pa.Table.from_batches(batches)
        # sort by columns and reset schema
        result = result.select(self._columns).cast(self._schema)
        return result, False

    def read_next_block(self) -> pa.RecordBatch:
        if self._cur_stripe == self._nstripes:
            return None
        result = self._file.read_stripe(self._cur_stripe, columns=self._columns)
        self._cur_stripe += 1
        return result

    def close(self) -> None:
        self._buffer.close()


@proxy(device_object_type=PYUObject)
class ORCWriter:
    def __init__(
        self, storage: Storage, uri: str, stripe_size=64 * 1024 * 1024
    ) -> None:
        self._buffer = storage.get_writer(uri)
        self._writer = orc.ORCWriter(
            self._buffer,
            stripe_size=stripe_size,
        )

    def write(self, data: pa.Table | pa.RecordBatch) -> int:
        self._writer.write(data)
        return data.num_rows

    def close(self) -> None:
        self._writer.close()
        self._buffer.close()


def new_reader(
    device: PYU,
    storage: Storage,
    format: str,
    uri: str,
    null_strs: list[str],
    schema: pa.Schema,
    batch_size: int,
    backend: str = "duckdb",
) -> IReader:
    if batch_size <= 0:
        batch_size = 50000
    if format == "csv":
        if backend == "duckdb":
            return DuckDbCSVReader(
                storage, uri, schema, batch_size, null_strs, device=device
            )
        else:
            return CSVReader(storage, uri, schema, batch_size, null_strs, device=device)

    elif format == "orc":
        return ORCReader(storage, uri, schema, batch_size, device=device)
    else:
        raise ValueError(f"unsupport format {format}")


def new_writer(device: PYU, storage: Storage, format: str, uri: str) -> IWriter:
    if format == "csv":
        return CSVWriter(storage, uri, device=device)
    elif format == "orc":
        return ORCWriter(storage, uri, device=device)
    else:
        raise ValueError(f"unsupport format {format}")
