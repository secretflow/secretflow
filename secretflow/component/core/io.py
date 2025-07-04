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
import csv
import io
import os
from dataclasses import dataclass
from typing import Tuple

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc

DEFAULT_BATCH_SIZE = 50000


class ReadOptions:
    pass


class IReader(abc.ABC):
    @abc.abstractmethod
    def read_all(self) -> pa.Table: ...

    @abc.abstractmethod
    def read_next(self) -> Tuple[pa.Table, bool]: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class WriteOptions:
    pass


class IWriter(abc.ABC):
    @abc.abstractmethod
    def write(self, data: pa.Table | pa.RecordBatch) -> int: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class BufferedIO:
    def __init__(self, buffer: io.IOBase, auto_closed: bool = False):
        self._native = buffer
        self._auto_closed = auto_closed

    @property
    def native(self) -> io.IOBase:
        return self._native

    def close(self):
        if self._auto_closed:
            self._native.close()


def _to_buffered_io(source: str | io.IOBase | BufferedIO, mode: str) -> BufferedIO:
    if isinstance(source, BufferedIO):
        return source
    elif isinstance(source, io.IOBase):
        return BufferedIO(source, False)
    elif isinstance(source, str):
        basedir = os.path.dirname(source)
        if "w" in mode and basedir:
            os.makedirs(basedir, exist_ok=True)
        file = open(source, mode)
        return BufferedIO(file, True)
    else:
        raise TypeError(f"unsupported type<{type(source)}>")


class BatchReader(abc.ABC):
    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self._remain: pa.RecordBatch = None

    def read_next_batches(self) -> list[pa.RecordBatch]:
        if self._batch_size <= 0:
            block = self.read_next_block()
            if block is None:
                return None
            return [block]

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
    def read_next_block(self) -> pa.RecordBatch | None:
        pass


@dataclass
class CSVReadOptions(ReadOptions):
    batch_size: int = DEFAULT_BATCH_SIZE
    null_values: list[str] = None


class CSVReader(IReader):
    def __init__(
        self,
        source: str | io.IOBase | BufferedIO,
        schema: pa.Schema = None,
        options: CSVReadOptions = None,
    ):
        if options is None:
            options = CSVReadOptions()
        assert isinstance(options, CSVReadOptions)

        self._buffer = _to_buffered_io(source, "rb")
        self._reader = self.new_reader(self._buffer, schema, options)
        self._schema = schema

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        tbl, eof = self.read_next()
        if eof:
            raise StopIteration
        return tbl

    def new_reader(
        self, buffer: BufferedIO, schema: pa.Schema, options: CSVReadOptions
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
        conn = duckdb.connect(":memory:")
        na_values = options.null_values if options.null_values else []
        if schema:
            duck_dtype = {f.name: PA_TYPE_TO_DUCK_TYPE[f.type] for f in schema}
            col_list = [duckdb.ColumnExpression(f.name) for f in schema]
            csv_db = conn.read_csv(buffer.native, dtype=duck_dtype, na_values=na_values)
            csv_db = csv_db.select(*col_list)
        else:
            csv_db = conn.read_csv(buffer.native, na_values=na_values)

        if options.batch_size <= 0:
            options.batch_size = DEFAULT_BATCH_SIZE
        reader = csv_db.fetch_arrow_reader(batch_size=options.batch_size)
        return reader

    def close(self) -> None:
        self._reader.close()
        self._buffer.close()

    def read_all(self) -> pa.Table:
        result = self._reader.read_all()
        if self._schema:
            result = result.cast(self._schema)
        return result

    def read_next(self) -> Tuple[pa.Table, bool]:
        try:
            batch = self._reader.read_next_batch()
            result = pa.Table.from_batches([batch])
            if self._schema:
                result = result.cast(self._schema)
            return result, False
        except StopIteration:
            return None, True


@dataclass
class CSVWriteOptions(WriteOptions):
    header: bool = True
    quoting: int = csv.QUOTE_MINIMAL
    na_rep: str = ""


class CSVWriter(IWriter):
    def __init__(
        self,
        source: str | io.IOBase | BufferedIO,
        schema: pa.Schema = None,
        options: CSVWriteOptions = None,
    ) -> None:
        if options is None:
            options = CSVWriteOptions()
        assert isinstance(options, CSVWriteOptions)

        buffer = _to_buffered_io(source, "wb")

        self._buffer = buffer
        self._options = options
        self._has_write_header = False
        if schema is not None:
            self.write_header(schema)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def write_header(self, schema: pa.Schema):
        if self._has_write_header or not self._options.header:
            return
        self._has_write_header = True
        df: pd.DataFrame = pa.Table.from_pylist([], schema=schema).to_pandas()
        df.to_csv(
            self._buffer.native,
            header=True,
            mode='w',
            quoting=self._options.quoting,
            index=False,
        )

    def write(self, data: pa.Table | pa.RecordBatch) -> int:
        self.write_header(data.schema)
        df: pd.DataFrame = data.to_pandas(integer_object_nulls=True)
        df.to_csv(
            self._buffer.native,
            header=False,
            mode='a',
            quoting=self._options.quoting,
            na_rep=self._options.na_rep,
            index=False,
        )

        return data.num_rows

    def close(self) -> None:
        self._buffer.close()


@dataclass
class ORCReadOptions(ReadOptions):
    batch_size: int = 0


class ORCReader(BatchReader, IReader):
    def __init__(
        self,
        source: str | io.BufferedIOBase | BufferedIO,
        schema: pa.Schema = None,
        options: ORCReadOptions = None,
    ) -> None:
        if options is None:
            options = ORCReadOptions()
        assert isinstance(options, ORCReadOptions)

        super().__init__(options.batch_size)
        self._buffer = _to_buffered_io(source, "rb")
        self._file = orc.ORCFile(self._buffer.native)
        self._nstripes = self._file.nstripes
        self._cur_stripe = 0
        self._schema = schema
        self._columns = schema.names if schema else None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        tbl, eof = self.read_next()
        if eof:
            raise StopIteration
        return tbl

    def read_all(self) -> pa.Table:
        result = self._file.read(columns=self._columns)
        # sort by columns and reset schema
        if self._schema:
            result = result.select(self._columns).cast(self._schema)
        return result

    def read_next(self) -> Tuple[pa.Table, bool]:
        batches = self.read_next_batches()
        if batches is None:
            return None, True
        result = pa.Table.from_batches(batches)
        # sort by columns and reset schema
        if self._schema:
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


@dataclass
class ORCWriteOptions:
    stripe_size: int = 64 * 1024 * 1024


class ORCWriter(IWriter):
    def __init__(
        self,
        source: str | io.BufferedIOBase | BufferedIO,
        schema: pa.Schema | None = None,
        options: ORCWriteOptions | None = None,
    ) -> None:
        if options is None:
            options = ORCWriteOptions()
        assert isinstance(options, ORCWriteOptions)

        self._buffer = _to_buffered_io(source, "wb")
        self._writer = orc.ORCWriter(
            self._buffer.native,
            stripe_size=options.stripe_size,
        )
        if schema is not None:
            self._writer.write(pa.Table.from_pylist([], schema=schema))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def write(self, data: pa.Table | pa.RecordBatch) -> int:
        if isinstance(data, pa.Table):
            self._writer.write(data)
        else:
            self._writer.write(pa.Table.from_batches([data]))
        return data.num_rows

    def close(self) -> None:
        self._writer.close()
        self._buffer.close()


def read_csv(
    source: str | io.BufferedIOBase | BufferedIO,
    schema: pa.Schema = None,
    options: CSVReadOptions = None,
) -> pa.Table:
    with CSVReader(source, schema, options) as r:
        return r.read_all()


def write_csv(
    data: pa.Table | dict,
    source: str | io.BufferedIOBase | BufferedIO,
    options: CSVWriteOptions = None,
):
    if isinstance(data, dict):
        data = pa.Table.from_pydict(data)
    with CSVWriter(source, options=options) as w:
        w.write(data)


def read_orc(
    source: str | io.BufferedIOBase | BufferedIO,
    schema: pa.Schema = None,
    options: ORCReadOptions = None,
) -> pa.Table:
    with ORCReader(source, schema, options) as reader:
        return reader.read_all()


def write_orc(
    data: pa.Table,
    source: str | io.BufferedIOBase | BufferedIO,
    options: ORCWriteOptions = None,
):
    with ORCWriter(source, options=options) as w:
        w.write(data)


def convert_io(
    input_format: str,
    input_stream: str | io.BufferedIOBase | BufferedIO,
    input_options: ReadOptions | None,
    output_format: str,
    output_stream: str | io.BufferedIOBase | BufferedIO,
    output_options: WriteOptions | None,
    schema: pa.Schema,
) -> int:
    assert input_format in ["orc", "csv"]
    assert output_format in ["orc", "csv"]
    if input_format == "orc":
        reader = ORCReader(input_stream, schema, input_options)
    else:
        reader = CSVReader(input_stream, schema, input_options)

    if output_format == "orc":
        writer = ORCWriter(output_stream, schema, output_options)
    else:
        writer = CSVWriter(output_stream, schema, output_options)

    num_rows = 0
    while True:
        batch, eof = reader.read_next()
        if eof:
            break
        num_rows += batch.num_rows
        writer.write(batch)

    reader.close()
    writer.close()
    return num_rows
