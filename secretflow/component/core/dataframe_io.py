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

import pyarrow as pa
from secretflow_spec import Storage

from secretflow.device import PYU, PYUObject, proxy

from .io import (
    BufferedIO,
    CSVReader,
    CSVReadOptions,
    CSVWriteOptions,
    CSVWriter,
    IReader,
    IWriter,
    ORCReader,
    ORCReadOptions,
    ORCWriteOptions,
    ORCWriter,
    WriteOptions,
)


@proxy(device_object_type=PYUObject)
class CSVReaderProxy(CSVReader):
    def __init__(
        self,
        storage: Storage,
        uri: str,
        schema: pa.Schema = None,
        options: CSVReadOptions = None,
    ):
        buffer = BufferedIO(storage.get_reader(uri), auto_closed=True)
        super().__init__(buffer, schema, options)


@proxy(device_object_type=PYUObject)
class CSVWriterProxy(CSVWriter):
    def __init__(
        self,
        storage: Storage,
        uri: str,
        schema: pa.Schema,
        options: CSVWriteOptions = None,
    ) -> None:
        buffer = BufferedIO(storage.get_writer(uri), auto_closed=True)
        super().__init__(buffer, schema, options)


@proxy(device_object_type=PYUObject)
class ORCReaderProxy(ORCReader):
    def __init__(
        self,
        storage: Storage,
        uri: str,
        schema: pa.Schema = None,
        options: ORCReadOptions = None,
    ) -> None:
        buffer = BufferedIO(storage.get_reader(uri), auto_closed=True)
        super().__init__(buffer, schema, options)


@proxy(device_object_type=PYUObject)
class ORCWriterProxy(ORCWriter):
    def __init__(
        self,
        storage: Storage,
        uri: str,
        schema: pa.Schema = None,
        options: ORCWriteOptions = None,
    ) -> None:
        buffer = BufferedIO(storage.get_writer(uri), auto_closed=True)
        super().__init__(buffer, schema, options)


def new_reader_proxy(
    device: PYU,
    storage: Storage,
    format: str,
    uri: str,
    schema: pa.Schema,
    null_strs: list[str] = None,
    batch_size: int = None,
) -> IReader:
    if batch_size is None:
        batch_size = 50000
    if format == "csv":
        options = CSVReadOptions(batch_size, null_strs)
        return CSVReaderProxy(storage, uri, schema, options, device=device)
    elif format == "orc":
        options = ORCReadOptions(batch_size)
        return ORCReaderProxy(storage, uri, schema, options, device=device)
    else:
        raise ValueError(f"unsupport format {format}")


def new_writer_proxy(
    device: PYU,
    storage: Storage,
    format: str,
    uri: str,
    schema: pa.Schema = None,
    options: WriteOptions = None,
) -> IWriter:
    if format == "csv":
        return CSVWriterProxy(storage, uri, schema, options, device=device)
    elif format == "orc":
        return ORCWriterProxy(storage, uri, schema, options, device=device)
    else:
        raise ValueError(f"unsupport format {format}")
