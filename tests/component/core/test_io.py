# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io

import pyarrow as pa

from secretflow.component.core import (
    CSVReader,
    CSVReadOptions,
    CSVWriteOptions,
    ORCReader,
    ORCReadOptions,
    ORCWriter,
    convert_io,
    read_csv,
    read_orc,
    write_csv,
    write_orc,
)


def test_csv():
    data = {
        "column1": [1, None, 3],
        "column2": ["a", "b", None],
        "column3": [3.5, None, 5.5],
    }
    excepted_data = "column1,column2,column3\n1,a,3.5\nNULL,b,NULL\n3,NULL,5.5\n"
    schema = pa.schema(
        [
            pa.field("column1", pa.int64()),
            pa.field("column2", pa.string()),
            pa.field("column3", pa.float64()),
        ]
    )
    tbl = pa.table(data)
    buffer = io.BytesIO()
    write_csv(tbl, buffer, CSVWriteOptions(na_rep="NULL"))
    assert buffer.getvalue().decode("utf-8") == excepted_data

    read_options = CSVReadOptions(null_values=["NULL"])
    out_tbl = read_csv(buffer, schema, read_options)
    assert tbl == out_tbl
    out_tbl = read_csv(buffer, options=read_options)
    assert tbl == out_tbl

    schema = pa.schema(
        [pa.field("column1", pa.int64()), pa.field("column3", pa.float64())]
    )
    out_tbl = read_csv(buffer, schema, read_options)
    assert out_tbl == tbl.select(["column1", "column3"])

    # test batch_size
    reader = CSVReader(
        buffer, options=CSVReadOptions(batch_size=1, null_values=["NULL"])
    )
    for batch_tbl in reader:
        assert batch_tbl.num_rows == 1

    # test convert_io
    out_buffer = io.StringIO()
    convert_io("csv", buffer, read_options, "csv", out_buffer, None, None)
    excepted_data = "column1,column2,column3\n1,a,3.5\n,b,\n3,,5.5\n"
    assert out_buffer.getvalue() == excepted_data


def test_orc():
    data = {
        "column1": [1, None, 3],
        "column2": ["a", "b", None],
        "column3": [3.5, None, 5.5],
    }

    tbl = pa.table(data)
    buffer = io.BytesIO()
    write_orc(tbl, buffer)
    out_tbl = read_orc(buffer)
    assert tbl == out_tbl

    schema = pa.schema(
        [
            pa.field("column1", pa.int64()),
            pa.field("column3", pa.float64()),
        ]
    )
    out_tbl = read_orc(buffer, schema=schema)
    assert out_tbl == tbl.select(["column1", "column3"])

    # test batch_size
    reader = ORCReader(buffer, options=ORCReadOptions(batch_size=1))
    for batch_tbl in reader:
        assert batch_tbl.num_rows == 1

    # test write empty table by schema
    w = ORCWriter(io.BytesIO(), schema=schema)
    w.close()

    # test convert_io
    out_buffer = io.StringIO()
    convert_io(
        "orc", buffer, None, "csv", out_buffer, CSVWriteOptions(na_rep="NULL"), None
    )
    excepted_data = "column1,column2,column3\n1,a,3.5\nNULL,b,NULL\n3,NULL,5.5\n"
    assert (
        out_buffer.getvalue() == excepted_data
    ), f"output:\n {repr(out_buffer.getvalue())}, except:\n {repr(excepted_data)}"
