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


import logging
import os

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.orc as orc
from secretflow_spec import Storage, VTableFormat, VTableSchema

from secretflow.component.core.dist_data.vtable_utils import VTableUtils

from .connector import IConnector, TableInfo

_mock_storage = {}


def add_mock_table(path: str, tbl: pa.Table | dict):
    if not path.startswith("/"):
        path = os.path.join("/", path)
    if isinstance(tbl, dict):
        tbl = pa.Table.from_pydict(tbl)
    _mock_storage[path] = tbl


class Mock(IConnector):
    def download_table(
        self,
        storage: Storage,
        data_dir: str,
        input_path: str,
        input_params: dict,
        output_uri: str,
        output_format: VTableFormat = VTableFormat.ORC,
    ) -> TableInfo:
        if input_path not in _mock_storage:
            raise ValueError(f"not found, path<{input_path}>")

        logging.info(
            f"download_table {storage.get_type()} {data_dir} {input_path} {input_params} {output_uri} {output_format}"
        )

        data: pa.Table = _mock_storage[input_path]

        w = storage.get_writer(output_uri)
        if output_format == VTableFormat.CSV:
            csv.write_csv(data, w)
        else:
            orc.write_table(data, w)

        return TableInfo(
            schema=VTableUtils.from_arrow_schema(data.schema, check_kind=False),
            line_count=data.num_rows,
        )

    def upload_table(
        self,
        storage: Storage,
        data_dir: str,
        input_uri: str,
        input_format: VTableFormat,
        input_schema: VTableSchema,
        output_path: str,
        output_params: dict,
    ):
        logging.info(
            f"upload_table {storage.get_type()} {data_dir} {input_uri} {input_format} {input_schema} {output_path} {output_params}"
        )
