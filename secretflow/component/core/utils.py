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

import math
import os
import uuid

import pandas as pd
import pyarrow as pa
from secretflow_spec import Registry, Storage, VTableFormat, VTableParty, VTableSchema
from secretflow_spec.v1.component_pb2 import CompListDef

import secretflow.compute as sc
from secretflow.device import PYU, reveal, wait

from .dist_data.vtable_utils import VTableUtils
from .io import CSVReadOptions, CSVWriteOptions, convert_io

COMP_LIST_NAME = "secretflow"
COMP_LIST_DESC = "First-party SecretFlow components."

_comp_list_def = CompListDef()


def get_comp_list_def() -> CompListDef:
    definitions = Registry.get_definitions()
    if len(_comp_list_def.comps) != len(definitions):
        res = _build_comp_list_def(definitions)
        _comp_list_def.CopyFrom(res)

    return _comp_list_def


def _build_comp_list_def(definitions) -> CompListDef:
    return Registry.build_comp_list_def(COMP_LIST_NAME, COMP_LIST_DESC, definitions)


def uuid4(pyu: PYU | str):
    if isinstance(pyu, str):
        pyu = PYU(pyu)
    return reveal(pyu(lambda: str(uuid.uuid4()))())


def float_almost_equal(
    a: sc.Array | float, b: sc.Array | float, epsilon: float = 1e-07
) -> sc.Array:
    return sc.less(sc.abs(sc.subtract(a, b)), epsilon)


def pad_inf_to_split_points(split_points: list[float]) -> list[float]:
    assert isinstance(split_points, list), f"{split_points}"
    return [-math.inf] + split_points + [math.inf]


def assert_almost_equal(
    t1: pa.Table | pd.DataFrame,
    t2: pa.Table | pd.DataFrame,
    ignore_order: bool = False,
    *args,
    **kwargs,
) -> bool:
    df1 = t1.to_pandas() if isinstance(t1, pa.Table) else t1
    df2 = t2.to_pandas() if isinstance(t2, pa.Table) else t2
    if ignore_order:
        df1 = df1[sorted(df1.columns)]
        df2 = df2[sorted(df2.columns)]
    pd.testing.assert_frame_equal(df1, df2, *args, **kwargs)


def download_files(
    storage: Storage,
    remote_files: dict[str, str],
    local_files: dict[str, str],
    overwrite: bool = True,
):
    if set(remote_files.keys()) != set(local_files.keys()):
        raise ValueError(f"parties mismatch, {remote_files}, {local_files}")

    def download_file(rpath: str, lpath: str):
        if not overwrite and os.path.exists(lpath):
            if not os.path.isfile(lpath):
                raise ValueError(f"{lpath} is not a file")
            return
        storage.download_file(rpath, lpath)

    waits = []
    for party, remote in remote_files.items():
        res = PYU(party)(download_file)(remote, local_files[party])
        waits.append(res)
    wait(waits)


def upload_files(
    storage: Storage,
    local_files: dict[str, str],
    remote_files: dict[str, str],
):
    if set(remote_files.keys()) != set(local_files.keys()):
        raise ValueError(f"parties mismatch, {remote_files}, {local_files}")

    def upload_file(lpath: str, rpath: str):
        storage.upload_file(lpath, rpath)

    waits = []
    for party, local in local_files.items():
        res = PYU(party)(upload_file)(local, remote_files[party])
        waits.append(res)
    wait(waits)


def download_csv(
    storage: Storage, input_info: VTableParty, output_csv_path: str, na_rep: str
):
    is_csv = input_info.format == VTableFormat.CSV
    if is_csv and not input_info.null_strs:
        storage.download_file(input_info.uri, output_csv_path)
        return

    input_options = CSVReadOptions(null_values=input_info.null_strs) if is_csv else None
    with storage.get_reader(input_info.uri) as input_buffer:
        convert_io(
            input_info.format,
            input_buffer,
            input_options,
            VTableFormat.CSV,
            output_csv_path,
            CSVWriteOptions(na_rep=na_rep),
            VTableUtils.to_arrow_schema(input_info.schema),
        )


def upload_orc(
    storage: Storage,
    output_uri: str,
    output_csv_path: str,
    schema: pa.Schema | VTableSchema | None,
    null_values: list[str] | str,
) -> int:
    if schema is not None:
        if isinstance(schema, VTableSchema):
            schema = VTableUtils.to_arrow_schema(schema)
        assert isinstance(schema, pa.Schema)
    if isinstance(null_values, str):
        null_values = [null_values]
    with storage.get_writer(output_uri) as output_buffer:
        num_rows = convert_io(
            VTableFormat.CSV,
            output_csv_path,
            CSVReadOptions(null_values=null_values),
            VTableFormat.ORC,
            output_buffer,
            None,
            schema,
        )

    return num_rows
