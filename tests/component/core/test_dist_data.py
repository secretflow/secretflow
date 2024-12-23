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


import logging

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.orc as orc

from secretflow.component.core import (
    CompVDataFrame,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    DistDataType,
    Storage,
    TimeTracer,
    VTable,
)
from secretflow.component.core.entry import setup_sf_cluster
from secretflow.device.driver import shutdown
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, TableSchema


def test_vtable(comp_prod_sf_cluster_config):
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    setup_sf_cluster(sf_cluster_config)
    storage = Storage(storage_config)
    input_path = "test_vtable/input.csv"
    output_path = "test_vtable/output.orc"
    stream_out_path = "test_vtable/output_stream.orc"

    input_data = {
        "pred": [i for i in range(100)],
    }
    pd.DataFrame(input_data).to_csv(storage.get_writer(input_path), index=False)

    meta = IndividualTable(
        schema=TableSchema(
            label_types=["float"],
            labels=["pred"],
        )
    )
    dd = DistData(
        name="input_ds",
        type=str(DistDataType.INDIVIDUAL_TABLE),
        data_refs=[
            DistData.DataRef(uri=input_path, party="alice", format="csv"),
        ],
    )
    dd.meta.Pack(meta)
    df: CompVDataFrame = CompVDataFrame.load(storage, dd)
    assert df.partition(0).shape == (100, 1)
    assert len(df.partitions) == 1

    tracer = TimeTracer()
    reader = CompVDataFrameReader(storage, tracer, dd, batch_size=10)
    writer = CompVDataFrameWriter(storage, tracer, stream_out_path)
    for df in reader:
        assert df.partition(0).shape == (10, 1)
        writer.write(df)

    def _fit_transform(tbl: pa.Table) -> pa.Table:
        name = 'pred'
        new_col = pc.add(tbl[name], 100)
        index = tbl.column_names.index(name)
        return tbl.set_column(index, tbl.field(name), new_col)

    # test apply
    out_tbl = df.apply(_fit_transform)
    out_tbl.dump(storage, output_path)

    if self_party == "alice":
        tbl_res = orc.read_table(storage.get_reader(output_path))
        logging.warning(f"...load vtable result:{self_party}... \n{tbl_res}\n.....\n")

    shutdown(
        barrier_on_shutdown=sf_cluster_config.public_config.barrier_on_shutdown,
    )


def test_vtable_null(comp_prod_sf_cluster_config):
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    setup_sf_cluster(sf_cluster_config)
    storage = Storage(storage_config)
    input_path = "test_vtable_null/input.csv"
    output_path = "test_vtable_null/output.orc"
    input_data = {
        "id": ["id1", "NA", "NaN", None],
        "col1": [1, 2, None, None],
        "col2": [0.11, 0.22, None, ""],
        "col3": [True, False, None, None],
        "col4": ["c1", "c2", "c3", "c4"],
    }
    if self_party == "alice":
        pd.DataFrame(input_data).to_csv(storage.get_writer(input_path), index=False)

    meta = IndividualTable(
        schema=TableSchema(
            label_types=["str", "int", "float", "bool", "str"],
            labels=["id", "col1", "col2", "col3", "col4"],
        )
    )
    dd = DistData(
        name="input_ds",
        type=str(DistDataType.INDIVIDUAL_TABLE),
        data_refs=[
            DistData.DataRef(
                uri=input_path,
                party="alice",
                format="csv",
                null_strs=["", "NaN", "NA"],
            ),
        ],
    )
    dd.meta.Pack(meta)
    col_selects = ["id", "col2", "col1", "col3"]
    df: CompVDataFrame = CompVDataFrame.load(
        storage, VTable.from_distdata(dd, columns=col_selects)
    )
    # check columns order
    assert df.columns == col_selects

    out_tbl = df.dump(storage, output_path)
    # read orc
    col_selects = ["col3", "id", "col2", "col1"]
    orc_tbl: CompVDataFrame = CompVDataFrame.load(
        storage, VTable.from_distdata(out_tbl, columns=col_selects)
    )
    assert orc_tbl.shape == (4, 4)
    assert orc_tbl.columns == col_selects

    if self_party == "alice":
        real_result = orc.read_table(storage.get_reader(output_path))
        logging.warning(f"...load_table_result:{self_party}... \n{real_result}\n.....")

    shutdown(
        barrier_on_shutdown=sf_cluster_config.public_config.barrier_on_shutdown,
    )
