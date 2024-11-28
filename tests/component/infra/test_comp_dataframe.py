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

import itertools
import logging
import os
import random
from typing import List, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import csv, orc

from secretflow import PYU, reveal, wait
from secretflow.component.component import CompEvalContext
from secretflow.component.dataframe import (
    CompDataFrame,
    CompTable,
    StreamingReader,
    StreamingWriter,
)
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.data_pb2 import (
    DistData,
    StorageConfig,
    SystemInfo,
    TableSchema,
    VerticalTable,
)
from tests.cluster import get_self_party
from tests.conftest import prepare_storage_path, setup_minio_server


def build_test_table(
    prefix: str, samples=50000, f_cols=40, i_cols=40, s_cols=9
) -> Tuple[pa.Table, List]:
    f = pd.DataFrame(
        np.random.random((samples, f_cols)),
        columns=[f"{prefix}_f{i+1}" for i in range(f_cols)],
    )
    i = pd.DataFrame(
        np.random.randint(-1000, 1000, (samples, i_cols)),
        columns=[f"{prefix}_i{i+1}" for i in range(i_cols)],
    )
    s = pd.DataFrame(
        np.random.choice(["aaaaa", "bbbbb", "cccc"], (samples, s_cols)),
        columns=[f"{prefix}_s{i+1}" for i in range(s_cols)],
    )
    id = pd.DataFrame(range(samples), columns=[f"{prefix}_id"])

    return pa.Table.from_pandas(pd.concat([f, i, s, id], axis=1)), [
        list(f.columns),
        list(i.columns),
        list(s.columns),
        [f"{prefix}_id"],
    ]


@pytest.fixture(scope="module")
def dataframe_case(sf_production_setup_devices):
    local_dir = prepare_storage_path(get_self_party())
    s3_dir = os.path.join(local_dir, "minio")

    local_ctx = CompEvalContext(
        data_dir=local_dir,
        comp_storage=ComponentStorage(
            StorageConfig(
                type="local_fs",
                local_fs=StorageConfig.LocalFSConfig(wd=local_dir),
            )
        ),
    )

    ms, s3_storage = setup_minio_server(s3_dir, get_self_party())

    s3_ctx = CompEvalContext(
        data_dir=s3_dir,
        comp_storage=ComponentStorage(s3_storage),
    )

    sf_production_setup_devices.local_ctx = local_ctx
    sf_production_setup_devices.s3_ctx = s3_ctx

    yield sf_production_setup_devices

    ms.kill()


def test_dataframe_streaming_read(dataframe_case):
    alice = dataframe_case.alice
    bob = dataframe_case.bob
    local_ctx = dataframe_case.local_ctx
    s3_ctx = dataframe_case.s3_ctx

    csv_path = os.path.join("test_dataframe_streaming_read", "df.csv")
    orc_path = os.path.join("test_dataframe_streaming_read", "df.orc")
    expected_row_cnt = 49991

    def create_file(prefix, f_cols, i_cols, s_cols):
        x, cols = build_test_table(prefix, expected_row_cnt, f_cols, i_cols, s_cols)
        with local_ctx.comp_storage.get_writer(csv_path) as w:
            csv.write_csv(x, w)
        with s3_ctx.comp_storage.get_writer(csv_path) as w:
            csv.write_csv(x, w)
        with local_ctx.comp_storage.get_writer(orc_path) as w:
            orc.write_table(
                x,
                w,
                stripe_size=2 * 1024 * 1024,
            )
        with s3_ctx.comp_storage.get_writer(orc_path) as w:
            orc.write_table(
                x,
                w,
                stripe_size=2 * 1024 * 1024,
            )
        return x, cols

    alice_x, alice_cols = reveal(alice(create_file)("alice", 50, 50, 14))
    # unbalanced
    bob_x, bob_cols = reveal(bob(create_file)("bob", 23, 23, 7))

    input_csv = DistData(
        name="train_dataset",
        type="sf.table.vertical_table",
        data_refs=[
            DistData.DataRef(uri=csv_path, party="alice", format="csv"),
            DistData.DataRef(uri=csv_path, party="bob", format="csv"),
        ],
    )
    input_orc = DistData(
        name="train_dataset",
        type="sf.table.vertical_table",
        data_refs=[
            DistData.DataRef(uri=orc_path, party="alice", format="orc"),
            DistData.DataRef(uri=orc_path, party="bob", format="orc"),
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                ids=alice_cols[3],
                id_types=["int64"],
                feature_types=["float64"] * len(alice_cols[0])
                + ["int64"] * len(alice_cols[1])
                + ["str"] * len(alice_cols[2]),
                features=alice_cols[0] + alice_cols[1] + alice_cols[2],
            ),
            TableSchema(
                ids=bob_cols[3],
                id_types=["int64"],
                feature_types=["float64"] * len(bob_cols[0])
                + ["int64"] * len(bob_cols[1])
                + ["str"] * len(bob_cols[2]),
                features=bob_cols[0] + bob_cols[1] + bob_cols[2],
            ),
        ],
    )
    input_csv.meta.Pack(meta)
    input_orc.meta.Pack(meta)

    alice_all_f = [c for pc in alice_cols for c in pc]
    bob_all_f = [c for pc in bob_cols for c in pc]
    all_f = alice_all_f + bob_all_f

    def sample_cols():
        alice_p_f = (
            random.sample(alice_all_f, int(len(alice_all_f) / 2)) + alice_cols[3]
        )
        alice_p_f = list(set(alice_p_f))
        bob_p_f = random.sample(bob_all_f, int(len(bob_all_f) / 2)) + bob_cols[3]
        bob_p_f = list(set(bob_p_f))
        part_f = alice_p_f + bob_p_f
        return alice_p_f, bob_p_f, part_f

    alice_p_f, bob_p_f, part_f = reveal(alice(sample_cols)())

    for ctx, input, use_all_f, order in itertools.product(
        [local_ctx, s3_ctx],
        [input_orc, input_orc],
        [True, False],
        [["alice", "bob"], ["bob", "alice"]],
    ):
        # logging.warn(f"test {ctx}, {input}, {use_all_f}, {order}")
        reader = StreamingReader.from_distdata(
            ctx,
            input,
            partitions_order=order,
            load_features=True,
            load_ids=True,
            load_labels=True,
            col_selects=all_f if use_all_f else part_f,
            batch_size=9871,
        )
        alice_ids = None
        bob_ids = None
        row_cnt = 0
        for batch in reader:
            assert list(map(lambda x: x.party, batch.partitions.keys())) == order
            partition_columns = batch.partition_columns
            assert partition_columns[alice] == alice_all_f if use_all_f else alice_p_f
            assert partition_columns[bob] == bob_all_f if use_all_f else bob_p_f
            shape = batch.shape
            assert shape[0] == 9871 or shape[0] == 636
            row_cnt += shape[0]
            assert shape[1] == len(all_f if use_all_f else part_f)
            alice_id = reveal(batch[alice_cols[3]].partitions[alice].data)
            bob_id = reveal(batch[bob_cols[3]].partitions[bob].data)
            if not alice_ids:
                alice_ids = alice_id
            else:
                alice_ids = pa.concat_tables([alice_ids, alice_id])
            if not bob_ids:
                bob_ids = bob_id
            else:
                bob_ids = pa.concat_tables([bob_ids, bob_id])

        assert alice_x.select(alice_cols[3]).equals(alice_ids)
        assert bob_x.select(bob_cols[3]).equals(bob_ids)
        assert expected_row_cnt == row_cnt


def test_dataframe_streaming_write(dataframe_case):
    alice = dataframe_case.alice
    bob = dataframe_case.bob
    local_ctx = dataframe_case.local_ctx
    s3_ctx = dataframe_case.s3_ctx

    _, alice_cols = build_test_table("alice", 1)
    alice_f = alice_cols[0] + alice_cols[1] + alice_cols[2]
    _, bob_cols = build_test_table("bob", 1)
    bob_f = bob_cols[0] + bob_cols[1] + bob_cols[2]

    uri = os.path.join("test_dataframe_streaming_write", "df.orc")

    for ctx in [local_ctx, s3_ctx]:
        writer = StreamingWriter(ctx, uri)
        batch_x = None
        with writer:
            batch_count = 10
            samples = reveal(alice(np.random.randint)(4000, 6000, (batch_count,)))
            samples = list(map(int, samples))
            while batch_count > 0:
                batch_count -= 1
                sample = samples[batch_count]
                alice_x, _ = alice(build_test_table)("alice", samples=sample)
                bob_x, _ = bob(build_test_table)("bob", samples=sample)
                if batch_count % 2 == 0:
                    alice_x = alice(
                        lambda t: t.select(map(int, np.random.permutation(t.shape[1])))
                    )(alice_x)
                    bob_x = bob(
                        lambda t: t.select(map(int, np.random.permutation(t.shape[1])))
                    )(bob_x)
                batch = CompDataFrame(
                    {
                        alice: CompTable(alice_x, alice_cols[3], alice_f, []),
                        bob: CompTable(bob_x, bob_cols[3], bob_f, []),
                    },
                    SystemInfo(),
                )
                writer.write(batch)
                if not batch_x:
                    batch_x = batch
                else:
                    batch_x = batch_x.concat(batch[batch_x.columns], axis=0)

        read_df = CompDataFrame.from_distdata(
            ctx,
            writer.to_distdata(),
            load_features=True,
            load_ids=True,
            load_labels=True,
        )
        assert set(read_df.partitions.keys()) == set(batch_x.partitions.keys())
        assert read_df.shape == batch_x.shape
        assert read_df.columns == batch_x.columns

        for pyu, r_x in read_df.partitions.items():
            b_x = batch_x.data(pyu)
            r_x, b_x = reveal([r_x.data, b_x])
            assert r_x.equals(b_x)


def test_dataframe_io(dataframe_case):
    alice = dataframe_case.alice
    bob = dataframe_case.bob
    local_ctx = dataframe_case.local_ctx
    s3_ctx = dataframe_case.s3_ctx

    in_csv_path = os.path.join("test_dataframe_io", "df.csv")
    out_orc_path = os.path.join("test_dataframe_io", "df.orc")

    input_csv = DistData(
        name="train_dataset",
        type="sf.table.vertical_table",
        data_refs=[
            DistData.DataRef(
                uri=in_csv_path,
                party="alice",
                format="csv",
                null_strs=["0", "NaN", "99.9"],
            ),
            DistData.DataRef(uri=in_csv_path, party="bob", format="csv"),
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                ids=["ida"],
                id_types=["str"],
                feature_types=["int32", "float32"],
                features=["ai", "af"],
                labels=["al"],
                label_types=["int"],
            ),
            TableSchema(
                ids=["idb"],
                id_types=["str"],
                feature_types=["int32", "float32"],
                features=["bi", "bf"],
                labels=["bl"],
                label_types=["float"],
            ),
        ],
    )
    input_csv.meta.Pack(meta)

    a_normal = (
        "ida,ai,af,al",
        "1,10,1.5,1",
        "2,20,2.5,2",
        "NaN,99.9,NaN,0",
    )
    b_normal = (
        "idb,bi,bf,bl",
        "1,10,1.1,1",
        "2,20,2.2,2",
        "3,30,3.3,1",
    )

    def write_csv(ctx, lines):
        with ctx.comp_storage.get_writer(in_csv_path) as w:
            w.write("\n".join(lines).encode())

    for ctx in [local_ctx, s3_ctx]:
        wait(alice(write_csv)(ctx, a_normal))
        wait(bob(write_csv)(ctx, b_normal))

        read_df = CompDataFrame.from_distdata(
            ctx,
            input_csv,
            partitions_order=["alice", "bob"],
            load_features=True,
            load_ids=True,
            load_labels=True,
        )

        a_cols = ["af", "ida", "al", "ai"]
        b_cols = ["idb", "bf", "bi", "bl"]

        def _shuffle(l):
            random.shuffle(l)
            return l

        a_cols = reveal(alice(_shuffle)(a_cols))
        b_cols = reveal(alice(_shuffle)(b_cols))
        select_df = read_df[b_cols + a_cols]
        assert list(select_df.partitions.keys()) == list(read_df.partitions.keys())
        assert select_df.columns == a_cols + b_cols
        a_cols = reveal(alice(_shuffle)(a_cols))
        b_cols = reveal(alice(_shuffle)(b_cols))
        select_df = read_df[a_cols + b_cols]
        assert select_df.columns == a_cols + b_cols
        assert list(select_df.partitions.keys()) == list(read_df.partitions.keys())
        select_df = read_df[["af", "idb"]]
        a_select, b_select = reveal([p.data for p in select_df.partitions.values()])
        assert [v.as_py() for v in a_select[0]] == [1.5, 2.5, None]
        assert [v.as_py() for v in b_select[0]] == ["1", "2", "3"]
        drop_df = select_df.drop("af")
        assert len(drop_df.partitions) == 1
        assert bob in drop_df.partitions
        select_df = read_df["af"]
        assert len(select_df.partitions) == 1
        assert alice in select_df.partitions

        v_data = read_df.to_pandas(check_null=False)
        from_pandas_df = CompDataFrame.from_pandas(
            v_data, input_csv.system_info, ["ida", "idb"], ["bl", "al"]
        )
        assert list(from_pandas_df.partitions.keys()) == list(read_df.partitions.keys())
        for pyu in from_pandas_df.partitions:
            assert (
                from_pandas_df.partitions[pyu].feature_cols
                == read_df.partitions[pyu].feature_cols
            )
            assert (
                from_pandas_df.partitions[pyu].id_cols
                == read_df.partitions[pyu].id_cols
            )
            assert (
                from_pandas_df.partitions[pyu].label_cols
                == read_df.partitions[pyu].label_cols
            )

        a_csv, b_csv = reveal([p.data for p in read_df.partitions.values()])
        assert a_csv.shape == b_csv.shape
        assert a_csv.shape == (3, 4)
        assert all([a_csv[c][2].as_py() is None for c in range(4)])
        assert a_csv.drop_null().shape == (2, 4)
        assert b_csv.drop_null().shape == (3, 4)

        orc_dd = read_df.to_distdata(ctx, out_orc_path)
        read_orc_df = CompDataFrame.from_distdata(
            ctx,
            orc_dd,
            load_features=True,
            load_ids=True,
            load_labels=True,
        )
        a_orc, b_orc = reveal([p.data for p in read_orc_df.partitions.values()])
        assert a_orc.equals(a_csv)
        assert b_orc.equals(b_csv)
