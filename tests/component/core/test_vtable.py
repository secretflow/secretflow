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

from secretflow.component.core import DistDataType, VTable
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable


def test_vtable():
    dd = DistData(
        name="input_ds",
        type=str(DistDataType.VERTICAL_TABLE),
        data_refs=[
            DistData.DataRef(uri="aa", party="alice", format="csv"),
            DistData.DataRef(uri="bb", party="bob", format="csv"),
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=["str", "str"],
                ids=["a1", "a2"],
                feature_types=["float", "int"],
                features=["a3", "a4"],
                label_types=["int"],
                labels=["a5"],
            ),
            TableSchema(
                id_types=["str", "str"],
                ids=["b1", "b2"],
                feature_types=["float", "int"],
                features=["b3", "b4"],
                label_types=["int"],
                labels=["b5"],
            ),
        ]
    )
    dd.meta.Pack(meta)

    t = VTable.from_distdata(dd)
    assert set(t.columns) == set(
        [f"a{i+1}" for i in range(5)] + [f"b{i+1}" for i in range(5)]
    )
    t1 = t.select(["a2", "a1"])
    assert t1.columns == ["a2", "a1"]
    t2 = t.select(["a3", "a1", "b2", "b5"])
    assert t2.columns == ["a3", "a1", "b2", "b5"]
    t3 = t.drop(["a2", "a3", "b2", "b5"])
    assert set(t3.columns) == set(["a1", "a4", "a5", "b1", "b3", "b4"])
    t4 = t.drop(["a1"])
    assert set(t4.columns) == set(
        ["a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4", "b5"]
    )

    t = VTable.from_distdata(dd, columns=['a1', "a5", "a3"])
    assert t.columns == ["a1", "a5", "a3"]

    t = VTable.from_distdata(dd, columns=['a1', "a5", "a3", "b2", "b4"])
    assert t.columns == ["a1", "a5", "a3", "b2", "b4"]
