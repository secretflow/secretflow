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


import numpy as np
import pandas as pd
from pyarrow import orc

from secretflow.component.data_utils import DistDataType, extract_data_infos
from secretflow.component.preprocessing.filter.condition_filter import (
    condition_filter_comp,
)
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_condition_filter(comp_prod_sf_cluster_config):
    alice_input_path = "test_condition_filter/alice.csv"
    bob_input_path = "test_condition_filter/bob.csv"
    train_output_path = "test_condition_filter/train.csv"
    test_output_path = "test_condition_filter/test.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    if self_party == "alice":
        df_alice = pd.DataFrame(
            {
                "id1": [1, 2, 3, 4],
                "a1": ["K5", "K1", None, "K6"],
                "a2": ["A5", "A1", "A2", "A6"],
                "a3": [5, 1, 2, 6],
                "y": [0, 1, 1, 0],
            }
        )
        df_alice.to_csv(
            comp_storage.get_writer(alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        df_bob = pd.DataFrame(
            {
                "id2": [1, 2, 3, 4],
                "b4": [10.2, 20.5, None, -0.4],
                "b5": ["B3", None, "B9", "B4"],
                "b6": [3, 1, 9, 4],
            }
        )
        df_bob.to_csv(
            comp_storage.get_writer(bob_input_path),
            index=False,
        )

    param = NodeEvalParam(
        domain="data_filter",
        name="condition_filter",
        version="0.0.2",
        attr_paths=[
            'input/in_ds/features',
            'comparator',
            'bound_value',
            'float_epsilon',
        ],
        attrs=[
            Attribute(ss=['b4']),
            Attribute(s='<'),
            Attribute(s='11'),
            Attribute(f=0.01),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=bob_input_path, party="bob", format="csv", null_strs=[""]
                    ),
                    DistData.DataRef(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                        null_strs=[""],
                    ),
                ],
            )
        ],
        output_uris=[
            train_output_path,
            test_output_path,
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=["str"],
                ids=["id2"],
                feature_types=["float32", "str", "float32"],
                features=["b4", "b5", "b6"],
            ),
            TableSchema(
                id_types=["str"],
                ids=["id1"],
                feature_types=["str", "str", "float32"],
                features=["a1", "a2", "a3"],
                label_types=["float32"],
                labels=["y"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    res = condition_filter_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    ds_info = extract_data_infos(res.outputs[0], load_ids=True)
    else_ds_info = extract_data_infos(res.outputs[1], load_ids=True)

    if self_party == "alice":
        ds_alice = orc.read_table(
            comp_storage.get_reader(ds_info["alice"].uri)
        ).to_pandas()
        ds_else_alice = orc.read_table(
            comp_storage.get_reader(else_ds_info["alice"].uri)
        ).to_pandas()
        np.testing.assert_equal(ds_alice.shape[0], 2)
        assert list(ds_alice["id1"]) == ["1", "4"]
        assert list(ds_else_alice["id1"]) == ["2", "3"]

    if self_party == "bob":
        ds_bob = orc.read_table(comp_storage.get_reader(ds_info["bob"].uri)).to_pandas()
        ds_else_bob = orc.read_table(
            comp_storage.get_reader(else_ds_info["bob"].uri)
        ).to_pandas()
        np.testing.assert_equal(ds_else_bob.shape[0], 2)
        assert list(ds_bob["id2"]) == ["1", "4"]
        assert list(ds_else_bob["id2"]) == ["2", "3"]

    # test null
    param.ClearField("attr_paths")
    param.ClearField("attrs")
    param.attr_paths.extend(['input/in_ds/features', 'comparator'])
    param.attrs.extend([Attribute(ss=['a1']), Attribute(s='NOTNULL')])
    res = condition_filter_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    if self_party == "alice":
        ds_alice = orc.read_table(
            comp_storage.get_reader(ds_info["alice"].uri)
        ).to_pandas()
        ds_else_alice = orc.read_table(
            comp_storage.get_reader(else_ds_info["alice"].uri)
        ).to_pandas()
        np.testing.assert_equal(ds_alice.shape[0], 3)
        assert list(ds_alice["id1"]) == ["1", "2", "4"]
        assert list(ds_else_alice["id1"]) == ["3"]
    if self_party == "bob":
        ds_bob = orc.read_table(comp_storage.get_reader(ds_info["bob"].uri)).to_pandas()
        ds_else_bob = orc.read_table(
            comp_storage.get_reader(else_ds_info["bob"].uri)
        ).to_pandas()
        np.testing.assert_equal(ds_else_bob.shape[0], 1)
        assert list(ds_bob["id2"]) == ["1", "2", "4"]
        assert list(ds_else_bob["id2"]) == ["3"]
