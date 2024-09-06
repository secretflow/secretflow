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
import operator

import numpy as np
import pandas as pd
import pytest
from pyarrow import orc
from sklearn.datasets import load_breast_cancer

from secretflow.component.core import DistDataType, Storage
from secretflow.component.entry import comp_eval
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


@pytest.mark.parametrize(
    "binary_op",
    [
        "+",
    ],  # "/"
)
@pytest.mark.parametrize("features", [["a4", "a9"], ["b5", "b6"]])  # ["a1", "a2"],
@pytest.mark.parametrize(
    "as_label",
    [
        # True,
        False,
    ],
)
@pytest.mark.parametrize(
    "new_feature_name",
    [
        "b6",
        "c1",
    ],
)
def test_binary_op_sample(
    comp_prod_sf_cluster_config,
    features,
    binary_op,
    as_label,
    new_feature_name,
):
    f1, f2 = [features[0]], [features[1]]

    alice_input_path = "test_binary_op/alice.csv"
    bob_input_path = "test_binary_op/bob.csv"
    output_path = "test_binary_op/out.csv"
    rule_path = "test_binary_op/binary_op.rule"
    sub_path = "test_binary_op/substitution.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    storage = Storage(storage_config)

    x = load_breast_cancer()["data"]
    alice_columns = [f"a{i}" for i in range(15)]
    bob_columns = [f"b{i}" for i in range(15)]

    if self_party == "alice":
        ds = pd.DataFrame(x[:, :15], columns=alice_columns)
        ds.to_csv(storage.get_writer(alice_input_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=bob_columns)
        ds.to_csv(storage.get_writer(bob_input_path), index=False)

    param = NodeEvalParam(
        domain="preprocessing",
        name="binary_op",
        version="1.0.0",
        attr_paths=[
            "f1",
            "f2",
            "binary_op",
            "as_label",
            "new_feature_name",
        ],
        attrs=[
            Attribute(ss=f1),
            Attribute(ss=f2),
            Attribute(s=binary_op),
            Attribute(b=as_label),
            Attribute(s=new_feature_name),
        ],
        inputs=[
            DistData(
                name="input",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[output_path, rule_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=alice_columns,
            ),
            TableSchema(
                feature_types=["float32"] * 15,
                features=bob_columns,
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)
    bad_case = (
        (features[0] in alice_columns) and (new_feature_name in bob_columns)
    ) or ((features[0] in bob_columns) and (new_feature_name in alice_columns))
    if bad_case:
        with pytest.raises(Exception):
            res = comp_eval(
                param=param,
                storage_config=storage_config,
                cluster_config=sf_cluster_config,
            )
        return
    else:
        res = comp_eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

    assert len(res.outputs) == 2

    def test(df, df_in):
        assert new_feature_name in df.columns
        item_0 = df_in[features[0]]
        item_1 = df_in[features[1]]
        expected_result = ops[binary_op](item_0, item_1)
        np.testing.assert_array_almost_equal(
            df[new_feature_name], expected_result, decimal=3
        )

    if features[0] in alice_columns and self_party == "alice":
        df = orc.read_table(storage.get_reader(output_path)).to_pandas()
        df_in = pd.DataFrame(x[:, :15], columns=alice_columns)
        test(df, df_in)
    elif features[0] in bob_columns and self_party == "bob":
        df = orc.read_table(storage.get_reader(output_path)).to_pandas()
        df_in = pd.DataFrame(x[:, 15:], columns=bob_columns)
        test(df, df_in)

    param2 = NodeEvalParam(
        domain="preprocessing",
        name="substitution",
        version="1.0.0",
        inputs=[param.inputs[0], res.outputs[1]],
        output_uris=[sub_path],
    )

    res = comp_eval(
        param=param2,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1
    if self_party == "alice":
        a_out = orc.read_table(storage.get_reader(sub_path)).to_pandas()
        logging.warning(f"....... \n{a_out}\n.,......")
