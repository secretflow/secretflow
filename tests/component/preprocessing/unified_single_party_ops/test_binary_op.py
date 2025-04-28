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
import time

import numpy as np
import pandas as pd
import pytest
from pyarrow import orc
from sklearn.datasets import load_breast_cancer

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    make_storage,
)
from secretflow.component.entry import comp_eval
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
    storage = make_storage(storage_config)

    x = load_breast_cancer()["data"]
    alice_columns = [f"a{i}" for i in range(15)]
    bob_columns = [f"b{i}" for i in range(15)]

    if self_party == "alice":
        ds = pd.DataFrame(x[:, :15], columns=alice_columns)
        ds.to_csv(storage.get_writer(alice_input_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=bob_columns)
        ds.to_csv(storage.get_writer(bob_input_path), index=False)

    param = build_node_eval_param(
        domain="preprocessing",
        name="binary_op",
        version="1.0.0",
        attrs={
            "input/input_ds/f1": f1,
            "input/input_ds/f2": f2,
            "binary_op": binary_op,
            "as_label": as_label,
            "new_feature_name": new_feature_name,
        },
        inputs=[
            VTable(
                name="input",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                        features={n: "float32" for n in alice_columns},
                    ),
                    VTableParty.from_dict(
                        uri=bob_input_path,
                        party="bob",
                        format="csv",
                        features={n: "float32" for n in bob_columns},
                    ),
                ],
            ),
        ],
        output_uris=[output_path, rule_path],
    )

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
        time.sleep(4)
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

    param2 = build_node_eval_param(
        domain="preprocessing",
        name="substitution",
        version="1.0.0",
        attrs=None,
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
