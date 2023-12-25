import logging
import operator
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.unified_single_party_ops.binary_op import (
    binary_op_comp,
)
from secretflow.component.preprocessing.unified_single_party_ops.substitution import (
    substitution,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from tests.conftest import TEST_STORAGE_ROOT

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
    local_fs_wd = storage_config.local_fs.wd

    x = load_breast_cancer()["data"]
    alice_columns = [f"a{i}" for i in range(15)]
    bob_columns = [f"b{i}" for i in range(15)]

    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, "test_binary_op"),
            exist_ok=True,
        )
        ds = pd.DataFrame(x[:, :15], columns=alice_columns)
        ds.to_csv(os.path.join(local_fs_wd, alice_input_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, "test_binary_op"),
            exist_ok=True,
        )
        ds = pd.DataFrame(x[:, 15:], columns=bob_columns)
        ds.to_csv(os.path.join(local_fs_wd, bob_input_path), index=False)

    param = NodeEvalParam(
        domain="preprocessing",
        name="binary_op",
        version="0.0.2",
        attr_paths=[
            "input/in_ds/f1",
            "input/in_ds/f2",
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
        with pytest.raises(AssertionError):
            res = binary_op_comp.eval(
                param=param,
                storage_config=storage_config,
                cluster_config=sf_cluster_config,
            )
        return
    else:
        res = binary_op_comp.eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

    assert len(res.outputs) == 2

    a_out = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", output_path))
    b_out = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "bob", output_path))
    a_in = pd.DataFrame(x[:, :15], columns=alice_columns)
    b_in = pd.DataFrame(x[:, 15:], columns=bob_columns)

    if features[0] in alice_columns:
        df = a_out
        df_in = a_in
    else:
        df = b_out
        df_in = b_in
    assert new_feature_name in df.columns

    item_0 = df_in[features[0]]
    item_1 = df_in[features[1]]

    expected_result = ops[binary_op](item_0, item_1)
    np.testing.assert_array_almost_equal(
        df[new_feature_name], expected_result, decimal=3
    )

    param2 = NodeEvalParam(
        domain="preprocessing",
        name="substitution",
        version="0.0.2",
        inputs=[param.inputs[0], res.outputs[1]],
        output_uris=[sub_path],
    )

    res = substitution.eval(
        param=param2,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    a_out = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", sub_path))
    logging.warning(f"....... \n{a_out}\n.,......")
