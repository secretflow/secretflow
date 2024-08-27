# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from google.protobuf.json_format import MessageToJson
from pyarrow import orc
from sklearn.datasets import load_breast_cancer

from secretflow.component.data_utils import DistDataType
from secretflow.component.io.io import io_write_data
from secretflow.component.ml.boost.sgb.sgb import sgb_predict_comp
from secretflow.component.storage.storage import ComponentStorage
from secretflow.spec.extend.xgb_model_pb2 import (
    Common,
    EnsembleLeafWeights,
    EnsembleSplitTrees,
    Features,
    SplitTree,
    TreeLeafWeights,
    XgbModel,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def mock_xgb_model_pb():
    xgb_model = XgbModel()
    xgb_model.label_holder = "alice"
    common = Common()
    common.base = 0.5
    common.link = "logit"
    alice_features = Features()
    alice_features.feature_names.extend(
        [
            'a0',
            'a1',
            'a2',
            'a3',
            'a4',
            'a5',
            'a6',
            'a7',
            'a8',
            'a9',
            'a10',
            'a11',
            'a12',
            'a13',
            'a14',
        ]
    )
    bob_features = Features()
    bob_features.feature_names.extend(
        [
            'b0',
            'b1',
            'b2',
            'b3',
            'b4',
            'b5',
            'b6',
            'b7',
            'b8',
            'b9',
            'b10',
            'b11',
            'b12',
            'b13',
            'b14',
        ]
    )
    partition_column = {
        "alice": alice_features,
        "bob": bob_features,
    }
    for key, value in partition_column.items():
        common.partition_column[key].CopyFrom(value)

    common.tree_num = 3
    xgb_model.common.CopyFrom(common)

    tree0_leaf_weights = TreeLeafWeights()
    tree0_leaf_weights.weights.extend(
        [
            0.33809237484295707,
            0.44480961815798964,
            0.07548405576166715,
            0.20886833442384128,
            -0.7026121766821838,
            0.33809237484295707,
            -0.774172543767616,
        ]
    )
    tree1_leaf_weights = TreeLeafWeights()
    tree1_leaf_weights.weights.extend(
        [
            0.41484053264989285,
            0.07669893947553322,
            -0.32633843601979995,
            0.3268590816826702,
            0.3392502418428151,
            -0.40940840950913954,
            -0.011439537057913927,
            -0.5150791906345678,
        ]
    )
    tree2_leaf_weights = TreeLeafWeights()
    tree2_leaf_weights.weights.extend(
        [
            0.3645181610173817,
            0.3952816293634549,
            -0.4742374677044488,
            0.29726246409572593,
            -0.3101663875858056,
            0.12095664258244537,
            -0.40535261423916374,
        ]
    )
    ensemble_leaf_weights = EnsembleLeafWeights()
    ensemble_leaf_weights.tree_leaf_weights.extend(
        [tree0_leaf_weights, tree1_leaf_weights, tree2_leaf_weights]
    )

    xgb_model.ensemble_leaf_weights.CopyFrom(ensemble_leaf_weights)

    tree0 = SplitTree()
    tree0.split_features.extend([7, 0, 2, 7, 1, 6])
    tree0.split_values.extend([0.074, 15.78, 75.17, 0.0335, 16.17, 0.06154])
    tree0.split_indices.extend([0, 1, 2, 3, 4, 6])
    tree0.leaf_indices.extend([5, 7, 8, 9, 10, 13, 14])

    tree1 = SplitTree()
    tree1.split_features.extend([-1, -1, -1, 13, 2, -1, -1])
    tree1.split_values.extend(
        [
            float('inf'),
            float('inf'),
            float('inf'),
            45.19,
            86.24,
            float('inf'),
            float('inf'),
        ]
    )
    tree1.split_indices.extend([0, 1, 2, 3, 4, 5, 6])
    tree1.leaf_indices.extend([7, 8, 9, 10, 11, 12, 13, 14])

    tree2 = SplitTree()
    tree2.split_features.extend([-1, -1, -1, -1, -1, -1])
    tree2.split_values.extend(
        [
            float('inf'),
            float('inf'),
            float('inf'),
            float('inf'),
            float('inf'),
            float('inf'),
        ]
    )
    tree2.split_indices.extend([0, 1, 2, 4, 5, 6])
    tree2.leaf_indices.extend([3, 9, 10, 11, 12, 13, 14])

    alice_ensemble_trees = EnsembleSplitTrees()
    alice_ensemble_trees.split_trees.extend([tree0, tree1, tree2])

    tree0 = SplitTree()
    tree0.split_features.extend([-1, -1, -1, -1, -1, -1])
    tree0.split_values.extend(
        [
            float('inf'),
            float('inf'),
            float('inf'),
            float('inf'),
            float('inf'),
            float('inf'),
        ]
    )
    tree0.split_indices.extend([0, 1, 2, 3, 4, 6])
    tree0.leaf_indices.extend([5, 7, 8, 9, 10, 13, 14])

    tree1 = SplitTree()
    tree1.split_features.extend([8, 12, 11, -1, -1, 5, 9])
    tree1.split_values.extend(
        [686.5, 0.1614, 0.2267, float('inf'), float('inf'), 18.79, 0.1166]
    )
    tree1.split_indices.extend([0, 1, 2, 3, 4, 5, 6])
    tree1.leaf_indices.extend([7, 8, 9, 10, 11, 12, 13, 14])

    tree2 = SplitTree()
    tree2.split_features.extend([8, 12, 11, 6, 6, 6])
    tree2.split_values.extend([686.5, 0.1614, 0.2267, 25.41, 29.72, 21.08])
    tree2.split_indices.extend([0, 1, 2, 4, 5, 6])
    tree2.leaf_indices.extend([3, 9, 10, 11, 12, 13, 14])

    bob_ensemble_trees = EnsembleSplitTrees()
    bob_ensemble_trees.split_trees.extend([tree0, tree1, tree2])

    ensemble_split_trees = {
        "alice": alice_ensemble_trees,
        "bob": bob_ensemble_trees,
    }

    for key, value in ensemble_split_trees.items():
        xgb_model.ensemble_split_trees[key].CopyFrom(value)

    write_data = MessageToJson(xgb_model)
    return write_data


def test_write_data(comp_prod_sf_cluster_config):
    new_sgb_model_path = "test_io/new_sgb_model"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    write_data = mock_xgb_model_pb()
    write_param = NodeEvalParam(
        domain="io",
        name="write_data",
        version="0.0.1",
        attr_paths=["write_data", "write_data_type"],
        attrs=[
            Attribute(s=write_data),
            Attribute(s=str(DistDataType.SGB_MODEL)),
        ],
        inputs=[
            DistData(name="null", type=str(DistDataType.NULL)),
        ],
        output_uris=[new_sgb_model_path],
    )
    write_res = io_write_data.eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    assert len(write_res.outputs) == 1

    alice_input_path = "test_io/alice.csv"
    bob_input_path = "test_io/bob.csv"
    predict_path = "test_io/predict.csv"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    ds = load_breast_cancer()
    x, y = ds["data"], ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(comp_storage.get_writer(alice_input_path), index=False)
    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(comp_storage.get_writer(bob_input_path), index=False)

    predict_param = NodeEvalParam(
        domain="ml.predict",
        name="sgb_predict",
        version="0.0.3",
        attr_paths=[
            "receiver",
            "save_ids",
            "save_label",
        ],
        attrs=[
            Attribute(ss=["alice"]),
            Attribute(b=False),
            Attribute(b=True),
        ],
        inputs=[
            write_res.outputs[0],
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[predict_path],
    )
    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
                labels=["y"],
                label_types=["float32"],
            ),
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = sgb_predict_comp.eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    if "alice" == sf_cluster_config.private_config.self_party:
        comp_storage = ComponentStorage(storage_config)
        input_y = pd.read_csv(comp_storage.get_reader(alice_input_path))
        output_y = orc.read_table(comp_storage.get_reader(predict_path)).to_pandas()

        # label & pred
        assert output_y.shape[1] == 2
        assert set(output_y.columns) == set(["pred", "y"])
        assert input_y.shape[0] == output_y.shape[0]

        predict_reference_path = "tests/component/io/predict_reference.csv.txt"
        predict_reference = pd.read_csv(
            predict_reference_path, dtype={"pred": "float32"}
        )
        assert output_y["pred"].equals(
            predict_reference["pred"]
        ), 'output_y["pred"] != predict_reference["pred"]'
