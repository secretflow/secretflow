import pandas as pd
from sklearn.datasets import load_breast_cancer

from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.filter.feature_filter import feature_filter_comp
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_feature_filter(comp_prod_sf_cluster_config):
    alice_input_path = "test_feature_filter/alice.csv"
    bob_input_path = "test_feature_filter/bob.csv"
    output_path = "test_feature_filter/out.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    x = load_breast_cancer()["data"]
    if self_party == "alice":
        ds = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        ds.to_csv(comp_storage.get_writer(alice_input_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(comp_storage.get_writer(bob_input_path), index=False)

    param = NodeEvalParam(
        domain="data_filter",
        name="feature_filter",
        version="0.0.1",
        attr_paths=[
            "input/in_ds/drop_features",
        ],
        attrs=[
            Attribute(ss=["a1", "b1", "a3", "b13"]),
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
        output_uris=[output_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
            ),
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)
    res = feature_filter_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    if self_party == "alice":
        a_out = pd.read_csv(comp_storage.get_reader(output_path))
        assert a_out.shape[1] == 13
        assert "a1" not in a_out.columns
        assert "a3" not in a_out.columns

    if self_party == "alice":
        b_out = pd.read_csv(comp_storage.get_reader(output_path))
        assert b_out.shape[1] == 13
        assert "b1" not in b_out.columns
        assert "b13" not in b_out.columns
