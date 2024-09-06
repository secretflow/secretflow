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


import pandas as pd
import pytest
from google.protobuf.json_format import MessageToJson

from secretflow.component.core import DistDataType, Storage
from secretflow.component.entry import comp_eval
from secretflow.component.stats.groupby_statistics import STR_TO_ENUM
from secretflow.spec.extend.groupby_aggregation_config_pb2 import (
    ColumnQuery,
    GroupbyAggregationConfig,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report


def value_agg_pairs_to_pb(value_agg_pairs) -> GroupbyAggregationConfig:  # type: ignore
    config = GroupbyAggregationConfig()
    for value, agg in value_agg_pairs:
        col_query = ColumnQuery()
        col_query.function = STR_TO_ENUM[agg]
        col_query.column_name = value
        config.column_queries.append(col_query)
    return config


# note that the report does not support approximatedly equal yet, we only test easy case, for more numeric tests see tests for groupby in tests/data/
# note nan values are zeros for spu.
@pytest.mark.parametrize("by", [["a"], ["a", "b"]])
@pytest.mark.parametrize(
    "value_agg_pairs", [[("c", "sum")], [("c", "count"), ("d", "sum")]]
)
def test_groupby_statistics(comp_prod_sf_cluster_config, by, value_agg_pairs):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    alice_input_path = "test_groupby_statistics/alice.csv"
    bob_input_path = "test_groupby_statistics/bob.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    storage = Storage(storage_config)

    test_data = pd.DataFrame(
        {
            "a": ['9', '6', '5', '5'],
            "b": [5, 5, 6, 7],
            "c": [1, 1, 2, 4],
            "d": [11, 55, 1, 99],
        }
    )
    test_data = test_data.astype("float32")
    test_data["a"] = test_data["a"].astype("string")

    if self_party == "alice":
        df_alice = test_data[["a", "c"]]
        df_alice.to_csv(storage.get_writer(alice_input_path), index=False)
    elif self_party == "bob":
        df_bob = test_data[["b", "d"]]
        df_bob.to_csv(storage.get_writer(bob_input_path), index=False)

    param = NodeEvalParam(
        domain="stats",
        name="groupby_statistics",
        version="1.0.0",
        attr_paths=["by", "aggregation_config"],
        attrs=[
            Attribute(ss=by),
            Attribute(s=MessageToJson(value_agg_pairs_to_pb(value_agg_pairs))),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                ],
            )
        ],
        output_uris=[""],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["str", "float32"],
                features=["a", "c"],
            ),
            TableSchema(
                feature_types=["float32", "float32"],
                features=["b", "d"],
            ),
        ],
    )

    param.inputs[0].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    # logging.info(f"report {comp_ret}")
    assert len(value_agg_pairs) == len(comp_ret.tabs)
    for idx, item in enumerate(value_agg_pairs):
        (name, agg) = item
        tab = comp_ret.tabs[idx]
        assert tab.name == f'{name}_{agg}'
        tbl = tab.divs[0].children[0].table
        names = [h.name for h in tbl.headers]
        assert set(by + [name]) == set(names)
