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
import pytest
from google.protobuf.json_format import MessageToJson
from secretflow_spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow_spec.v1.report_pb2 import Report

from secretflow.component.core import (
    DistDataType,
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.stats.groupby_statistics import STR_TO_ENUM, GroupbyStatistics
from secretflow.spec.extend.groupby_aggregation_config_pb2 import (
    ColumnQuery,
    GroupbyAggregationConfig,
)


def value_agg_pairs_to_pb(value_agg_pairs) -> GroupbyAggregationConfig:
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
@pytest.mark.mpc
def test_groupby_statistics(sf_production_setup_comp, by, value_agg_pairs):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    alice_input_path = "test_groupby_statistics/alice.csv"
    bob_input_path = "test_groupby_statistics/bob.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

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

    param = build_node_eval_param(
        domain="stats",
        name="groupby_statistics",
        version="1.0.0",
        attrs={
            "input/input_ds/by": by,
            "aggregation_config": MessageToJson(value_agg_pairs_to_pb(value_agg_pairs)),
        },
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
    logging.info(f"report {comp_ret}")
    assert len(value_agg_pairs) == len(comp_ret.tabs)
    for idx, item in enumerate(value_agg_pairs):
        (name, agg) = item
        tab = comp_ret.tabs[idx]
        assert tab.name == f'{name}_{agg}'
        tbl = tab.divs[0].children[0].table
        names = [h.name for h in tbl.headers]
        assert set(by + [name]) == set(names)


@pytest.mark.parametrize("by", [["a"]])
@pytest.mark.parametrize("value_agg_pairs", [[("d", "sum")]])
def test_groupby_statistics_forbid_on_label(by, value_agg_pairs):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    input_tbl = VTable(
        name="input_data",
        parties=[
            VTableParty.from_dict(
                uri="alice_input_path.csv",
                party="alice",
                format="csv",
                features={"c": "float32"},
                labels={"a": "str"},
            ),
            VTableParty.from_dict(
                uri="bob_input_path.csv",
                party="bob",
                format="csv",
                features={"b": "float32", "d": "float32"},
            ),
        ],
    )
    comp = GroupbyStatistics(
        aggregation_config=value_agg_pairs_to_pb(value_agg_pairs),
        by=by,
        input_ds=input_tbl.to_distdata(),
    )
    with pytest.raises(Exception, match=r"kind of .* mismatch, expected .*"):
        comp.evaluate(None)
