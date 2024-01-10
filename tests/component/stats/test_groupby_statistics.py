import logging
import os

import pandas as pd
import pytest
from google.protobuf.json_format import MessageToJson

from secretflow.component.data_utils import DistDataType
from secretflow.component.stats.groupby_statistics import (
    STR_TO_ENUM,
    gen_groupby_statistic_reports,
    groupby_statistics_comp,
)
from secretflow.spec.extend.groupby_aggregation_config_pb2 import (
    ColumnQuery,
    GroupbyAggregationConfig,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report


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
def test_groupby_statistics(comp_prod_sf_cluster_config, by, value_agg_pairs):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    alice_input_path = "test_groupby_statistics/alice.csv"
    bob_input_path = "test_groupby_statistics/bob.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

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
        os.makedirs(os.path.join(local_fs_wd, "test_groupby_statistics"), exist_ok=True)
        df_alice.to_csv(os.path.join(local_fs_wd, alice_input_path), index=False)
    elif self_party == "bob":
        df_bob = test_data[["b", "d"]]
        os.makedirs(os.path.join(local_fs_wd, "test_groupby_statistics"), exist_ok=True)
        df_bob.to_csv(os.path.join(local_fs_wd, bob_input_path), index=False)

    logging.info("data preparation complete")
    param = NodeEvalParam(
        domain="stats",
        name="groupby_statistics",
        version="0.0.3",
        attr_paths=["input/input_data/by", "aggregation_config"],
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

    res = groupby_statistics_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.info(comp_ret)

    result_true = {}
    for value, agg in value_agg_pairs:
        true_df = getattr(test_data.groupby(by), agg)()[value].fillna(0).reset_index()
        true_df.columns = by + [value]
        result_true[value + "_" + agg] = true_df
    true_ret = gen_groupby_statistic_reports(result_true, by)

    assert comp_ret == true_ret, f"comp_ret {comp_ret}, \n true {true_ret}"
