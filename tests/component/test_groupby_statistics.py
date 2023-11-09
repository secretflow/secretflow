import logging
import os

import pandas as pd
import pytest

from secretflow.component.data_utils import DistDataType
from secretflow.component.stats.groupby_statistics import (
    gen_groupby_statistic_reports,
    groupby_statistics_comp,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, TableSchema
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report


# note that the report does not support approximatedly equal yet, we only test easy case, for more numeric tests see tests for groupby in tests/data/
# note nan values are zeros for spu.
@pytest.mark.parametrize("by", [["a"], ["a", "b"]])
@pytest.mark.parametrize(
    "target",
    [
        ["c"],
        ["c", "d"],
    ],
)
@pytest.mark.parametrize("aggs", [["sum"], ["count", "sum"]])
def test_groupby_statistics(comp_prod_sf_cluster_config, by, target, aggs):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    alice_input_path = "test_groupby_statistics/alice.csv"

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
        df_alice = test_data
        os.makedirs(os.path.join(local_fs_wd, "test_groupby_statistics"), exist_ok=True)
        df_alice.to_csv(os.path.join(local_fs_wd, alice_input_path), index=False)

    param = NodeEvalParam(
        domain="stats",
        name="groupby_statistics",
        version="0.0.2",
        attr_paths=[
            "input/input_data/by",
            "input/input_data/values",
            "aggs",
        ],
        attrs=[
            Attribute(ss=by),
            Attribute(ss=target),
            Attribute(ss=aggs),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv")
                ],
            )
        ],
        output_uris=[""],
    )
    meta = IndividualTable(
        schema=TableSchema(
            feature_types=["str", "float32", "float32", "float32"],
            features=["a", "b", "c", "d"],
        )
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
    for agg in aggs:
        true_df = getattr(test_data.groupby(by), agg)()[target].fillna(0).reset_index()
        true_df.columns = by + target
        result_true[agg] = true_df
    true_ret = gen_groupby_statistic_reports(result_true)

    assert comp_ret == true_ret, f"comp_ret {comp_ret}, \n true {true_ret}"
