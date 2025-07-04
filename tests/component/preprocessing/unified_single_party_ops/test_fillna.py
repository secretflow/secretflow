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
import pyarrow as pa
import pytest
from pyarrow import orc

import secretflow.compute as sc
from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.preprocessing.unified_single_party_ops.fillna import (
    apply_fillna_rule_on_table,
    fit_col,
)


def test_apply():
    col_i = pa.array([2, 4, 999, None])  # .cast(pa.int64())
    col_f = pa.array([1.0, float("nan"), 999.9, None])
    col_s = pa.array(["a", "b", "zzzz", None])
    col_b = pa.array([True, True, False, None])

    table = pa.Table.from_arrays(
        [col_i, col_f, col_s, col_b], names=["i", "f", "s", "b"]
    )

    fill_rules = {
        "i": {"outlier_values": [999], "fill_value": int(10)},
        "f": {"outlier_values": [999.9], "fill_value": float(10.5)},
        "s": {"outlier_values": ["zzzz"], "fill_value": "z"},
        "b": {"outlier_values": [], "fill_value": False},
    }

    out = apply_fillna_rule_on_table(
        sc.Table.from_pyarrow(table), {"nan_is_null": False, "fill_rules": fill_rules}
    ).to_table()

    assert [v.as_py() for v in out.column("i")] == [2, 4, 10, 10]
    assert [v.as_py() for v in out.column("f")][2:] == [10.5, 10.5]
    assert np.isnan([v.as_py() for v in out.column("f")])[1]
    assert [v.as_py() for v in out.column("s")] == ["a", "b", "z", "z"]
    assert [v.as_py() for v in out.column("b")] == [True, True, False, False]

    out = apply_fillna_rule_on_table(
        sc.Table.from_pyarrow(table), {"nan_is_null": True, "fill_rules": fill_rules}
    ).to_table()

    assert [v.as_py() for v in out.column("i")] == [2, 4, 10, 10]
    assert [v.as_py() for v in out.column("f")] == [1.0, 10.5, 10.5, 10.5]
    assert [v.as_py() for v in out.column("s")] == ["a", "b", "z", "z"]
    assert [v.as_py() for v in out.column("b")] == [True, True, False, False]


@pytest.mark.parametrize("strategy_count", range(4))
def test_fit(strategy_count):
    col_i = pa.array([2, 2, 4, 999, None])
    col_f = pa.array([1.0, 1.0, float("nan"), 999.9, None])
    col_s = pa.array(["a", "a", "b", "zzzz", None])
    col_b = pa.array([True, True, True, False, None])

    table = pa.Table.from_arrays(
        [col_i, col_f, col_s, col_b], names=["i", "f", "s", "b"]
    )

    strategies = ["constant", "most_frequent", "mean", "median"]
    str_fill_strategy = strategies[int(strategy_count / 2)]
    others_strategy = strategies[strategy_count]

    col = table.column("i")
    i_r = fit_col("i", col, [999], others_strategy, 10, False)
    col = table.column("f")
    f_r = fit_col("f", col, [999.9], others_strategy, 10.5, False)
    col = table.column("s")
    s_r = fit_col("s", col, ["zzzz"], str_fill_strategy, "z", False)
    col = table.column("b")
    b_r = fit_col("b", col, [], str_fill_strategy, False, False)

    assert s_r["outlier_values"] == ["zzzz"]
    assert b_r["outlier_values"] == []
    if str_fill_strategy == "constant":
        assert s_r["fill_value"] == "z"
        assert b_r["fill_value"] == False
    else:
        assert s_r["fill_value"] == "a"
        assert b_r["fill_value"] == True

    assert i_r["outlier_values"] == [999]
    assert f_r["outlier_values"] == [999.9]
    if others_strategy == "constant":
        assert i_r["fill_value"] == 10
        assert f_r["fill_value"] == 10.5
    elif others_strategy == "most_frequent":
        assert i_r["fill_value"] == 2
        assert f_r["fill_value"] == 1.0
    elif others_strategy == "mean":
        assert i_r["fill_value"] == 3
        assert f_r["fill_value"] == 1.0
    elif others_strategy == "median":
        assert i_r["fill_value"] == 2
        assert f_r["fill_value"] == 1.0


@pytest.mark.parametrize("nan_is_null", [True, False])
@pytest.mark.parametrize("strategy_count", range(4))
@pytest.mark.mpc
def test_fillna(sf_production_setup_comp, nan_is_null, strategy_count):
    alice_input_path = "test_fillna/alice.csv"
    bob_input_path = "test_fillna/bob.csv"
    rule_path = "test_fillna/fillna.rule"
    sub_path = "test_fillna/substitution.orc"
    sub_comp = "test_fillna/sub_comp.orc"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        a_csv = (
            "ida,ai,af,as,ab",
            "1,10,1.5,aa,true",
            "2,20,2.5,aa,true",
            "3,20,2.5,bb,false",
            "4,99,99.9,zzzz,NULL",
            "5,NULL,nan,NULL,NULL",
        )
        with storage.get_writer(alice_input_path) as w:
            w.write("\n".join(a_csv).encode())

    if self_party == "bob":
        b_csv = (
            "idb,bi,bf,bs,bb",
            "1,100,22.5,aaa,false",
            "2,100,22.5,aaa,false",
            "3,100,22.5,aaa,true",
            "4,999,999.9,kkkk,NULL",
            "5,NULL,NULL,NULL,NULL",
        )
        with storage.get_writer(bob_input_path) as w:
            w.write("\n".join(b_csv).encode())

    strategies = ["constant", "most_frequent", "mean", "median"]

    str_fill_strategy = strategies[int(strategy_count % 2)]
    others_strategy = strategies[strategy_count]

    fill_na_features = ["ai", "af", "as", "ab", "bi", "bf", "bs", "bb"]
    fill_param = build_node_eval_param(
        domain="preprocessing",
        name="fillna",
        version="1.0.0",
        attrs={
            "nan_is_null": nan_is_null,
            "float_outliers": [99.9, 999.9],
            "int_outliers": [99, 999],
            "str_outliers": ["zzzz", "kkkk"],
            "str_fill_strategy": str_fill_strategy,
            "fill_value_str": "fill_str_v",
            "int_fill_strategy": others_strategy,
            "fill_value_int": 12121,
            "float_fill_strategy": others_strategy,
            "fill_value_float": 12121.5,
            "bool_fill_strategy": str_fill_strategy,
            "fill_value_bool": False,
            "input/input_ds/fill_na_features": fill_na_features,
        },
        inputs=[
            VTable(
                name="input_data",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                        null_strs=["NULL"],
                        ids={"ida": "str"},
                        features={
                            "ai": "int32",
                            "af": "float32",
                            "as": "str",
                            "ab": "bool",
                        },
                    ),
                    VTableParty.from_dict(
                        uri=bob_input_path,
                        party="bob",
                        format="csv",
                        null_strs=["NULL"],
                        ids={"idb": "str"},
                        features={
                            "bi": "int32",
                            "bf": "float32",
                            "bs": "str",
                            "bb": "bool",
                        },
                    ),
                ],
            )
        ],
        output_uris=[
            sub_path,
            rule_path,
        ],
    )

    res = comp_eval(
        param=fill_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    sub_param = build_node_eval_param(
        domain="preprocessing",
        name="substitution",
        version="1.0.0",
        attrs=None,
        inputs=[fill_param.inputs[0], res.outputs[1]],
        output_uris=[sub_comp],
    )

    res = comp_eval(
        param=sub_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    def to_list(i):
        return [v.as_py() for v in i]

    if self_party == "alice":
        a_out = orc.read_table(storage.get_reader(sub_path))
        a_sub = orc.read_table(storage.get_reader(sub_comp))
        if nan_is_null:
            a_sub = a_sub.select(a_out.column_names)  # ignore columns order
            assert a_out.equals(a_sub)
        else:
            # nan always != nan
            pass

        assert to_list(a_out.column("ida")) == ["1", "2", "3", "4", "5"]
        a_s = to_list(a_out.column("as"))
        a_b = to_list(a_out.column("ab"))
        a_i = to_list(a_out.column("ai"))
        a_f = to_list(a_out.column("af"))
        if others_strategy == "most_frequent":
            assert a_s == ["aa", "aa", "bb", "aa", "aa"]
            assert a_b == [True, True, False, True, True]
            assert a_i == [10, 20, 20, 20, 20]
            if nan_is_null:
                assert a_f == [1.5, 2.5, 2.5, 2.5, 2.5]
            else:
                assert np.isnan(a_f[4])
        elif others_strategy == "constant":
            assert a_s == ["aa", "aa", "bb", "fill_str_v", "fill_str_v"]
            assert a_b == [True, True, False, False, False]
            assert a_i == [10, 20, 20, 12121, 12121]
            if nan_is_null:
                assert a_f == [1.5, 2.5, 2.5, 12121.5, 12121.5]
            else:
                assert np.isnan(a_f[4])
        elif others_strategy == "mean":
            assert a_i == [10, 20, 20, 17, 17]
            if nan_is_null:
                assert a_f == [1.5, 2.5, 2.5, 6.5 / 3, 6.5 / 3]
            else:
                assert np.isnan(a_f[4])
        else:
            assert a_i == [10, 20, 20, 20, 20]
            if nan_is_null:
                assert a_f == [1.5, 2.5, 2.5, 2.5, 2.5]
            else:
                assert np.isnan(a_f[4])

    if self_party == "bob":
        b_out = orc.read_table(storage.get_reader(sub_path))
        b_sub = orc.read_table(storage.get_reader(sub_comp))
        if nan_is_null:
            b_sub = b_sub.select(b_out.column_names)  # ignore columns order
            assert b_out.equals(b_sub)
        else:
            # nan always != nan
            pass

        assert to_list(b_out.column("idb")) == ["1", "2", "3", "4", "5"]
        b_s = to_list(b_out.column("bs"))
        b_b = to_list(b_out.column("bb"))
        b_i = to_list(b_out.column("bi"))
        b_f = to_list(b_out.column("bf"))
        if others_strategy == "most_frequent":
            assert b_s == ["aaa", "aaa", "aaa", "aaa", "aaa"]
            assert b_b == [False, False, True, False, False]
            assert b_i == [100, 100, 100, 100, 100]
            assert b_f == [22.5, 22.5, 22.5, 22.5, 22.5]
        elif others_strategy == "constant":
            assert b_s == ["aaa", "aaa", "aaa", "fill_str_v", "fill_str_v"]
            assert b_b == [False, False, True, False, False]
            assert b_i == [100, 100, 100, 12121, 12121]
            assert b_f == [22.5, 22.5, 22.5, 12121.5, 12121.5]
        elif others_strategy == "mean":
            assert b_i == [100, 100, 100, 100, 100]
            assert b_f == [22.5, 22.5, 22.5, 22.5, 22.5]
        else:
            assert b_i == [100, 100, 100, 100, 100]
            assert b_f == [22.5, 22.5, 22.5, 22.5, 22.5]
