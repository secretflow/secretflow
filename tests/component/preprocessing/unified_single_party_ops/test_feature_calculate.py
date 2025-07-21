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

import numpy as np
import pandas as pd
import pytest
from google.protobuf.json_format import MessageToJson
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from secretflow.component.core import (
    DistDataType,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.spec.extend.calculate_rules_pb2 import CalculateOpRules

test_data_alice = pd.DataFrame(
    {
        "a1": [i * (-0.8) for i in range(3)],
        "a2": [0.1] * 3,
        "a3": ["AAA", "BBB", "CCC"],
    }
)

test_data_bob = pd.DataFrame(
    {
        "b1": [i for i in range(3)],
    }
)


def _almost_equal(df1, df2, rtol=1.0e-5):
    try:
        pd.testing.assert_frame_equal(df1, df2, rtol=rtol, check_dtype=False)
        return True
    except AssertionError:
        logging.exception("assert_frame_equal failed")
        return False


def _build_test():
    names = []
    tests = []
    features = []
    expected = []

    # --------TEST STANDARDIZE---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.STANDARDIZE
    scaler = StandardScaler()

    alice_data = test_data_alice.copy()
    alice_feature = ['a1', 'a2']
    alice_data[alice_feature] = scaler.fit_transform(alice_data[alice_feature])

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data[bob_feature] = scaler.fit_transform(bob_data[bob_feature])

    names.append("standardize")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST NORMALIZATION---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.NORMALIZATION
    scaler = MinMaxScaler()

    alice_data = test_data_alice.copy()
    alice_feature = ['a1', 'a2']
    alice_data[alice_feature] = scaler.fit_transform(alice_data[alice_feature])

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data[bob_feature] = scaler.fit_transform(bob_data[bob_feature])

    names.append("normalization")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST RANGE_LIMIT---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.RANGE_LIMIT
    rule.operands.extend(["1", "2"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data.loc[alice_data['a1'] < 1, 'a1'] = 1
    alice_data.loc[alice_data['a1'] > 2, 'a1'] = 2

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data.loc[bob_data['b1'] < 1, 'b1'] = 1
    bob_data.loc[bob_data['b1'] > 2, 'b1'] = 2
    bob_data = bob_data.astype(float)

    names.append("range_limit")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST UNARY(+)---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.UNARY
    rule.operands.extend(["+", "+", "1"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = alice_data['a1'] + 1.0

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = bob_data['b1'] + 1.0

    names.append("unary_+")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST UNARY(-)---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.UNARY
    rule.operands.extend(["+", "-", "1"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = alice_data['a1'] - 1.0

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = bob_data['b1'] - 1.0

    names.append("unary_-")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST UNARY(reverse-)---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.UNARY
    rule.operands.extend(["-", "-", "1"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = 1.0 - alice_data['a1']

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = 1.0 - bob_data['b1']

    names.append("unary_reverse_-")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST UNARY(*)---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.UNARY
    rule.operands.extend(["+", "*", "2"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = alice_data['a1'] * 2.0

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = bob_data['b1'] * 2.0

    names.append("unary_*")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST UNARY(/)---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.UNARY
    rule.operands.extend(["+", "/", "3"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = alice_data['a1'] / 3.0

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = bob_data['b1'] / 3.0

    names.append("unary_/")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST UNARY(reverse/)---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.UNARY
    rule.operands.extend(["-", "/", "3"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = 3.0 / alice_data['a1']

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = 3.0 / bob_data['b1']

    names.append("unary_reverse_/")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST RECIPROCAL---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.RECIPROCAL

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = 1.0 / alice_data['a1']

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = 1.0 / bob_data['b1']

    names.append("reciprocal")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST ROUND---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.ROUND

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = alice_data['a1'].round()

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = bob_data['b1'].round()

    names.append("round")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST LOGROUND---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.LOG_ROUND
    rule.operands.extend(["10"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = np.log2(alice_data['a1'] + 10)
    alice_data['a1'] = alice_data['a1'].round()

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = np.log2(bob_data['b1'] + 10)
    bob_data['b1'] = bob_data['b1'].round()

    names.append("log_round")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST SQRT---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.SQRT

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = np.sqrt(alice_data['a1'])

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = np.sqrt(bob_data['b1'])

    names.append("sqrt")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST LOG BASE E---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.LOG
    rule.operands.extend(["e", "10"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = np.log(alice_data['a1'] + 10)

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = np.log(bob_data['b1'] + 10)

    names.append("log base e")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST LOG BASE 2---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.LOG
    rule.operands.extend(["2", "10"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = np.log(alice_data['a1'] + 10) / np.log(2)

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = np.log(bob_data['b1'] + 10) / np.log(2)

    names.append("log base 2")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST EXP---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.EXP

    alice_data = test_data_alice.copy()
    alice_feature = ['a1']
    alice_data['a1'] = np.exp(alice_data['a1'])

    bob_data = test_data_bob.copy()
    bob_feature = ['b1']
    bob_data['b1'] = np.exp(bob_data['b1'])

    names.append("exp")
    tests.append(rule)
    features.append(alice_feature + bob_feature)
    expected.append((alice_data, bob_data))

    # --------TEST LENGTH---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.LENGTH

    alice_data = test_data_alice.copy()
    alice_feature = ['a3']
    alice_data['a3'] = alice_data['a3'].str.len()

    names.append("length")
    tests.append(rule)
    features.append(alice_feature)
    expected.append((alice_data, test_data_bob))

    # --------TEST SUBSTR---------
    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.SUBSTR
    rule.operands.extend(["0", "2"])

    alice_data = test_data_alice.copy()
    alice_feature = ['a3']
    alice_data['a3'] = alice_data['a3'].str[:2]

    names.append("substr")
    tests.append(rule)
    features.append(alice_feature)
    expected.append((alice_data, test_data_bob))

    return names, tests, features, expected


@pytest.mark.mpc
def test_feature_calculate(sf_production_setup_comp):
    alice_input_path = "test_feature_calculate/alice.csv"
    bob_input_path = "test_feature_calculate/bob.csv"
    out_path = "test_feature_calculate/out.csv"
    rule_path = "test_feature_calculate/feature_calculate.rule"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)
    df_alice = pd.DataFrame()
    if self_party == "alice":
        df_alice = test_data_alice
        df_alice.to_csv(storage.get_writer(alice_input_path), index=False)
    elif self_party == "bob":
        df_bob = test_data_bob
        df_bob.to_csv(storage.get_writer(bob_input_path), index=False)

    param = build_node_eval_param(
        domain="preprocessing",
        name="feature_calculate",
        version="1.0.0",
        attrs={
            "rules": "{}",
            "input/input_ds/features": [""],
        },
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                ],
            )
        ],
        output_uris=[
            out_path,
            rule_path,
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["int32"],
                features=["b1"],
            ),
            TableSchema(
                feature_types=[
                    "float64",
                    "float64",
                    "str",
                ],
                features=["a1", "a2", "a3"],
            ),
        ],
    )

    param.inputs[0].meta.Pack(meta)
    for n, t, f, e in zip(*_build_test()):
        param.attrs[0].s = MessageToJson(t)
        param.attrs[1].ss[:] = f

        res = comp_eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        assert len(res.outputs) == 2

        if self_party == "alice":
            alice_out = orc.read_table(storage.get_reader(out_path)).to_pandas()
            logging.warning(f"----> {alice_out}")
            assert _almost_equal(
                alice_out, e[0]
            ), f"{n}\n===out===\n{alice_out}\n===e===\n{e[0]}\n===r===\n{param.attrs[0].s}"

        if self_party == "bob":
            bob_out = orc.read_table(storage.get_reader(out_path)).to_pandas()
            assert _almost_equal(
                bob_out, e[1]
            ), f"{n}\n===out===\n{bob_out}\n===e===\n{e[1]}\n===r===\n{param.attrs[0].s}"

        assert len(res.outputs) == 2
        break
