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
import pandas as pd
import pytest

from secretflow.data import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow_fl.preprocessing.cond_filter_v import ConditionFilter
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob

    vdf_alice = pd.DataFrame(
        {
            'a1': ['K5', 'K1', None, 'K6'],
            'a2': ['A5', 'A1', 'A2', 'A6'],
            'a3': [5, 1, 2, 6],
        }
    )

    vdf_bob = pd.DataFrame(
        {
            'b4': [10.2, 20.5, 12.3, -0.4],
            'b5': ['B3', None, 'B9', 'B4'],
            'b6': [3, 1, 9, 4],
        }
    )

    vdf = VDataFrame(
        {
            pyu_alice: partition(data=pyu_alice(lambda: vdf_alice)()),
            pyu_bob: partition(data=pyu_bob(lambda: vdf_bob)()),
        }
    )

    return sf_production_setup_devices, {
        'vdf_alice': vdf_alice,
        'vdf_bob': vdf_bob,
        'vdf': vdf,
    }


def test_constructor_valid_values():
    filter = ConditionFilter("field1", "==", "STRING", ["value"], 0.1)
    np.testing.assert_equal(filter.field_name, "field1")
    np.testing.assert_equal(filter.comparator, "==")
    np.testing.assert_equal(filter.value_type, "STRING")
    np.testing.assert_equal(filter.bound_value, ["value"])
    np.testing.assert_equal(filter.float_epsilon, 0.1)


def test_constructor_invalid_comparator():
    with np.testing.assert_raises(ValueError):
        ConditionFilter("field1", "INVALID", "STRING", ["value"], 0.1)


def test_constructor_invalid_value_type():
    with np.testing.assert_raises(ValueError):
        ConditionFilter("field1", "==", "INVALID", ["value"], 0.1)


def test_constructor_invalid_bound_value():
    with np.testing.assert_raises(ValueError):
        ConditionFilter("field1", "==", "STRING", ["value1", "value2"], 0.1)


@pytest.mark.mpc
def test_fit_valid_df(prod_env_and_data):
    filter = ConditionFilter("b5", "IN", "STRING", ["B9"], 0.1)
    env, data = prod_env_and_data
    df = data['vdf']
    filter = filter.fit(df)
    np.testing.assert_equal(type(filter), ConditionFilter)
    np.testing.assert_(filter.in_table is not None)
    np.testing.assert_(filter.out_table is not None)


@pytest.mark.mpc
def test_transform_valid_df(prod_env_and_data):
    filter = ConditionFilter("a3", "<", "FLOAT", ["3.14"], 0.1)
    env, data = prod_env_and_data
    df = data['vdf']
    filter = filter.fit(df)
    result = filter.transform(df)
    np.testing.assert_equal(type(result), VDataFrame)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 2)


@pytest.mark.mpc
def test_transform_valid_df_float(prod_env_and_data):
    filter = ConditionFilter("b4", "==", "FLOAT", ["10.1"], 0.1)
    env, data = prod_env_and_data
    df = data['vdf']
    filter = filter.fit(df)
    result = filter.transform(df)
    np.testing.assert_equal(type(result), VDataFrame)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 1)


@pytest.mark.mpc
def test_transform_valid_df_float_in(prod_env_and_data):
    filter = ConditionFilter("b4", "IN", "FLOAT", ["10.1", "20.4"], 0.15)
    env, data = prod_env_and_data
    df = data['vdf']
    filter = filter.fit(df)
    result = filter.transform(df)
    np.testing.assert_equal(type(result), VDataFrame)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 2)


@pytest.mark.mpc
def test_fit_transform_valid_df(prod_env_and_data):
    filter = ConditionFilter("b4", "<", "FLOAT", ["11"], 0.1)
    env, data = prod_env_and_data
    df = data['vdf']
    result = filter.fit_transform(df)
    np.testing.assert_equal(type(result), VDataFrame)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 2)


@pytest.mark.mpc
def test_fit_transform_valid_df_2(prod_env_and_data):
    filter = ConditionFilter("b6", "==", "FLOAT", ["1"], 0.1)
    env, data = prod_env_and_data
    df = data['vdf']
    result = filter.fit_transform(df)
    np.testing.assert_equal(type(result), VDataFrame)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 1)


@pytest.mark.mpc
def test_fit_transform_null(prod_env_and_data):
    filter = ConditionFilter("a1", "NOTNULL", "", None, 0.1)
    _, data = prod_env_and_data
    df: VDataFrame = data['vdf']
    result = filter.fit_transform(df)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 3)
