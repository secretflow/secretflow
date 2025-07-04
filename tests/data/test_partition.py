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

from secretflow import reveal
from secretflow.utils.simulation.datasets import load_iris
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice

    iris = load_iris(parts=[pyu_alice])
    part = iris.partitions[pyu_alice]
    df = reveal(part.data)

    yield sf_production_setup_devices, {"iris": iris, "part": part, "df": df}


@pytest.mark.mpc
def test_mean_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].mean(numeric_only=True)

    # THEN
    expected = data['df'].mean(numeric_only=True)
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_var_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].var(numeric_only=True)

    # THEN
    expected = data['df'].var(numeric_only=True)
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_std_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].std(numeric_only=True)

    # THEN
    expected = data['df'].std(numeric_only=True)
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_sem_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].sem(numeric_only=True)

    # THEN
    expected = data['df'].sem(numeric_only=True)
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_skew_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].skew(numeric_only=True)

    # THEN
    expected = data['df'].skew(numeric_only=True)
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_kurtosis_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].kurtosis(numeric_only=True)

    # THEN
    expected = data['df'].kurtosis(numeric_only=True)
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_quantile_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].quantile()

    # THEN
    expected = data['df'].quantile()
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_min_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].min()

    # THEN
    expected = data['df'].min()
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_max_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].max()

    # THEN
    expected = data['df'].max()
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_count_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].count()

    # THEN
    expected = data['df'].count()
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_pow_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].select_dtypes('number').pow(2.3).sum()

    # THEN
    expected = data['df'].select_dtypes('number').pow(2.3).sum()
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_select_dtypes_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].select_dtypes('number').mean()

    # THEN
    expected = data['df'].select_dtypes('number').mean()
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_subtract_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    part_num = data['part'].select_dtypes('number')
    means = part_num.mean()
    value = part_num.subtract(means)[part_num.columns].mean(numeric_only=True)

    # THEN
    df_num = data['df'].select_dtypes('number')
    df_means = df_num.mean()
    expected = df_num.subtract(df_means)[df_num.columns].mean(numeric_only=True)
    pd.testing.assert_series_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_round_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].round(1)

    # THEN
    expected = data['df'].round(1)
    pd.testing.assert_frame_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_dtypes_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].dtypes

    # THEN
    expected = data['df'].dtypes
    pd.testing.assert_series_equal(pd.Series(value), pd.Series(expected))


@pytest.mark.mpc
def test_index_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].index

    # THEN
    expected = data['df'].index
    pd.testing.assert_index_equal(value, expected)


@pytest.mark.mpc
def test_len_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = len(data['part'])

    # THEN
    expected = len(data['df'])
    assert value == expected


@pytest.mark.mpc
def test_iloc_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].iloc(0)
    # THEN
    expected = data['df'].iloc[0]
    pd.testing.assert_series_equal(reveal(value.data), expected)

    # WHEN
    value = data['part'].iloc([0, 1])
    # THEN
    expected = data['df'].iloc[[0, 1]]
    pd.testing.assert_frame_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_getitem_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part']['sepal_length']
    # THEN
    expected = data['df'][['sepal_length']]
    pd.testing.assert_frame_equal(reveal(value.data), expected)

    # WHEN
    value = data['part'][['sepal_length', 'sepal_width']]
    # THEN
    expected = data['df'][['sepal_length', 'sepal_width']]
    pd.testing.assert_frame_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_setitem_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].copy()
    value['sepal_length'] = 2

    # THEN
    expected = data['df'].copy(deep=True)
    expected['sepal_length'] = 2
    pd.testing.assert_frame_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_setitem_on_partition_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].copy()
    value['sepal_length'] = data['part']['sepal_width']

    # THEN
    expected = data['df'].copy(deep=True)
    expected['sepal_length'] = expected['sepal_width']
    pd.testing.assert_frame_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_setitem_on_different_partition_should_error(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN and THEN
    with pytest.raises(
        AssertionError, match='Can not assign a partition with different device.'
    ):
        part = load_iris(parts=[env.bob]).partitions[env.bob]
        value = data['part'].copy()
        value['sepal_length'] = part['sepal_width']


@pytest.mark.mpc
def test_drop_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # Case 1: not inplace.
    # WHEN
    value = data['part'].drop(columns=['sepal_length'], inplace=False)

    # THEN
    expected = data['df'].drop(columns=['sepal_length'], inplace=False)
    pd.testing.assert_frame_equal(reveal(value.data), expected)

    # Case 2: inplace.
    # WHEN
    value = data['part'].copy()
    value.drop(columns=['sepal_length'], inplace=True)

    # THEN
    expected = data['df'].copy(deep=True)
    expected.drop(columns=['sepal_length'], inplace=True)
    pd.testing.assert_frame_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_fillna_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # Case 1: not inplace.
    # WHEN
    value = data['part'].fillna(value='test', inplace=False)

    # THEN
    expected = data['df'].fillna(value='test', inplace=False)
    pd.testing.assert_frame_equal(reveal(value.data), expected)

    # Case 2: inplace.
    # WHEN
    value = data['part'].copy()
    value.fillna(value='test', inplace=True)

    # THEN
    expected = data['df'].copy(deep=True)
    expected.fillna(value='test', inplace=True)
    pd.testing.assert_frame_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_replace_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    val = data['df'].iloc[1, 1]
    val_to = 0.31312
    value = data['part'].replace(val, val_to)

    # THEN
    expected = data['df'].replace(val, val_to)
    pd.testing.assert_frame_equal(reveal(value.data), expected)


@pytest.mark.mpc
def test_mode_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['part'].mode()

    # THEN
    expected = data['df'].mode().iloc[0, :]
    pd.testing.assert_series_equal(reveal(value.data), expected)
