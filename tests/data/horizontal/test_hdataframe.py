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

from secretflow import reveal
from secretflow.security.aggregation import PlainAggregator, SPUAggregator
from secretflow.security.compare import PlainComparator, SPUComparator
from secretflow.utils.simulation.datasets import load_iris
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob
    spu = sf_production_setup_devices.spu

    df_plain = load_iris(
        parts=[
            pyu_alice,
            pyu_bob,
        ],
        aggregator=PlainAggregator(pyu_alice),
        comparator=PlainComparator(pyu_alice),
    )
    df_spu = load_iris(
        parts=[
            pyu_alice,
            pyu_bob,
        ],
        aggregator=SPUAggregator(spu),
        comparator=SPUComparator(spu),
    )
    df_alice = reveal(df_plain.partitions[pyu_alice].data)
    df_bob = reveal(df_plain.partitions[pyu_bob].data)

    yield sf_production_setup_devices, {
        "df_plain": df_plain,
        "df_spu": df_spu,
        "df_alice": df_alice,
        "df_bob": df_bob,
    }


@pytest.mark.mpc
def test_mean_with_plain_aggr_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    mean = data['df_plain'].mean(numeric_only=True)

    # THEN
    expected = np.average(
        [
            data['df_alice'].mean(numeric_only=True),
            data['df_bob'].mean(numeric_only=True),
        ],
        weights=[
            data['df_alice'].count(numeric_only=True),
            data['df_bob'].count(numeric_only=True),
        ],
        axis=0,
    )
    pd.testing.assert_series_equal(
        mean,
        pd.Series(
            expected,
            index=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        ),
    )


@pytest.mark.mpc
def test_mean_with_spu_aggr_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    mean = data['df_spu'].mean(numeric_only=True)

    # THEN
    expected = np.average(
        [
            data['df_alice'].mean(numeric_only=True),
            data['df_bob'].mean(numeric_only=True),
        ],
        weights=[
            data['df_alice'].count(numeric_only=True),
            data['df_bob'].count(numeric_only=True),
        ],
        axis=0,
    )
    pd.testing.assert_series_equal(
        mean,
        pd.Series(
            expected,
            index=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        ),
    )


@pytest.mark.mpc
def test_min_with_plain_comp_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    min = data['df_plain'].min(numeric_only=True)

    # THEN
    expected = np.minimum(
        data['df_alice'].min(numeric_only=True), data['df_bob'].min(numeric_only=True)
    )
    pd.testing.assert_series_equal(min, expected)


@pytest.mark.mpc
def test_min_with_spu_comp_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    min = data['df_spu'].min(numeric_only=True)

    # THEN
    expected = np.minimum(
        data['df_alice'].min(numeric_only=True), data['df_bob'].min(numeric_only=True)
    )
    pd.testing.assert_series_equal(min, expected)


@pytest.mark.mpc
def test_max_with_plain_comp_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    max = data['df_plain'].max(numeric_only=True)

    # THEN
    expected = np.maximum(
        data['df_alice'].max(numeric_only=True), data['df_bob'].max(numeric_only=True)
    )
    pd.testing.assert_series_equal(max, expected)


@pytest.mark.mpc
def test_max_with_spu_comp_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    max = data['df_spu'].max(numeric_only=True)

    # THEN
    expected = np.maximum(
        data['df_alice'].max(numeric_only=True), data['df_bob'].max(numeric_only=True)
    )
    pd.testing.assert_series_equal(max, expected)


@pytest.mark.mpc
def test_count_with_plain_aggr_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    count = data['df_plain'].count()

    # THEN
    expected = data['df_alice'].count() + data['df_bob'].count()
    pd.testing.assert_series_equal(count, expected)


@pytest.mark.mpc
def test_count_with_spu_aggr_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    count = data['df_spu'].count()

    # THEN
    expected = data['df_alice'].count() + data['df_bob'].count()
    pd.testing.assert_series_equal(count, expected)


@pytest.mark.mpc
def test_count_na_with_plain_aggr_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    # Note currently, our device execution may result in different types
    # compared to original pandas, like int32 not int64
    count = data['df_plain'].isna().sum().astype(np.int64)

    # THEN
    expected = data['df_alice'].isna().sum() + data['df_bob'].isna().sum()
    pd.testing.assert_series_equal(count, expected)


@pytest.mark.mpc
def test_count_na_with_spu_aggr_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    # Note currently, our device execution may result in different types
    # compared to original pandas, like int32 not int64
    count = data['df_spu'].isna().sum().astype(np.int64)

    # THEN
    expected = data['df_alice'].isna().sum() + data['df_bob'].isna().sum()
    pd.testing.assert_series_equal(count, expected)


@pytest.mark.mpc
def test_len_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    length = len(data['df_plain'])

    # THEN
    expected = len(data['df_alice']) + len(data['df_bob'])
    assert length == expected


@pytest.mark.mpc
def test_getitem_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # Case 1: single item.
    # WHEN
    value = data['df_plain']['sepal_length']
    # THEN
    expected_alice = data['df_alice'][['sepal_length']]
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )
    expected_bob = data['df_bob'][['sepal_length']]
    pd.testing.assert_frame_equal(reveal(value.partitions[env.bob].data), expected_bob)

    # Case 2: multi items.
    # WHEN
    value = data['df_plain'][['sepal_length', 'sepal_width']]
    # THEN
    expected_alice = data['df_alice'][['sepal_length', 'sepal_width']]
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )
    expected_bob = data['df_bob'][['sepal_length', 'sepal_width']]
    pd.testing.assert_frame_equal(reveal(value.partitions[env.bob].data), expected_bob)


@pytest.mark.mpc
def test_setitem_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    hdf = data['df_plain'].copy()

    # Case 1: single item.
    # WHEN
    hdf['sepal_length'] = 'test'
    # THEN
    expected_alice = data['df_alice']
    expected_alice['sepal_length'] = 'test'
    pd.testing.assert_frame_equal(
        reveal(hdf.partitions[env.alice].data), expected_alice
    )
    expected_bob = data['df_bob']
    expected_bob['sepal_length'] = 'test'
    pd.testing.assert_frame_equal(reveal(hdf.partitions[env.bob].data), expected_bob)

    # Case 2: multi items.
    # WHEN
    hdf[['sepal_length', 'sepal_width']] = data['df_alice'][
        ['sepal_length', 'sepal_width']
    ]
    # THEN
    expected_alice = data['df_alice']
    pd.testing.assert_frame_equal(
        reveal(hdf.partitions[env.alice].data), expected_alice
    )
    expected_bob = data['df_bob']
    expected_bob[['sepal_length', 'sepal_width']] = data['df_alice'][
        ['sepal_length', 'sepal_width']
    ]
    pd.testing.assert_frame_equal(reveal(hdf.partitions[env.bob].data), expected_bob)


@pytest.mark.mpc
def test_drop(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    hdf = data['df_plain'].copy()

    # Case 1: not inplace.
    # WHEN
    new_hdf = hdf.drop(columns='sepal_length', inplace=False)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(new_hdf.partitions[env.alice].data),
        data['df_alice'].drop(columns='sepal_length', inplace=False),
    )
    pd.testing.assert_frame_equal(
        reveal(new_hdf.partitions[env.bob].data),
        data['df_bob'].drop(columns='sepal_length', inplace=False),
    )

    # Case 2: inplace.
    # WHEN
    hdf.drop(columns='sepal_length', inplace=True)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(hdf.partitions[env.alice].data),
        data['df_alice'].drop(columns='sepal_length', inplace=False),
    )
    pd.testing.assert_frame_equal(
        reveal(hdf.partitions[env.bob].data),
        data['df_bob'].drop(columns='sepal_length', inplace=False),
    )


@pytest.mark.mpc
def test_fillna(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    hdf = data['df_plain'].copy()

    # Case 1: not inplace.
    # WHEN
    new_hdf = hdf.fillna(value='test', inplace=False)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(new_hdf.partitions[env.alice].data),
        data['df_alice'].fillna(value='test', inplace=False),
    )
    pd.testing.assert_frame_equal(
        reveal(new_hdf.partitions[env.bob].data),
        data['df_bob'].fillna(value='test', inplace=False),
    )

    # Case 2: inplace.
    # WHEN
    hdf.fillna(value='test', inplace=True)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(hdf.partitions[env.alice].data),
        data['df_alice'].fillna(value='test', inplace=False),
    )
    pd.testing.assert_frame_equal(
        reveal(hdf.partitions[env.bob].data),
        data['df_bob'].fillna(value='test', inplace=False),
    )


@pytest.mark.mpc
def test_astype_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    hdf = data['df_plain'][
        ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    ]
    hdf.fillna(value=1, inplace=True)

    # Case 1: single dtype
    # WHEN
    new_hdf = hdf.astype(np.int32)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(new_hdf.partitions[env.alice].data),
        data['df_alice'].iloc[:, 0:4].fillna(1).astype(np.int32),
    )
    pd.testing.assert_frame_equal(
        reveal(new_hdf.partitions[env.bob].data),
        data['df_bob'].iloc[:, 0:4].fillna(1).astype(np.int32),
    )

    # Case 2: dtype dict.
    # WHEN
    dtype = {'sepal_length': np.int32, 'sepal_width': np.int32}
    new_hdf = hdf.astype(dtype)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(new_hdf.partitions[env.alice].data),
        data['df_alice'].iloc[:, 0:4].fillna(1).astype(dtype),
    )
    pd.testing.assert_frame_equal(
        reveal(new_hdf.partitions[env.bob].data),
        data['df_bob'].iloc[:, 0:4].fillna(1).astype(dtype),
    )
