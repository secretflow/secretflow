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
from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob

    df_alice = pd.DataFrame(
        {
            'a1': ['K5', 'K1', None, 'K6'],
            'a2': ['A5', 'A1', 'A2', 'A6'],
            'a3': [5, 1, 2, 6],
        }
    )

    df_bob = pd.DataFrame(
        {
            'b4': [10.2, 20.5, None, -0.4],
            'b5': ['B3', None, 'B9', 'B4'],
            'b6': [3, 1, 9, 4],
        }
    )

    df = VDataFrame(
        {
            pyu_alice: partition(data=pyu_alice(lambda: df_alice)()),
            pyu_bob: partition(data=pyu_bob(lambda: df_bob)()),
        }
    )

    df_cleartext = pd.DataFrame(
        {
            'a1': ['K5', 'K1', None, 'K6'],
            'a2': ['A5', 'A1', 'A2', 'A6'],
            'a3': [5, 1, 2, 6],
            'b4': [10.2, 20.5, None, -0.4],
            'b5': ['B3', None, 'B9', 'B4'],
            'b6': [3, 1, 9, 4],
        }
    )

    yield sf_production_setup_devices, {
        "df_alice": df_alice,
        "df_bob": df_bob,
        "df": df,
        "df_cleartext": df_cleartext,
    }


@pytest.mark.mpc
def test_columns_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    columns = data['df'].columns
    # THEN
    alice_columns = data['df_alice'].columns.to_list()
    bob_columns = data['df_bob'].columns.to_list()
    alice_columns.extend(bob_columns)
    np.testing.assert_equal(columns, alice_columns)


@pytest.mark.mpc
def test_pow_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].select_dtypes('number').pow(2).mean()

    # THEN
    expected_alice = data['df_alice'].select_dtypes('number').pow(2).mean()
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].select_dtypes('number').pow(2).mean()
    pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)


@pytest.mark.mpc
def test_select_dtypes_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].select_dtypes('number').mean()

    # THEN
    expected_alice = data['df_alice'].select_dtypes('number').mean()
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].select_dtypes('number').mean()
    pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)


@pytest.mark.mpc
def test_subtract_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = (
        data['df'].subtract(data['df'].mean(numeric_only=True)).mean(numeric_only=True)
    )

    # THEN
    expected_alice = (
        data['df_alice']
        .subtract(data['df_alice'].mean(numeric_only=True))
        .mean(numeric_only=True)
    )
    assert value['a3'] == expected_alice['a3']
    expected_bob = (
        data['df_bob']
        .subtract(data['df_bob'].mean(numeric_only=True))
        .mean(numeric_only=True)
    )
    # TODO(zoupeicheng.zpc): Ray's pandas ignored series containing None
    pd.testing.assert_series_equal(value[['b6']], expected_bob[['b6']])


@pytest.mark.mpc
def test_round_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].round({'b4': 0})

    # THEN
    value_alice = reveal(value.partitions[env.alice].data)
    expected_alice = data['df_alice']
    pd.testing.assert_frame_equal(value_alice, expected_alice)
    expected_bob = data['df_bob'].round({'b4': 0})
    value_bob = reveal(value.partitions[env.bob].data)
    pd.testing.assert_frame_equal(value_bob, expected_bob)


@pytest.mark.mpc
def test_min_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].min(numeric_only=True)

    # THEN
    expected_alice = data['df_alice'].min(numeric_only=True)
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].min(numeric_only=True)
    pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)


@pytest.mark.mpc
def test_max_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].max(numeric_only=True)

    # THEN
    expected_alice = data['df_alice'].max(numeric_only=True)
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].max(numeric_only=True)
    pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)


@pytest.mark.mpc
def test_mean_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].mean(numeric_only=True)

    # THEN
    expected_alice = data['df_alice'].mean(numeric_only=True)
    pd.testing.assert_series_equal(value[expected_alice.index], expected_alice)
    expected_bob = data['df_bob'].mean(numeric_only=True)
    pd.testing.assert_series_equal(value[expected_bob.index], expected_bob)


@pytest.mark.mpc
def test_var_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].var(numeric_only=True)
    # THEN
    expected_alice = data['df_alice'].var(numeric_only=True)
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].var(numeric_only=True)
    # TODO(zoupeicheng.zpc): Ray's pandas ignored series containing None
    pd.testing.assert_series_equal(value[['b6']], expected_bob[['b6']])


@pytest.mark.mpc
def test_std_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].std(numeric_only=True)
    # THEN
    expected_alice = data['df_alice'].std(numeric_only=True)
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].std(numeric_only=True)
    # TODO(zoupeicheng.zpc): Ray's pandas ignored series containing None
    pd.testing.assert_series_equal(value[['b6']], expected_bob[['b6']])


@pytest.mark.mpc
def test_sem_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].sem(numeric_only=True)
    # THEN
    expected_alice = data['df_alice'].sem(numeric_only=True)
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].sem(numeric_only=True)
    # TODO(zoupeicheng.zpc): Ray's pandas ignored series containing None
    pd.testing.assert_series_equal(value[['b6']], expected_bob[['b6']])


@pytest.mark.mpc
def test_skew_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].skew(numeric_only=True)
    # THEN
    expected_alice = data['df_alice'].skew(numeric_only=True)
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].skew(numeric_only=True)
    # TODO(zoupeicheng.zpc): Ray's pandas ignored series containing None
    pd.testing.assert_series_equal(value[['b6']], expected_bob[['b6']])


@pytest.mark.mpc
def test_quantile_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].quantile()
    # THEN
    expected_alice = data['df_alice'].quantile()
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].quantile()
    # TODO(zoupeicheng.zpc): Ray's pandas ignored series containing None
    pd.testing.assert_series_equal(value[['b6']], expected_bob[['b6']])


@pytest.mark.mpc
def test_count_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].count()

    # THEN
    expected_alice = pd.concat([data['df_alice'], data['df_bob']], axis=1).count()
    pd.testing.assert_series_equal(value, expected_alice)


@pytest.mark.mpc
def test_mode_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].mode()

    # THEN
    expected_alice = (
        pd.concat([data['df_alice'], data['df_bob']], axis=1).mode().iloc[0, :]
    )
    pd.testing.assert_series_equal(value, expected_alice)


@pytest.mark.mpc
def test_count_na_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].isna().sum()

    # THEN
    expected_alice = pd.concat([data['df_alice'], data['df_bob']], axis=1).isna().sum()
    pd.testing.assert_series_equal(value, expected_alice)


@pytest.mark.mpc
def test_get_single_item_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df']['a1']

    # THEN
    expected_alice = data['df_alice'][['a1']]
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )


@pytest.mark.mpc
def test_get_non_exist_items_should_error(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN and THEN
    with pytest.raises(KeyError, match='does not exist'):
        _ = data['df']['a1', 'non_exist']


@pytest.mark.mpc
def test_get_multi_items_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'][['a1', 'b4']]
    # THEN
    expected_alice = data['df_alice'][['a1']]
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )
    expected_bob = data['df_bob'][['b4']]
    pd.testing.assert_frame_equal(reveal(value.partitions[env.bob].data), expected_bob)

    # WHEN
    value = data['df'][['a1', 'a2', 'b5']]
    # THEN
    expected_alice = data['df_alice'][['a1', 'a2']]
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )
    expected_bob = data['df_bob'][['b5']]
    pd.testing.assert_frame_equal(reveal(value.partitions[env.bob].data), expected_bob)


@pytest.mark.mpc
def test_set_item_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # Case 1: single item.
    # WHEN
    value = data['df']
    value['a1'] = 'test'
    # THEN
    expected_alice = data['df_alice'].copy(deep=True)
    expected_alice['a1'] = 'test'
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )

    # Case 2: multi items on different parties.
    # WHEN
    value = data['df']
    value[['a1', 'b4', 'b5']] = 'test'
    # THEN
    expected_alice = data['df_alice'].copy(deep=True)
    expected_alice['a1'] = 'test'
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )
    expected_bob = data['df_bob'].copy(deep=True)
    expected_bob[['b4', 'b5']] = 'test'
    pd.testing.assert_frame_equal(reveal(value.partitions[env.bob].data), expected_bob)


@pytest.mark.mpc
def test_set_item_on_partition_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df']
    value['a1'] = value['a2']
    # THEN
    expected_alice = data['df_alice'].copy(deep=True)
    expected_alice['a1'] = expected_alice['a2']
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )


@pytest.mark.mpc(parties=3)
def test_set_item_on_non_exist_partition_should_error(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    with pytest.raises(
        AssertionError,
        match='Device of the partition to assgin is not in this dataframe devices.',
    ):
        part = partition(
            env.carol(
                lambda: pd.DataFrame(
                    {
                        'a1': ['K5', 'K1', None, 'K6'],
                        'a2': ['A5', 'A1', 'A2', 'A6'],
                        'a3': [5, 1, 2, 6],
                    }
                )
            )()
        )
        value = data['df']
        value['a1'] = part['a2']


@pytest.mark.mpc
def test_set_item_on_vdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df']
    value[['a1', 'b4']] = data['df'][['a2', 'b5']]

    # THEN
    expected_alice = data['df_alice'].copy(deep=True)
    expected_alice['a1'] = expected_alice['a2']
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )

    expected_bob = data['df_bob'].copy(deep=True)
    expected_bob['b4'] = expected_bob['b5']
    pd.testing.assert_frame_equal(reveal(value.partitions[env.bob].data), expected_bob)


@pytest.mark.mpc(parties=3)
def test_set_item_on_different_vdataframe_should_error(prod_env_and_data):
    env, data = prod_env_and_data
    with pytest.raises(
        AssertionError,
        match='Partitions to assgin is not same with this dataframe partitions.',
    ):
        df = VDataFrame(
            {
                env.alice: partition(
                    data=env.alice(
                        lambda: pd.DataFrame(
                            {
                                'a1': ['K5', 'K1', None, 'K6'],
                                'a2': ['A5', 'A1', 'A2', 'A6'],
                                'a3': [5, 1, 2, 6],
                            }
                        )
                    )()
                ),
                env.carol: partition(
                    data=env.carol(
                        lambda: pd.DataFrame(
                            {
                                'b4': [10.2, 20.5, None, -0.4],
                                'b5': ['B3', None, 'B9', 'B4'],
                                'b6': [3, 1, 9, 4],
                            }
                        )
                    )()
                ),
            }
        )
        value = data['df']
        value[['a1', 'b4']] = df[['a2', 'b5']]


@pytest.mark.mpc
def test_drop(prod_env_and_data):
    env, data = prod_env_and_data
    # Case 1: not inplace.
    # WHEN
    value = data['df'].drop(columns='a1', inplace=False)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data),
        data['df_alice'].drop(columns='a1', inplace=False),
    )
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.bob].data), data['df_bob']
    )

    # Case 2: inplace.
    # WHEN
    value = data['df'].copy()
    value.drop(columns=['a1', 'b4', 'b5'], inplace=True)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data),
        data['df_alice'].drop(columns='a1', inplace=False),
    )
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.bob].data),
        data['df_bob'].drop(columns=['b4', 'b5'], inplace=False),
    )


@pytest.mark.mpc
def test_replace_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    val = data['df_alice'].iloc[1, 1]
    val_to = 0.131212
    # WHEN
    value = data['df'].replace(val, val_to)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data),
        data['df_alice'].replace(val, val_to),
    )
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.bob].data),
        data['df_bob'].replace(val, val_to),
    )


@pytest.mark.mpc
def test_fillna(prod_env_and_data):
    env, data = prod_env_and_data
    # Case 1: not inplace.
    # WHEN
    value = data['df'].fillna(value='test', inplace=False)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data),
        data['df_alice'].fillna(value='test', inplace=False),
    )
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.bob].data),
        data['df_bob'].fillna(value='test', inplace=False),
    )

    # Case 2: inplace.
    # WHEN
    value = data['df'].copy()
    value.fillna(value='test', inplace=True)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data),
        data['df_alice'].fillna(value='test', inplace=False),
    )
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.bob].data),
        data['df_bob'].fillna(value='test', inplace=False),
    )


@pytest.mark.parametrize("agg_name", ['sum', 'count', 'min', 'max', 'mean', "var"])
@pytest.mark.mpc
def test_groupby_agg(prod_env_and_data, agg_name):
    env, data = prod_env_and_data
    # GIVEN
    df = data['df'][['a1', 'a2', 'a3', 'b4', 'b5', 'b6']].fillna(value=0, inplace=False)
    df[["a3", "b4", "b6"]] = (
        df[["a3", "b4", "b6"]].fillna(value=0, inplace=False).astype(float)
    )
    df_cleartext = data['df_cleartext'].fillna(value=0, inplace=False)

    our_values = getattr(df.groupby(env.spu, ['a3'])['b6', 'b4'], agg_name)()
    true_values = getattr(df_cleartext.groupby(['a3'])['b6', 'b4'], agg_name)().fillna(
        value=0, inplace=False
    )
    if agg_name in ["mean", "var"]:
        decimal = 3
    else:
        decimal = 6
    np.testing.assert_array_almost_equal(our_values, true_values, decimal=decimal)

    our_values = getattr(df.groupby(env.spu, ['a3'])['b6'], agg_name)()
    true_values = getattr(df_cleartext.groupby(['a3'])['b6'], agg_name)().fillna(
        value=0, inplace=False
    )
    np.testing.assert_array_almost_equal(our_values, true_values, decimal=decimal)
