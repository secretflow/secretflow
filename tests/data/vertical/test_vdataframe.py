import numpy as np
import pandas as pd
import pytest

from secretflow import reveal
from secretflow.data.base import partition
from secretflow.data.vertical import VDataFrame
from secretflow.utils.errors import NotFoundError


@pytest.fixture(scope='function')
def prod_env_and_data(sf_production_setup_devices):
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
            sf_production_setup_devices.alice: partition(
                data=sf_production_setup_devices.alice(lambda: df_alice)()
            ),
            sf_production_setup_devices.bob: partition(
                data=sf_production_setup_devices.bob(lambda: df_bob)()
            ),
        }
    )

    yield sf_production_setup_devices, {
        "df_alice": df_alice,
        "df_bob": df_bob,
        "df": df,
    }


def test_columns_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    columns = data['df'].columns
    # THEN
    alice_columns = data['df_alice'].columns.to_list()
    bob_columns = data['df_bob'].columns.to_list()
    alice_columns.extend(bob_columns)
    np.testing.assert_equal(columns, alice_columns)


def test_pow_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].select_dtypes('number').pow(2).mean()

    # THEN
    expected_alice = data['df_alice'].select_dtypes('number').pow(2).mean()
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].select_dtypes('number').pow(2).mean()
    pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)


def test_select_dtypes_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].select_dtypes('number').mean()

    # THEN
    expected_alice = data['df_alice'].select_dtypes('number').mean()
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].select_dtypes('number').mean()
    pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)


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


def test_min_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].min(numeric_only=True)

    # THEN
    expected_alice = data['df_alice'].min(numeric_only=True)
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].min(numeric_only=True)
    pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)


def test_max_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].max(numeric_only=True)

    # THEN
    expected_alice = data['df_alice'].max(numeric_only=True)
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].max(numeric_only=True)
    pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)


def test_mean_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].mean(numeric_only=True)

    # THEN
    expected_alice = data['df_alice'].mean(numeric_only=True)
    pd.testing.assert_series_equal(value[expected_alice.index], expected_alice)
    expected_bob = data['df_bob'].mean(numeric_only=True)
    pd.testing.assert_series_equal(value[expected_bob.index], expected_bob)


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


def test_quantle_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].quantile()
    # THEN
    expected_alice = data['df_alice'].quantile()
    assert value['a3'] == expected_alice['a3']
    expected_bob = data['df_bob'].quantile()
    # TODO(zoupeicheng.zpc): Ray's pandas ignored series containing None
    pd.testing.assert_series_equal(value[['b6']], expected_bob[['b6']])


def test_count_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].count()

    # THEN
    expected_alice = pd.concat([data['df_alice'], data['df_bob']], axis=1).count()
    pd.testing.assert_series_equal(value, expected_alice)


def test_mode_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].mode()

    # THEN
    expected_alice = (
        pd.concat([data['df_alice'], data['df_bob']], axis=1).mode().iloc[0, :]
    )
    pd.testing.assert_series_equal(value, expected_alice)


def test_count_na_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df'].isna().sum()

    # THEN
    expected_alice = pd.concat([data['df_alice'], data['df_bob']], axis=1).isna().sum()
    pd.testing.assert_series_equal(value, expected_alice)


def test_get_single_item_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value = data['df']['a1']

    # THEN
    expected_alice = data['df_alice'][['a1']]
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data), expected_alice
    )


def test_get_non_exist_items_should_error(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN and THEN
    with pytest.raises(NotFoundError, match='does not exist'):
        data['df']['a1', 'non_exist']


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


def test_astype_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    df = data['df'][['a3', 'b4']].fillna(value=1, inplace=False)

    # Case 1: single dtype
    # WHEN
    new_df = df.astype(np.int32)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(new_df.partitions[env.alice].data),
        data['df_alice'][['a3']].fillna(1).astype(np.int32),
    )
    pd.testing.assert_frame_equal(
        reveal(new_df.partitions[env.bob].data),
        data['df_bob'][['b4']].fillna(1).astype(np.int32),
    )

    # Case 2: dtype dict.
    # WHEN
    dtype = {'a3': np.int32}
    new_df = df.astype(dtype)
    # THEN
    pd.testing.assert_frame_equal(
        reveal(new_df.partitions[env.alice].data),
        data['df_alice'][['a3']].fillna(1).astype(dtype),
    )
    pd.testing.assert_frame_equal(
        reveal(new_df.partitions[env.bob].data),
        data['df_bob'][['b4']].fillna(1),
    )
