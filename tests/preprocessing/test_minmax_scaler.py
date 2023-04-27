import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.mix.dataframe import MixDataFrame
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow.utils.simulation.datasets import load_iris


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
    hdf = load_iris(
        parts=[sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        aggregator=PlainAggregator(sf_production_setup_devices.alice),
        comparator=PlainComparator(sf_production_setup_devices.carol),
    )
    hdf_alice = reveal(hdf.partitions[sf_production_setup_devices.alice].data)
    hdf_bob = reveal(hdf.partitions[sf_production_setup_devices.bob].data)

    vdf_alice = pd.DataFrame(
        {
            'a1': ['K5', 'K1', None, 'K6'],
            'a2': ['A5', 'A1', 'A2', 'A6'],
            'a3': [5, 1, 2, 6],
        }
    )

    vdf_bob = pd.DataFrame(
        {
            'b4': [10.2, 20.5, None, -0.4],
            'b5': ['B3', None, 'B9', 'B4'],
            'b6': [3, 1, 9, 4],
        }
    )

    vdf = VDataFrame(
        {
            sf_production_setup_devices.alice: Partition(
                data=sf_production_setup_devices.alice(lambda: vdf_alice)()
            ),
            sf_production_setup_devices.bob: Partition(
                data=sf_production_setup_devices.bob(lambda: vdf_bob)()
            ),
        }
    )

    yield sf_production_setup_devices, {
        'hdf': hdf,
        'hdf_alice': hdf_alice,
        'hdf_bob': hdf_bob,
        'vdf_alice': vdf_alice,
        'vdf_bob': vdf_bob,
        'vdf': vdf,
    }


def test_on_hdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    selected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    scaler = MinMaxScaler()

    # WHEN
    value = scaler.fit_transform(data['hdf'][selected_cols])
    params = scaler.get_params()

    # THEN
    assert params
    sk_scaler = SkMinMaxScaler()
    sk_scaler.fit(
        pd.concat([data['hdf_alice'][selected_cols], data['hdf_bob'][selected_cols]])
    )
    expect_alice = sk_scaler.transform(data['hdf_alice'][selected_cols])
    np.testing.assert_almost_equal(
        reveal(value.partitions[env.alice].data),
        expect_alice,
    )
    expect_bob = sk_scaler.transform(data['hdf_bob'][selected_cols])
    np.testing.assert_almost_equal(reveal(value.partitions[env.bob].data), expect_bob)


def test_on_vdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    scaler = MinMaxScaler()

    # WHEN
    value = scaler.fit_transform(data['vdf'][['a3', 'b4', 'b6']])
    params = scaler.get_params()

    # THEN
    assert params
    sk_scaler = SkMinMaxScaler()
    expect_alice = sk_scaler.fit_transform(data['vdf_alice'][['a3']])
    np.testing.assert_equal(reveal(value.partitions[env.alice].data), expect_alice)

    expect_bob = sk_scaler.fit_transform(data['vdf_bob'][['b4', 'b6']])
    np.testing.assert_equal(reveal(value.partitions[env.bob].data), expect_bob)


def test_on_h_mixdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    df_part0 = pd.DataFrame(
        {
            'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4'],
            'a2': ['A2', 'B2', 'C2', 'D2', 'A5', 'B5', 'C5', 'D5'],
            'a3': [5, 1, 2, 6, 15, None, 23, 6],
        }
    )

    df_part1 = pd.DataFrame(
        {
            'b4': [10.2, 20.5, None, -0.4, None, 0.5, None, -10.4],
            'b5': ['B3', None, 'B9', 'B4', 'A3', None, 'C9', 'E4'],
            'b6': [3, 1, 9, 4, 31, 12, 9, 21],
        }
    )
    h_part0 = VDataFrame(
        {
            env.alice: Partition(data=env.alice(lambda: df_part0.iloc[:4, :])()),
            env.bob: Partition(data=env.bob(lambda: df_part1.iloc[:4, :])()),
        }
    )
    h_part1 = VDataFrame(
        {
            env.alice: Partition(data=env.alice(lambda: df_part0.iloc[4:, :])()),
            env.bob: Partition(data=env.bob(lambda: df_part1.iloc[4:, :])()),
        }
    )
    h_mix = MixDataFrame(partitions=[h_part0, h_part1])

    scaler = MinMaxScaler()

    # WHEN
    value = scaler.fit_transform(h_mix[['a3', 'b4', 'b6']])
    params = scaler.get_params()

    # THEN
    assert params
    sk_scaler = SkMinMaxScaler()
    expect_alice = sk_scaler.fit_transform(df_part0[['a3']])
    np.testing.assert_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.alice].data),
                reveal(value.partitions[1].partitions[env.alice].data),
            ]
        ),
        expect_alice,
    )
    expect_bob = sk_scaler.fit_transform(df_part1[['b4', 'b6']])
    np.testing.assert_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.bob].data),
                reveal(value.partitions[1].partitions[env.bob].data),
            ]
        ),
        expect_bob,
    )


def test_on_v_mixdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    df_part0 = pd.DataFrame(
        {
            'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4'],
            'a2': ['A2', 'B2', 'C2', 'D2', 'A5', 'B5', 'C5', 'D5'],
            'a3': [5, 1, 2, 6, 15, None, 23, 6],
        }
    )

    df_part1 = pd.DataFrame(
        {
            'b4': [10.2, 20.5, None, -0.4, None, 0.5, None, -10.4],
            'b5': ['B3', None, 'B9', 'B4', 'A3', None, 'C9', 'E4'],
            'b6': [3, 1, 9, 4, 31, 12, 9, 21],
        }
    )
    v_part0 = HDataFrame(
        {
            env.alice: Partition(data=env.alice(lambda: df_part0.iloc[:4, :])()),
            env.bob: Partition(data=env.bob(lambda: df_part0.iloc[4:, :])()),
        },
        aggregator=PlainAggregator(env.carol),
        comparator=PlainComparator(env.carol),
    )
    v_part1 = HDataFrame(
        {
            env.alice: Partition(data=env.alice(lambda: df_part1.iloc[:4, :])()),
            env.bob: Partition(data=env.bob(lambda: df_part1.iloc[4:, :])()),
        },
        aggregator=PlainAggregator(env.carol),
        comparator=PlainComparator(env.carol),
    )
    v_mix = MixDataFrame(partitions=[v_part0, v_part1])

    scaler = MinMaxScaler()

    # WHEN
    value = scaler.fit_transform(v_mix[['a3', 'b4', 'b6']])
    params = scaler.get_params()

    # THEN
    assert params
    sk_scaler = SkMinMaxScaler()
    expect_alice = sk_scaler.fit_transform(df_part0[['a3']])
    np.testing.assert_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.alice].data),
                reveal(value.partitions[0].partitions[env.bob].data),
            ]
        ),
        expect_alice,
    )
    expect_bob = sk_scaler.fit_transform(df_part1[['b4', 'b6']])
    np.testing.assert_almost_equal(
        pd.concat(
            [
                reveal(value.partitions[1].partitions[env.alice].data),
                reveal(value.partitions[1].partitions[env.bob].data),
            ]
        ),
        expect_bob,
    )


def test_should_error_when_not_dataframe(prod_env_and_data):
    env, data = prod_env_and_data
    scaler = MinMaxScaler()
    with pytest.raises(
        AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
    ):
        scaler.fit(['test'])
    scaler.fit(data['vdf']['a3'])
    with pytest.raises(
        AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
    ):
        scaler.transform('test')


def test_transform_should_error_when_not_fit(prod_env_and_data):
    env, data = prod_env_and_data
    with pytest.raises(AssertionError, match='Scaler has not been fit yet.'):
        MinMaxScaler().transform('test')
