from functools import partial

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import FunctionTransformer as SkFunctionTransformer

from secretflow import reveal
from secretflow.data import partition
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.mix.dataframe import MixDataFrame
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.preprocessing import LogroundTransformer
from secretflow.preprocessing.transformer import _FunctionTransformer
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
            sf_production_setup_devices.alice: partition(
                data=sf_production_setup_devices.alice(
                    lambda: pl.from_pandas(vdf_alice)
                )(),
                backend="polars",
            ),
            sf_production_setup_devices.bob: partition(
                data=sf_production_setup_devices.bob(lambda: pl.from_pandas(vdf_bob))(),
                backend="polars",
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


def test_on_vdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    transformer = _FunctionTransformer(partial(np.add, 1))

    # WHEN
    value = transformer.fit_transform(data['vdf'][['a3', 'b4', 'b6']])
    params = transformer.get_params()

    # THEN
    assert params
    sk_transformer = SkFunctionTransformer(partial(np.add, 1))
    expect_alice = sk_transformer.fit_transform(data['vdf_alice'][['a3']])
    pd.testing.assert_frame_equal(
        reveal(value.to_pandas().partitions[env.alice].data), expect_alice
    )

    expect_bob = sk_transformer.fit_transform(data['vdf_bob'][['b4', 'b6']])
    pd.testing.assert_frame_equal(
        reveal(value.to_pandas().partitions[env.bob].data), expect_bob
    )


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
            env.alice: partition(data=env.alice(lambda: df_part0.iloc[:4, :])()),
            env.bob: partition(data=env.bob(lambda: df_part1.iloc[:4, :])()),
        }
    )
    h_part1 = VDataFrame(
        {
            env.alice: partition(data=env.alice(lambda: df_part0.iloc[4:, :])()),
            env.bob: partition(data=env.bob(lambda: df_part1.iloc[4:, :])()),
        }
    )
    h_mix = MixDataFrame(partitions=[h_part0, h_part1])

    transformer = _FunctionTransformer(partial(np.add, 1))

    # WHEN
    value = transformer.fit_transform(h_mix[['a3', 'b4', 'b6']])
    params = transformer.get_params()

    # THEN
    assert params
    sk_transformer = SkFunctionTransformer(partial(np.add, 1))
    expect_alice = sk_transformer.fit_transform(df_part0[['a3']])
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.alice].data),
                reveal(value.partitions[1].partitions[env.alice].data),
            ]
        ),
        expect_alice,
    )
    expect_bob = sk_transformer.fit_transform(df_part1[['b4', 'b6']])
    pd.testing.assert_frame_equal(
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
            env.alice: partition(data=env.alice(lambda: df_part0.iloc[:4, :])()),
            env.bob: partition(data=env.bob(lambda: df_part0.iloc[4:, :])()),
        },
        aggregator=PlainAggregator(env.carol),
        comparator=PlainComparator(env.carol),
    )
    v_part1 = HDataFrame(
        {
            env.alice: partition(data=env.alice(lambda: df_part1.iloc[:4, :])()),
            env.bob: partition(data=env.bob(lambda: df_part1.iloc[4:, :])()),
        },
        aggregator=PlainAggregator(env.carol),
        comparator=PlainComparator(env.carol),
    )
    v_mix = MixDataFrame(partitions=[v_part0, v_part1])

    transformer = _FunctionTransformer(partial(np.add, 1))

    # WHEN
    value = transformer.fit_transform(v_mix[['a3', 'b4', 'b6']])
    params = transformer.get_params()

    # THEN
    assert params
    sk_transformer = SkFunctionTransformer(partial(np.add, 1))
    expect_alice = sk_transformer.fit_transform(df_part0[['a3']])
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.alice].data),
                reveal(value.partitions[0].partitions[env.bob].data),
            ]
        ),
        expect_alice,
    )
    expect_bob = sk_transformer.fit_transform(df_part1[['b4', 'b6']])
    pd.testing.assert_frame_equal(
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
    transformer = _FunctionTransformer(partial(np.add, 1))
    with pytest.raises(
        AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
    ):
        transformer.fit(['test'])
    transformer.fit(data['vdf']['a3'])
    with pytest.raises(
        AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
    ):
        transformer.transform('test')


def test_transform_should_ok_when_not_fit(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    selected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    transformer = _FunctionTransformer(partial(np.add, 1))

    # WHEN
    value = transformer.transform(data['hdf'][selected_cols])

    # THEN
    assert value


def test_loground_on_vdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    transformer = LogroundTransformer(decimals=2, bias=1)

    # WHEN
    value = transformer.fit_transform(data['vdf'][['a3', 'b4', 'b6']])
    params = transformer.get_params()

    # THEN
    assert params

    def loground(x: pd.DataFrame):
        return x.add(1).apply(np.log2).round(2)

    sk_transformer = SkFunctionTransformer(loground)
    expect_alice = sk_transformer.fit_transform(data['vdf_alice'][['a3']])
    pd.testing.assert_frame_equal(
        reveal(value.to_pandas().partitions[env.alice].data), expect_alice
    )

    expect_bob = sk_transformer.fit_transform(data['vdf_bob'][['b4', 'b6']])
    pd.testing.assert_frame_equal(
        reveal(value.to_pandas().partitions[env.bob].data), expect_bob
    )


def test_loground_on_hdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    selected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    transformer = LogroundTransformer(decimals=2, bias=1)

    # WHEN
    value = transformer.fit_transform(data['hdf'][selected_cols])
    params = transformer.get_params()

    # THEN
    assert params

    def loground(x: pd.DataFrame):
        return x.add(1).apply(np.log2).round(2)

    sk_transformer = SkFunctionTransformer(loground)
    sk_transformer.fit(
        pd.concat([data['hdf_alice'][selected_cols], data['hdf_bob'][selected_cols]])
    )
    expect_alice = sk_transformer.transform(data['hdf_alice'][selected_cols])
    pd.testing.assert_frame_equal(
        reveal(value.partitions[env.alice].data),
        expect_alice,
    )
    expect_bob = sk_transformer.transform(data['hdf_bob'][selected_cols])
    pd.testing.assert_frame_equal(reveal(value.partitions[env.bob].data), expect_bob)
