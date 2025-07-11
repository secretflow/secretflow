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
from sklearn.preprocessing import KBinsDiscretizer as SkKBinsDiscretizer

from secretflow import reveal
from secretflow.data import partition
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.mix.dataframe import MixDataFrame
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow.utils.simulation.datasets import load_iris
from secretflow_fl.preprocessing.discretization import KBinsDiscretizer
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob
    pyu_carol = sf_production_setup_devices.carol

    hdf = load_iris(
        parts=[
            pyu_alice,
            pyu_bob,
        ],
        aggregator=PlainAggregator(pyu_carol),
        comparator=PlainComparator(pyu_carol),
    )
    hdf.fillna(0, inplace=True)
    hdf_alice = reveal(hdf.partitions[pyu_alice].data)
    hdf_bob = reveal(hdf.partitions[pyu_bob].data)

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
        'hdf': hdf,
        'hdf_alice': hdf_alice,
        'hdf_bob': hdf_bob,
        'vdf_alice': vdf_alice,
        'vdf_bob': vdf_bob,
        'vdf': vdf,
    }


def on_hdataframe(
    env, data, sf_est: KBinsDiscretizer, sk_est: SkKBinsDiscretizer = None
):
    # GIVEN
    selected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # WHEN
    value = sf_est.fit_transform(data['hdf'][selected_cols])
    params = sf_est.get_params()
    assert params

    if sk_est is not None:
        sk_est.fit(
            pd.concat(
                [data['hdf_alice'][selected_cols], data['hdf_bob'][selected_cols]]
            )
        )
        expect_alice = sk_est.transform(data['hdf_alice'][selected_cols])
        np.testing.assert_almost_equal(
            reveal(value.partitions[env.alice].data),
            expect_alice,
        )
        expect_bob = sk_est.transform(data['hdf_bob'][selected_cols])
        np.testing.assert_almost_equal(
            reveal(value.partitions[env.bob].data), expect_bob
        )


@pytest.mark.mpc(parties=3)
def test_on_hdataframe_should_ok_when_quantile(prod_env_and_data):
    env, data = prod_env_and_data
    sf_est = KBinsDiscretizer(n_bins=5, strategy='quantile')
    on_hdataframe(env, data, sf_est)


@pytest.mark.mpc(parties=3)
def test_on_hdataframe_should_ok_when_uniform(prod_env_and_data):
    env, data = prod_env_and_data
    sf_est = KBinsDiscretizer(n_bins=5, strategy='uniform')
    sk_est = SkKBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    on_hdataframe(env, data, sf_est, sk_est)


def on_vdataframe(
    env, data, sf_est: KBinsDiscretizer, sk_est: SkKBinsDiscretizer = None
):
    # WHEN
    value = sf_est.fit_transform(data['vdf'][['a3', 'b4', 'b6']])
    params = sf_est.get_params()
    assert params

    if sk_est is not None:
        expect_alice = sk_est.fit_transform(data['vdf_alice'][['a3']])
        np.testing.assert_equal(reveal(value.partitions[env.alice].data), expect_alice)

        expect_bob = sk_est.fit_transform(data['vdf_bob'][['b4', 'b6']])
        np.testing.assert_equal(reveal(value.partitions[env.bob].data), expect_bob)


@pytest.mark.mpc(parties=3)
def test_on_vdataframe_should_ok_when_quantile(prod_env_and_data):
    env, data = prod_env_and_data
    sf_est = KBinsDiscretizer(n_bins=5, strategy='quantile')
    on_vdataframe(env, data, sf_est)


@pytest.mark.mpc(parties=3)
def test_on_vdataframe_should_ok_when_uniform(prod_env_and_data):
    env, data = prod_env_and_data
    sf_est = KBinsDiscretizer(n_bins=5, strategy='uniform')
    sk_est = SkKBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    on_vdataframe(env, data, sf_est, sk_est)


def on_h_mixdataframe(
    env, data, sf_est: KBinsDiscretizer, sk_est: SkKBinsDiscretizer = None
):
    # GIVEN
    df_part0 = pd.DataFrame(
        {
            'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4'],
            'a2': ['A2', 'B2', 'C2', 'D2', 'A5', 'B5', 'C5', 'D5'],
            'a3': [5, 1, 2, 6, 15, 3, 23, 6],
        }
    )

    df_part1 = pd.DataFrame(
        {
            'b4': [10.2, 20.5, -2.3, -0.4, 4, 0.5, 30, -10.4],
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

    # WHEN
    value = sf_est.fit_transform(
        h_mix[['a3', 'b4', 'b6']],
        aggregator=PlainAggregator(env.alice),
        comparator=PlainComparator(env.alice),
    )
    params = sf_est.get_params()

    # THEN
    assert params
    if sk_est is not None:
        expect_alice = sk_est.fit_transform(df_part0[['a3']])
        np.testing.assert_equal(
            pd.concat(
                [
                    reveal(value.partitions[0].partitions[env.alice].data),
                    reveal(value.partitions[1].partitions[env.alice].data),
                ]
            ),
            expect_alice,
        )
        expect_bob = sk_est.fit_transform(df_part1[['b4', 'b6']])
        np.testing.assert_equal(
            pd.concat(
                [
                    reveal(value.partitions[0].partitions[env.bob].data),
                    reveal(value.partitions[1].partitions[env.bob].data),
                ]
            ),
            expect_bob,
        )


@pytest.mark.mpc(parties=3)
def test_on_h_mixdataframe_should_ok_when_uniform(prod_env_and_data):
    env, data = prod_env_and_data
    sf_est = KBinsDiscretizer(n_bins=5, strategy='uniform')
    sk_est = SkKBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    on_h_mixdataframe(env, data, sf_est, sk_est)


@pytest.mark.mpc(parties=3)
def test_on_h_mixdataframe_should_ok_when_quantile(prod_env_and_data):
    env, data = prod_env_and_data
    sf_est = KBinsDiscretizer(n_bins=3, strategy='quantile')
    on_h_mixdataframe(env, data, sf_est)


def on_v_mixdataframe(
    env, data, sf_est: KBinsDiscretizer, sk_est: SkKBinsDiscretizer = None
):
    # GIVEN
    df_part0 = pd.DataFrame(
        {
            'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4'],
            'a2': ['A2', 'B2', 'C2', 'D2', 'A5', 'B5', 'C5', 'D5'],
            'a3': [5, 1, 2, 6, 15, 0, 23, 6],
        }
    )

    df_part1 = pd.DataFrame(
        {
            'b4': [10.2, 20.5, 5.3, -0.4, 5, 0.5, 15.5, -10.4],
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

    # WHEN
    value = sf_est.fit_transform(v_mix[['a3', 'b4', 'b6']])
    params = sf_est.get_params()

    # THEN
    assert params
    if sk_est is not None:
        expect_alice = sk_est.fit_transform(df_part0[['a3']])
        np.testing.assert_equal(
            pd.concat(
                [
                    reveal(value.partitions[0].partitions[env.alice].data),
                    reveal(value.partitions[0].partitions[env.bob].data),
                ]
            ),
            expect_alice,
        )
        expect_bob = sk_est.fit_transform(df_part1[['b4', 'b6']])
        np.testing.assert_almost_equal(
            pd.concat(
                [
                    reveal(value.partitions[1].partitions[env.alice].data),
                    reveal(value.partitions[1].partitions[env.bob].data),
                ]
            ),
            expect_bob,
        )


@pytest.mark.mpc(parties=3)
def test_on_v_mixdataframe_should_ok_when_quantile(prod_env_and_data):
    env, data = prod_env_and_data
    sf_est = KBinsDiscretizer(n_bins=5, strategy='quantile')
    on_v_mixdataframe(env, data, sf_est)


@pytest.mark.mpc(parties=3)
def test_on_v_mixdataframe_should_ok_when_uniform(prod_env_and_data):
    env, data = prod_env_and_data
    sf_est = KBinsDiscretizer(n_bins=5, strategy='uniform')
    sk_est = SkKBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    on_v_mixdataframe(env, data, sf_est, sk_est)


@pytest.mark.mpc(parties=3)
def test_should_error_when_not_dataframe(prod_env_and_data):
    env, data = prod_env_and_data
    est = KBinsDiscretizer()
    with pytest.raises(
        AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
    ):
        est.fit(['test'])
    est.fit(data['vdf']['a3'])
    with pytest.raises(
        AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
    ):
        est.transform('test')


@pytest.mark.mpc(parties=3)
def test_transform_should_error_when_not_fit(prod_env_and_data):
    env, data = prod_env_and_data
    with pytest.raises(AssertionError, match='Discretizer has not been fit yet.'):
        KBinsDiscretizer().transform('test')


@pytest.mark.mpc(parties=3)
def test_tranform_should_error_when_diff_features_num(prod_env_and_data):
    env, data = prod_env_and_data
    est = KBinsDiscretizer()
    est.fit(data['hdf'][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
    with pytest.raises(
        AssertionError,
        match='X has 1 features but KBinsDiscretizeris expecting 4 features as input.',
    ):
        est.transform(data['hdf'][['sepal_length']])
