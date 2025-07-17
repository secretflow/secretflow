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
import polars as pl
import pytest
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from sklearn.utils.validation import column_or_1d

from secretflow import reveal
from secretflow.data import partition
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.mix.dataframe import MixDataFrame
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow.utils.simulation.datasets import load_iris
from secretflow_fl.preprocessing.encoder_fl import LabelEncoder, OneHotEncoder
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def prod_env_and_label_encoder_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob
    pyu_carol = sf_production_setup_devices.carol

    hdf = load_iris(
        parts=[pyu_alice, pyu_bob],
        aggregator=PlainAggregator(pyu_alice),
        comparator=PlainComparator(pyu_carol),
    )
    hdf_alice = reveal(hdf.partitions[pyu_alice].data)
    hdf_bob = reveal(hdf.partitions[pyu_bob].data)

    vdf_alice = pd.DataFrame(
        {
            'a1': ['K5', 'K1', None, 'K6'],
            'a2': ['A5', 'A5', 'A2', 'A2'],
            'a3': [5, 1, 2, 6],
        }
    )

    vdf_bob = pd.DataFrame(
        {
            'b4': [10.2, 20.5, None, -0.4],
            'b5': ['B3', 'B2', 'B3', 'B4'],
            'b6': [3, 1, 9, 4],
        }
    )
    vdf_alice_poalrs = pl.from_pandas(vdf_alice)
    vdf_bob_polars = pl.from_pandas(vdf_bob)
    vdf = VDataFrame(
        {
            pyu_alice: partition(
                data=pyu_alice(lambda: vdf_alice_poalrs)(),
                backend="polars",
            ),
            pyu_bob: partition(
                data=pyu_bob(lambda: vdf_bob_polars)(),
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


@pytest.mark.mpc(parties=3)
class TestLabelEncoder:
    def test_on_hdataframe_should_ok(self, prod_env_and_label_encoder_data):
        env, data = prod_env_and_label_encoder_data
        # GIVEN
        encoder = LabelEncoder()

        # WHEN
        value = encoder.fit_transform(data['hdf']['class'])
        params = encoder.get_params()

        # THEN
        assert params
        sk_encoder = SkLabelEncoder()
        sk_encoder.fit(
            column_or_1d(
                pd.concat([data['hdf_alice'][['class']], data['hdf_bob'][['class']]])
            )
        )
        expect_alice = sk_encoder.transform(column_or_1d(data['hdf_alice'][['class']]))[
            np.newaxis
        ].T
        np.testing.assert_equal(reveal(value.partitions[env.alice].data), expect_alice)
        expect_bob = sk_encoder.transform(column_or_1d(data['hdf_bob'][['class']]))[
            np.newaxis
        ].T
        np.testing.assert_equal(reveal(value.partitions[env.bob].data), expect_bob)
        assert value.dtypes['class'] == np.int64

    def test_on_vdataframe_should_ok(self, prod_env_and_label_encoder_data):
        env, data = prod_env_and_label_encoder_data
        # GIVEN
        encoder = LabelEncoder()

        # WHEN
        value = encoder.fit_transform(data['vdf']['a2'])
        params = encoder.get_params()

        # THEN
        assert params
        assert len(value.partitions) == 1
        sk_encoder = SkLabelEncoder()
        expect_alice = sk_encoder.fit_transform(
            column_or_1d(data['vdf_alice'][['a2']])
        )[np.newaxis].T
        np.testing.assert_equal(
            reveal(value.to_pandas().partitions[env.alice].data), expect_alice
        )
        assert value.to_pandas().dtypes['a2'] == np.int64

    def test_on_h_mixdataframe_should_ok(self, prod_env_and_label_encoder_data):
        env, data = prod_env_and_label_encoder_data
        # GIVEN
        df = pd.DataFrame({'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4']})
        h_part0 = VDataFrame(
            {env.alice: partition(data=env.alice(lambda: df.iloc[:4, :])())}
        )
        h_part1 = VDataFrame(
            {env.alice: partition(data=env.alice(lambda: df.iloc[4:, :])())}
        )
        h_mix = MixDataFrame(partitions=[h_part0, h_part1])

        encoder = LabelEncoder()

        # WHEN
        value = encoder.fit_transform(h_mix)
        params = encoder.get_params()

        # THEN
        assert params
        sk_encoder = SkLabelEncoder()
        expected = sk_encoder.fit_transform(column_or_1d(df))[np.newaxis].T
        np.testing.assert_equal(
            pd.concat(
                [
                    reveal(value.partitions[0].partitions[env.alice].data),
                    reveal(value.partitions[1].partitions[env.alice].data),
                ]
            ),
            expected,
        )

    def test_on_v_mixdataframe_should_ok(self, prod_env_and_label_encoder_data):
        env, data = prod_env_and_label_encoder_data
        # GIVEN
        df = pd.DataFrame({'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4']})
        v_part0 = HDataFrame(
            {
                env.alice: partition(data=env.alice(lambda: df.iloc[:4, :])()),
                env.bob: partition(data=env.bob(lambda: df.iloc[4:, :])()),
            },
            aggregator=PlainAggregator(env.carol),
            comparator=PlainComparator(env.carol),
        )
        v_mix = MixDataFrame(partitions=[v_part0])

        encoder = LabelEncoder()

        # WHEN
        value = encoder.fit_transform(v_mix)
        params = encoder.get_params()

        # THEN
        assert params
        sk_encoder = SkLabelEncoder()
        expected = sk_encoder.fit_transform(column_or_1d(df))[np.newaxis].T
        np.testing.assert_equal(
            pd.concat(
                [
                    reveal(value.partitions[0].partitions[env.alice].data),
                    reveal(value.partitions[0].partitions[env.bob].data),
                ]
            ),
            expected,
        )

    def test_on_hdataframe_more_than_one_column_should_error(
        self,
        prod_env_and_label_encoder_data,
    ):
        env, data = prod_env_and_label_encoder_data
        with pytest.raises(
            AssertionError,
            match='DataFrame to encode should have one and only one column',
        ):
            encoder = LabelEncoder()
            encoder.fit_transform(data['hdf'])

    def test_on_vdataframe_more_than_one_column_should_error(
        self,
        prod_env_and_label_encoder_data,
    ):
        env, data = prod_env_and_label_encoder_data
        with pytest.raises(
            AssertionError,
            match='DataFrame to encode should have one and only one column',
        ):
            encoder = LabelEncoder()
            encoder.fit_transform(data['vdf'])

    def test_should_error_when_not_dataframe(self, prod_env_and_label_encoder_data):
        env, data = prod_env_and_label_encoder_data
        encoder = LabelEncoder()
        with pytest.raises(
            AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
        ):
            encoder.fit(['test'])
        encoder.fit(data['vdf']['a2'])
        with pytest.raises(
            AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
        ):
            encoder.transform(['test'])

    def test_transform_should_error_when_not_fit(self, prod_env_and_label_encoder_data):
        env, data = prod_env_and_label_encoder_data
        with pytest.raises(AssertionError, match='Encoder has not been fit yet.'):
            LabelEncoder().transform('test')


@mpc_fixture
def prod_env_and_onehot_encoder_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob

    hdf = load_iris(
        parts=[pyu_alice, pyu_bob],
        aggregator=PlainAggregator(pyu_alice),
        comparator=PlainComparator(pyu_alice),
    )
    hdf_alice = reveal(hdf.partitions[pyu_alice].data)
    hdf_bob = reveal(hdf.partitions[pyu_bob].data)

    vdf_alice = pd.DataFrame(
        {
            'a1': ['K5', 'K1', None, 'K6'],
            'a2': ['A5', 'A5', 'A2', 'A2'],
            'a3': [5, 1, 2, 1],
        }
    )

    vdf_bob = pd.DataFrame(
        {
            'b4': [10.2, 20.5, None, -0.4],
            'b5': ['B3', 'B2', 'B3', 'B4'],
            'b6': [3, 1, 9, 4],
        }
    )

    vdf = VDataFrame(
        {
            pyu_alice: partition(
                data=pyu_alice(lambda: pl.from_pandas(vdf_alice))(),
                backend="polars",
            ),
            pyu_bob: partition(
                data=pyu_bob(lambda: pl.from_pandas(vdf_bob))(),
                backend="polars",
            ),
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


@pytest.mark.mpc
class TestOneHotEncoder:
    def test_on_hdataframe_should_ok(self, prod_env_and_onehot_encoder_data):
        env, data = prod_env_and_onehot_encoder_data
        # GIVEN
        encoder = OneHotEncoder()

        # WHEN
        value = encoder.fit_transform(data['hdf']['class'])
        params = encoder.get_params()

        # THEN
        assert params
        sk_encoder = SkOneHotEncoder()
        sk_encoder.fit(
            pd.concat([data['hdf_alice'][['class']], data['hdf_bob'][['class']]])
        )
        expect_alice = sk_encoder.transform(data['hdf_alice'][['class']]).toarray()
        np.testing.assert_equal(reveal(value.partitions[env.alice].data), expect_alice)
        expect_bob = sk_encoder.transform(data['hdf_bob'][['class']]).toarray()
        np.testing.assert_equal(reveal(value.partitions[env.bob].data), expect_bob)

    def test_on_vdataframe_should_ok(self, prod_env_and_onehot_encoder_data):
        env, data = prod_env_and_onehot_encoder_data
        # GIVEN
        encoder = OneHotEncoder()

        # WHEN
        value = encoder.fit_transform(data['vdf'][['a1', 'a2', 'b5']])
        params = encoder.get_params()

        # THEN
        assert params
        sk_encoder = SkOneHotEncoder()
        expect_alice = sk_encoder.fit_transform(
            data['vdf_alice'][['a1', 'a2']]
        ).toarray()
        np.testing.assert_equal(reveal(value.partitions[env.alice].data), expect_alice)
        expect_bob = sk_encoder.fit_transform(data['vdf_bob'][['b5']]).toarray()
        np.testing.assert_equal(reveal(value.partitions[env.bob].data), expect_bob)

    def test_on_h_mixdataframe_should_ok(self, prod_env_and_onehot_encoder_data):
        env, data = prod_env_and_onehot_encoder_data
        # GIVEN
        df_part0 = pd.DataFrame(
            {
                'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'C4'],
                'a2': [5, 5, 2, 6, 15, None, 23, 6],
            }
        )

        df_part1 = pd.DataFrame(
            {'b4': [10.2, 20.5, None, -0.4, None, 0.5, None, -10.4]}
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

        encoder = OneHotEncoder()

        # WHEN
        value = encoder.fit_transform(h_mix)
        params = encoder.get_params()

        # THEN
        assert params
        sk_encoder = SkOneHotEncoder()
        expected = sk_encoder.fit_transform(
            pd.concat([df_part0, df_part1], axis=1)
        ).toarray()
        np.testing.assert_equal(
            pd.concat(
                [
                    pd.concat(
                        [
                            reveal(value.partitions[0].partitions[env.alice].data),
                            reveal(value.partitions[1].partitions[env.alice].data),
                        ]
                    ),
                    pd.concat(
                        [
                            reveal(value.partitions[0].partitions[env.bob].data),
                            reveal(value.partitions[1].partitions[env.bob].data),
                        ]
                    ),
                ],
                axis=1,
            ),
            expected,
        )

    @pytest.mark.mpc(parties=3)
    def test_on_v_mixdataframe_should_ok(self, prod_env_and_onehot_encoder_data):
        env, data = prod_env_and_onehot_encoder_data
        # GIVEN
        df_part0 = pd.DataFrame(
            {
                'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'C4'],
                'a2': [5, 5, 2, 6, 15, None, 23, 6],
            }
        )

        df_part1 = pd.DataFrame(
            {'b4': [10.2, 20.5, None, -0.4, None, 0.5, None, -10.4]}
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

        encoder = OneHotEncoder()

        # WHEN
        value = encoder.fit_transform(v_mix)
        params = encoder.get_params()

        # THEN
        assert params
        sk_encoder = SkOneHotEncoder()
        expected = sk_encoder.fit_transform(
            pd.concat([df_part0, df_part1], axis=1)
        ).toarray()
        np.testing.assert_equal(
            pd.concat(
                [
                    pd.concat(
                        [
                            reveal(value.partitions[0].partitions[env.alice].data),
                            reveal(value.partitions[0].partitions[env.bob].data),
                        ]
                    ),
                    pd.concat(
                        [
                            reveal(value.partitions[1].partitions[env.alice].data),
                            reveal(value.partitions[1].partitions[env.bob].data),
                        ]
                    ),
                ],
                axis=1,
            ),
            expected,
        )

    def test_min_frequency_on_vdataframe_should_ok(
        self, prod_env_and_onehot_encoder_data
    ):
        env, data = prod_env_and_onehot_encoder_data
        # WHEN
        selected_columns = ['a1', 'a2', 'b5']
        encoder = OneHotEncoder(min_frequency=2)
        df = encoder.fit_transform(data['vdf'][selected_columns])
        params = encoder.get_params()

        # THEN
        assert params
        sk_encoder = SkOneHotEncoder(min_frequency=2)
        expect_alice = pd.DataFrame(
            sk_encoder.fit_transform(data['vdf_alice'][['a1', 'a2']]).toarray(),
            columns=sk_encoder.get_feature_names_out(),
        )
        assert set(expect_alice.columns).issubset(set(df.partitions[env.alice].columns))

        alice_columns = expect_alice.columns
        pd.testing.assert_frame_equal(
            reveal(df.to_pandas().partitions[env.alice][alice_columns].data),
            expect_alice,
        )

        expect_bob = pd.DataFrame(
            sk_encoder.fit_transform(data['vdf_bob'][['b5']]).toarray(),
            columns=sk_encoder.get_feature_names_out(),
        )
        assert set(expect_bob.columns).issubset(set(df.partitions[env.bob].columns))

        bob_columns = expect_bob.columns
        pd.testing.assert_frame_equal(
            reveal(df.to_pandas().partitions[env.bob][bob_columns].data), expect_bob
        )

    def test_max_categories_on_vdataframe_should_ok(
        self, prod_env_and_onehot_encoder_data
    ):
        env, data = prod_env_and_onehot_encoder_data
        # WHEN
        selected_columns = ['a1', 'a2', 'b5']
        encoder = OneHotEncoder(max_categories=3)
        df = encoder.fit_transform(data['vdf'][selected_columns])
        params = encoder.get_params()

        # THEN
        assert params
        sk_encoder = SkOneHotEncoder(max_categories=3)
        expect_alice = pd.DataFrame(
            sk_encoder.fit_transform(data['vdf_alice'][['a1', 'a2']]).toarray(),
            columns=sk_encoder.get_feature_names_out(),
        )
        assert set(expect_alice.columns).issubset(set(df.partitions[env.alice].columns))

        alice_columns = expect_alice.columns
        pd.testing.assert_frame_equal(
            reveal(df.to_pandas().partitions[env.alice][alice_columns].data),
            expect_alice,
        )

        expect_bob = pd.DataFrame(
            sk_encoder.fit_transform(data['vdf_bob'][['b5']]).toarray(),
            columns=sk_encoder.get_feature_names_out(),
        )
        assert set(expect_bob.columns).issubset(set(df.partitions[env.bob].columns))

        bob_columns = expect_bob.columns
        pd.testing.assert_frame_equal(
            reveal(df.to_pandas().partitions[env.bob][bob_columns].data), expect_bob
        )

    def test_should_error_on_hdataframe_with_args(
        self, prod_env_and_onehot_encoder_data
    ):
        env, data = prod_env_and_onehot_encoder_data
        encoder = OneHotEncoder(min_frequency=3)
        with pytest.raises(
            AssertionError,
            match='Args min_frequency/max_categories are only supported in VDataFrame',
        ):
            encoder.fit_transform(data['hdf'])

        encoder = OneHotEncoder(max_categories=3)
        with pytest.raises(
            AssertionError,
            match='Args min_frequency/max_categories are only supported in VDataFrame',
        ):
            encoder.fit_transform(data['hdf'])

    def test_should_error_when_not_dataframe(self, prod_env_and_onehot_encoder_data):
        env, data = prod_env_and_onehot_encoder_data
        encoder = OneHotEncoder()
        with pytest.raises(
            AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
        ):
            encoder.fit(['test'])
        encoder.fit(data['hdf'])
        with pytest.raises(
            AssertionError, match='Accepts HDataFrame/VDataFrame/MixDataFrame only'
        ):
            encoder.transform(['test'])

    def test_transform_should_error_when_not_fit(
        self, prod_env_and_onehot_encoder_data
    ):
        env, data = prod_env_and_onehot_encoder_data
        with pytest.raises(AssertionError, match='Encoder has not been fit yet.'):
            OneHotEncoder().transform('test')
