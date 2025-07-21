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
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.mix import MixDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.security.aggregation import PlainAggregator
from secretflow.security.compare import PlainComparator
from secretflow.utils.errors import InvalidArgumentError
from tests.sf_fixtures import DeviceInventory, mpc_fixture


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices: DeviceInventory):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob
    pyu_carol = sf_production_setup_devices.carol

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
            pyu_alice: partition(data=pyu_alice(lambda: df_part0.iloc[:4, :])()),
            pyu_bob: partition(data=pyu_bob(lambda: df_part1.iloc[:4, :])()),
        }
    )
    h_part1 = VDataFrame(
        {
            pyu_alice: partition(data=pyu_alice(lambda: df_part0.iloc[4:, :])()),
            pyu_bob: partition(data=pyu_bob(lambda: df_part1.iloc[4:, :])()),
        }
    )
    h_mix = MixDataFrame(partitions=[h_part0, h_part1])

    v_part0 = HDataFrame(
        {
            pyu_alice: partition(data=pyu_alice(lambda: df_part0.iloc[:4, :])()),
            pyu_bob: partition(data=pyu_bob(lambda: df_part0.iloc[4:, :])()),
        },
        aggregator=PlainAggregator(pyu_carol),
        comparator=PlainComparator(pyu_carol),
    )
    v_part1 = HDataFrame(
        {
            pyu_alice: partition(data=pyu_alice(lambda: df_part1.iloc[:4, :])()),
            pyu_bob: partition(data=pyu_bob(lambda: df_part1.iloc[4:, :])()),
        },
        aggregator=PlainAggregator(pyu_carol),
        comparator=PlainComparator(pyu_carol),
    )
    v_mix = MixDataFrame(partitions=[v_part0, v_part1])

    return sf_production_setup_devices, {
        "df_part0": df_part0,
        "df_part1": df_part1,
        "h_part0": h_part0,
        "h_part1": h_part1,
        "h_mix": h_mix,
        "v_part0": v_part0,
        "v_part1": v_part1,
        "v_mix": v_mix,
    }


@pytest.mark.mpc(parties=3)
def test_mean_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value_h, value_v = (
        data['h_mix'].mean(numeric_only=True),
        data['v_mix'].mean(numeric_only=True),
    )

    # THEN
    expected = pd.concat(
        [
            data['df_part0'].mean(numeric_only=True),
            data['df_part1'].mean(numeric_only=True),
        ]
    )
    pd.testing.assert_series_equal(value_h, expected)
    pd.testing.assert_series_equal(value_v, expected)


@pytest.mark.mpc(parties=3)
def test_min_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value_h, value_v = (
        data['h_mix'].min(numeric_only=True),
        data['v_mix'].min(numeric_only=True),
    )

    # THEN
    expected = pd.concat(
        [
            data['df_part0'].min(numeric_only=True),
            data['df_part1'].min(numeric_only=True),
        ]
    )
    pd.testing.assert_series_equal(value_h, expected)
    pd.testing.assert_series_equal(value_v, expected)


@pytest.mark.mpc(parties=3)
def test_max_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value_h, value_v = (
        data['h_mix'].max(numeric_only=True),
        data['v_mix'].max(numeric_only=True),
    )

    # THEN
    expected = pd.concat(
        [
            data['df_part0'].max(numeric_only=True),
            data['df_part1'].max(numeric_only=True),
        ]
    )
    pd.testing.assert_series_equal(value_h, expected)
    pd.testing.assert_series_equal(value_v, expected)


@pytest.mark.mpc(parties=3)
def test_count_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value_h, value_v = data['h_mix'].count(), data['v_mix'].count()

    # THEN
    expected = pd.concat([data['df_part0'].count(), data['df_part1'].count()])
    pd.testing.assert_series_equal(value_h, expected)
    pd.testing.assert_series_equal(value_v, expected)


@pytest.mark.mpc(parties=3)
def test_len_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value_h, value_v = len(data['h_mix']), len(data['v_mix'])

    # THEN
    expected = len(data['df_part0'])
    assert value_h == expected
    assert value_v == expected


@pytest.mark.mpc(parties=3)
def test_getitem_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # Case 1: single item.
    # WHEN
    value_h, value_v = data['h_mix']['a1'], data['v_mix']['a1']
    # THEN
    expected_alice = data['df_part0'][['a1']]
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value_h.partitions[0].partitions[env.alice].data),
                reveal(value_h.partitions[1].partitions[env.alice].data),
            ]
        ),
        expected_alice,
    )
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value_v.partitions[0].partitions[env.alice].data),
                reveal(value_v.partitions[0].partitions[env.bob].data),
            ]
        ),
        expected_alice,
    )

    # Case 2: multi items.
    # WHEN
    value_h, value_v = (
        data['h_mix'][['a2', 'b4', 'b5']],
        data['v_mix'][['a2', 'b4', 'b5']],
    )
    # THEN
    expected_alice = data['df_part0'][['a2']]
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value_h.partitions[0].partitions[env.alice].data),
                reveal(value_h.partitions[1].partitions[env.alice].data),
            ]
        ),
        expected_alice,
    )
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value_v.partitions[0].partitions[env.alice].data),
                reveal(value_v.partitions[0].partitions[env.bob].data),
            ]
        ),
        expected_alice,
    )
    expected_bob = data['df_part1'][['b4', 'b5']]
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value_h.partitions[0].partitions[env.bob].data),
                reveal(value_h.partitions[1].partitions[env.bob].data),
            ]
        ),
        expected_bob,
    )
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value_v.partitions[1].partitions[env.alice].data),
                reveal(value_v.partitions[1].partitions[env.bob].data),
            ]
        ),
        expected_bob,
    )


@pytest.mark.mpc(parties=3)
def test_setitem_should_ok_when_single_value(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    value_h, value_v = data['h_mix'].copy(), data['v_mix'].copy()

    # WEHN
    value_h['a1'] = 'test'
    value_v['a1'] = 'test'

    # THEN
    assert (
        pd.concat(
            [
                reveal(value_h.partitions[0].partitions[env.alice].data),
                reveal(value_h.partitions[1].partitions[env.alice].data),
            ]
        )['a1']
        == 'test'
    ).all()

    assert (
        pd.concat(
            [
                reveal(value_v.partitions[0].partitions[env.alice].data),
                reveal(value_v.partitions[0].partitions[env.bob].data),
            ]
        )['a1']
        == 'test'
    ).all()


@pytest.mark.mpc(parties=3)
def test_setitem_should_ok_when_hmix(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    value = data['h_mix'].copy()
    v_alice = pd.DataFrame({'a1': [f'a{i}' for i in range(8)]})

    v_bob = pd.DataFrame(
        {'b4': [10.5 + i for i in range(8)], 'b6': [i for i in range(8)]}
    )
    part0 = VDataFrame(
        {
            env.alice: partition(data=env.alice(lambda: v_alice.iloc[:4, :])()),
            env.bob: partition(data=env.bob(lambda: v_bob.iloc[:4, :])()),
        }
    )
    part1 = VDataFrame(
        {
            env.alice: partition(data=env.alice(lambda: v_alice.iloc[4:, :])()),
            env.bob: partition(data=env.bob(lambda: v_bob.iloc[4:, :])()),
        }
    )
    to = MixDataFrame(partitions=[part0, part1])

    # WHEN
    value[['a1', 'b4', 'b6']] = to

    # THEN
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.alice].data),
                reveal(value.partitions[1].partitions[env.alice].data),
            ]
        )[['a1']],
        v_alice,
    )
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.bob].data),
                reveal(value.partitions[1].partitions[env.bob].data),
            ]
        )[['b4', 'b6']],
        v_bob,
    )


@pytest.mark.mpc(parties=3)
def test_setitem_should_ok_when_vmix(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    value = data['v_mix'].copy()
    v_alice = pd.DataFrame({'a1': [f'a{i}' for i in range(8)]})

    v_bob = pd.DataFrame(
        {'b4': [10.5 + i for i in range(8)], 'b6': [i for i in range(8)]}
    )
    part0 = HDataFrame(
        {
            env.alice: partition(data=env.alice(lambda: v_alice.iloc[:4, :])()),
            env.bob: partition(data=env.bob(lambda: v_alice.iloc[4:, :])()),
        }
    )
    part1 = HDataFrame(
        {
            env.alice: partition(data=env.alice(lambda: v_bob.iloc[:4, :])()),
            env.bob: partition(data=env.bob(lambda: v_bob.iloc[4:, :])()),
        }
    )
    to = MixDataFrame(partitions=[part0, part1])

    # WHEN
    value[['a1', 'b4', 'b6']] = to

    # THEN
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.alice].data),
                reveal(value.partitions[0].partitions[env.bob].data),
            ]
        )[['a1']],
        v_alice,
    )
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[1].partitions[env.alice].data),
                reveal(value.partitions[1].partitions[env.bob].data),
            ]
        )[['b4', 'b6']],
        v_bob,
    )


@pytest.mark.mpc(parties=3)
def test_astype_should_ok_when_vmix(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    value = data['v_mix'][['a3', 'b4']].fillna(1)

    # WHEN
    value = value.astype({'a3': np.int_})

    # THEN
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.alice].data),
                reveal(value.partitions[0].partitions[env.bob].data),
            ]
        ),
        data['df_part0'][['a3']].fillna(1).astype(np.int_),
    )
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[1].partitions[env.alice].data),
                reveal(value.partitions[1].partitions[env.bob].data),
            ]
        ),
        data['df_part1'][['b4']].fillna(1),
    )


@pytest.mark.mpc(parties=3)
def test_astype_should_ok_when_hmix(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    value = data['h_mix'][['a3', 'b4']].fillna(1)

    # WHEN
    value = value.astype(np.int_)

    # THEN
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.alice].data),
                reveal(value.partitions[1].partitions[env.alice].data),
            ]
        ),
        data['df_part0'][['a3']].fillna(1).astype(np.int_),
    )
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                reveal(value.partitions[0].partitions[env.bob].data),
                reveal(value.partitions[1].partitions[env.bob].data),
            ]
        ),
        data['df_part1'][['b4']].fillna(1).astype(np.int_),
    )


@pytest.mark.mpc(parties=3)
def test_setitem_should_error_when_wrong_value_type(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    value = data['h_mix'].copy()
    to = pd.DataFrame({'a1': [f'test{i}' for i in range(8)]})

    # WHEN & THEN
    with pytest.raises(
        InvalidArgumentError,
        match='Can not assgin a HDataFrame/VDataFrame/Partition to MixDataFrame.',
    ):
        value['a1'] = partition(data=env.alice(lambda: to)())
    with pytest.raises(
        InvalidArgumentError,
        match='Can not assgin a HDataFrame/VDataFrame/Partition to MixDataFrame.',
    ):
        value['a1'] = data['h_part0']
    with pytest.raises(
        InvalidArgumentError,
        match='Can not assgin a HDataFrame/VDataFrame/Partition to MixDataFrame.',
    ):
        value['a1'] = data['v_part1']


@pytest.mark.mpc(parties=3)
def test_construct_should_error_when_diff_part_types(prod_env_and_data):
    env, data = prod_env_and_data
    with pytest.raises(AssertionError, match='All partitions should have same type'):
        MixDataFrame([data['h_part0'], data['v_part0']])


@pytest.mark.mpc(parties=3)
def test_construct_should_error_when_none_or_empty_parts(prod_env_and_data):
    env, data = prod_env_and_data
    with pytest.raises(AssertionError, match='Partitions should not be None or empty.'):
        MixDataFrame()

    with pytest.raises(AssertionError, match='Partitions should not be None or empty.'):
        MixDataFrame([])


@pytest.mark.mpc(parties=3)
def test_set_partitions_should_error_when_diff_types(prod_env_and_data):
    env, data = prod_env_and_data
    with pytest.raises(AssertionError, match='All partitions should have same type'):
        mix = MixDataFrame([data['h_part0'], data['h_part1']])
        mix.partitions = [data['h_part0'], data['v_part0']]


@pytest.mark.mpc(parties=3)
def test_set_partitions_should_error_when_none_or_empty_parts(prod_env_and_data):
    env, data = prod_env_and_data
    with pytest.raises(AssertionError, match='Partitions should not be None or empty.'):
        mix = MixDataFrame([data['h_part0'], data['h_part1']])
        mix.partitions = None

    with pytest.raises(AssertionError, match='Partitions should not be None or empty.'):
        mix = MixDataFrame([data['h_part0'], data['h_part1']])
        mix.partitions = []
