import numpy as np
import pandas as pd
import pytest

from secretflow.data.base import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.preprocessing.cond_filter_v import ConditionFilter


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
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
            sf_production_setup_devices.alice: partition(
                data=sf_production_setup_devices.alice(lambda: vdf_alice)()
            ),
            sf_production_setup_devices.bob: partition(
                data=sf_production_setup_devices.bob(lambda: vdf_bob)()
            ),
        }
    )

    yield sf_production_setup_devices, {
        'vdf_alice': vdf_alice,
        'vdf_bob': vdf_bob,
        'vdf': vdf,
    }


def test_constructor_valid_values():
    filter = ConditionFilter("field1", "EQ", "STRING", ["value"], 0.1)
    np.testing.assert_equal(filter.field_name, "field1")
    np.testing.assert_equal(filter.comparator, "EQ")
    np.testing.assert_equal(filter.value_type, "STRING")
    np.testing.assert_equal(filter.bound_value, ["value"])
    np.testing.assert_equal(filter.float_epsilon, 0.1)


def test_constructor_invalid_comparator():
    with np.testing.assert_raises(ValueError):
        ConditionFilter("field1", "INVALID", "STRING", ["value"], 0.1)


def test_constructor_invalid_value_type():
    with np.testing.assert_raises(ValueError):
        ConditionFilter("field1", "EQ", "INVALID", ["value"], 0.1)


def test_constructor_invalid_bound_value():
    with np.testing.assert_raises(ValueError):
        ConditionFilter("field1", "EQ", "STRING", ["value1", "value2"], 0.1)


def test_fit_valid_df(prod_env_and_data):
    filter = ConditionFilter("b5", "IN", "STRING", ["B9"], 0.1)
    env, data = prod_env_and_data
    df = data['vdf']
    filter = filter.fit(df)
    np.testing.assert_equal(type(filter), ConditionFilter)
    np.testing.assert_(filter.in_table is not None)
    np.testing.assert_(filter.out_table is not None)


def test_transform_valid_df(prod_env_and_data):
    filter = ConditionFilter("a3", "LT", "FLOAT", ["3.14"], 0.1)
    env, data = prod_env_and_data
    df = data['vdf']
    filter = filter.fit(df)
    result = filter.transform(df)
    np.testing.assert_equal(type(result), VDataFrame)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 2)


def test_transform_valid_df_float(prod_env_and_data):
    filter = ConditionFilter("b4", "EQ", "FLOAT", ["10.1"], 0.1)
    env, data = prod_env_and_data
    df = data['vdf']
    filter = filter.fit(df)
    result = filter.transform(df)
    np.testing.assert_equal(type(result), VDataFrame)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 1)


def test_transform_valid_df_float_in(prod_env_and_data):
    filter = ConditionFilter("b4", "IN", "FLOAT", ["10.1", "20.4"], 0.15)
    env, data = prod_env_and_data
    df = data['vdf']
    filter = filter.fit(df)
    result = filter.transform(df)
    np.testing.assert_equal(type(result), VDataFrame)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 2)


def test_fit_transform_valid_df(prod_env_and_data):
    filter = ConditionFilter("b4", "LT", "FLOAT", ["11"], 0.1)
    env, data = prod_env_and_data
    df = data['vdf']
    result = filter.fit_transform(df)
    np.testing.assert_equal(type(result), VDataFrame)
    data_len = result.count().max()
    np.testing.assert_equal(data_len, 2)
