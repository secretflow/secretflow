import numpy as np
import pandas as pd
import pytest
from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats.groupby_v import (
    ordinal_encoded_groupby_agg,
    ordinal_encoded_groupby_aggs,
)


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


@pytest.mark.parametrize(
    "by",
    [
        ["b5"],
        ['a2', 'b5'],
    ],
)
@pytest.mark.parametrize(
    "values",
    [["a3", "b4"], ['b6']],  # non-numeric columns cannot be here
)
@pytest.mark.parametrize(
    "agg_name",
    ["min", "count", "var"],
)
def test_groupby_agg(prod_env_and_data, by, values, agg_name):
    assert len(set(by).intersection(set(values))) == 0, "no intersection allowed"
    env, data = prod_env_and_data
    # GIVEN
    df = data['df'][['a1', 'a2', 'a3', 'b4', 'b5', 'b6']]
    df[["a1", "a2", "b5"]] = df[["a1", "a2", "b5"]].fillna(value="0", inplace=False)
    df[["a3", "b4", "b6"]] = (
        df[["a3", "b4", "b6"]].fillna(value=0, inplace=False).astype(float)
    )
    df_cleartext = data['df_cleartext']
    df_cleartext[["a1", "a2", "b5"]] = df_cleartext[["a1", "a2", "b5"]].fillna(
        value="0", inplace=False
    )
    df_cleartext[["a3", "b4", "b6"]] = (
        df_cleartext[["a3", "b4", "b6"]].fillna(value=0, inplace=False).astype(float)
    )

    our_values = ordinal_encoded_groupby_agg(df, by, values, env.spu, agg_name)
    true_values = getattr(df_cleartext.groupby(by)[values], agg_name)().fillna(
        value=0, inplace=False
    )
    decimal = 6
    if agg_name == "var":
        decimal = 3
    np.testing.assert_array_almost_equal(our_values, true_values, decimal=decimal)


@pytest.mark.parametrize(
    "by",
    [
        ["b5"],
        ['a2', 'b5'],
    ],
)
@pytest.mark.parametrize(
    "values",
    [["a3", "b4"], ['b6']],  # non-numeric columns cannot be here
)
@pytest.mark.parametrize(
    "aggs",
    [["min", "sum"], ["count", "max", "var"]],
)
def test_groupby_aggs(prod_env_and_data, by, values, aggs):
    assert len(set(by).intersection(set(values))) == 0, "no intersection allowed"
    env, data = prod_env_and_data
    # GIVEN
    df = data['df'][['a1', 'a2', 'a3', 'b4', 'b5', 'b6']]
    df[["a1", "a2", "b5"]] = df[["a1", "a2", "b5"]].fillna(value="0", inplace=False)
    df[["a3", "b4", "b6"]] = (
        df[["a3", "b4", "b6"]].fillna(value=0, inplace=False).astype(float)
    )
    df_cleartext = data['df_cleartext']
    df_cleartext[["a1", "a2", "b5"]] = df_cleartext[["a1", "a2", "b5"]].fillna(
        value="0", inplace=False
    )
    df_cleartext[["a3", "b4", "b6"]] = (
        df_cleartext[["a3", "b4", "b6"]].fillna(value=0, inplace=False).astype(float)
    )

    our_result = ordinal_encoded_groupby_aggs(df, by, values, env.spu, aggs)

    true_result = {}
    for agg in aggs:
        true_result[agg] = getattr(df_cleartext.groupby(by)[values], agg)().fillna(
            value=0, inplace=False
        )
    for agg in aggs:
        true_values = true_result[agg]
        our_values = our_result[agg]
        decimal = 6
        if agg == "var":
            decimal = 3
        np.testing.assert_array_almost_equal(our_values, true_values, decimal=decimal)
