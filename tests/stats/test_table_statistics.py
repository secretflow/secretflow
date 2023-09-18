import pandas as pd
import pytest
from sklearn.datasets import load_iris

from secretflow.data.base import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.stats import table_statistics


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
    iris = load_iris(as_frame=True)
    data = pd.concat([iris.data, iris.target], axis=1)
    data.iloc[1, 1] = None
    data.iloc[100, 1] = None

    # Restore target to its original name.
    data['target'] = data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    # Vertical partitioning.
    v_alice, v_bob = data.iloc[:, :2], data.iloc[:, 2:]
    df_v = VDataFrame(
        partitions={
            sf_production_setup_devices.alice: partition(
                sf_production_setup_devices.alice(lambda: v_alice)()
            ),
            sf_production_setup_devices.bob: partition(
                sf_production_setup_devices.bob(lambda: v_bob)()
            ),
        }
    )
    df = data
    yield sf_production_setup_devices, {'df_v': df_v, 'df': df}


def test_table_statistics(prod_env_and_data):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    env, data = prod_env_and_data
    correct_summary = table_statistics(data['df'])
    summary = table_statistics(data['df_v'])
    result = summary.equals(correct_summary)
    if not result:
        n_rows = correct_summary.shape[0]
        n_cols = correct_summary.shape[1]
        assert n_rows == summary.shape[0], "row number mismatch"
        assert n_cols == summary.shape[1], "col number mismatch"
        for i in range(n_rows):
            for j in range(n_cols):
                assert (
                    correct_summary.iloc[i, j] == summary.iloc[i, j]
                ), "row {}, col {} mismatch".format(i, summary.columns[j])
