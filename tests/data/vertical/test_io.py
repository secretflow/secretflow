import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.vertical import VDataFrame, read_csv, to_csv


@pytest.fixture(scope='function')
def prod_env_and_data(sf_production_setup_devices):
    df1 = pd.DataFrame(
        {
            'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
            'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
            'c3': [5, 1, 2, 6, 4, 3],
        }
    )

    df2 = pd.DataFrame(
        {
            'c1': ['K3', 'K1', 'K9', 'K4'],
            'c4': ['B3', 'B1', 'B9', 'B4'],
            'c5': [3, 1, 9, 4],
        }
    )

    _, path1 = tempfile.mkstemp()
    _, path2 = tempfile.mkstemp()

    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    filepath = {
        sf_production_setup_devices.alice: path1,
        sf_production_setup_devices.bob: path2,
    }

    yield sf_production_setup_devices, filepath

    for path in filepath.values():
        os.remove(path)


def cleartmp(paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def test_read_csv(prod_env_and_data):
    env, data = prod_env_and_data
    df = read_csv(data, spu=env.spu, keys='c1', drop_keys={env.alice: 'c1'})

    expected_alice = pd.DataFrame({'c2': ['A1', 'A3', 'A4'], 'c3': [1, 3, 4]})
    df_alice = reveal(df.partitions[env.alice].data)
    pd.testing.assert_frame_equal(df_alice.reset_index(drop=True), expected_alice)

    expected_bob = pd.DataFrame(
        {'c1': ['K1', 'K3', 'K4'], 'c4': ['B1', 'B3', 'B4'], 'c5': [1, 3, 4]}
    )
    df_bob = reveal(df.partitions[env.bob].data)
    pd.testing.assert_frame_equal(df_bob.reset_index(drop=True), expected_bob)


def test_read_csv_drop_keys(prod_env_and_data):
    env, data = prod_env_and_data
    df = read_csv(data, spu=env.spu, keys='c1', drop_keys='c1')

    expected = pd.DataFrame({'c2': ['A1', 'A3', 'A4'], 'c3': [1, 3, 4]})
    pd.testing.assert_frame_equal(
        reveal(df.partitions[env.alice].data).reset_index(drop=True), expected
    )

    expected = pd.DataFrame({'c4': ['B1', 'B3', 'B4'], 'c5': [1, 3, 4]})
    pd.testing.assert_frame_equal(
        reveal(df.partitions[env.bob].data).reset_index(drop=True), expected
    )


def test_read_csv_with_dtypes(prod_env_and_data):
    env, data = prod_env_and_data
    dtypes = {
        env.alice: {'c1': str, 'c2': str},
        env.bob: {'c1': str, 'c5': np.int64},
    }
    df = read_csv(data, spu=env.spu, keys='c1', dtypes=dtypes, drop_keys='c1')

    expected = pd.DataFrame({'c2': ['A1', 'A3', 'A4']})
    pd.testing.assert_frame_equal(
        reveal(df.partitions[env.alice].data).reset_index(drop=True), expected
    )

    expected = pd.DataFrame({'c5': [1, 3, 4]})
    pd.testing.assert_frame_equal(
        reveal(df.partitions[env.bob].data).reset_index(drop=True), expected
    )


# @unittest.skip('spu reset not works now FIXME @raofei')
# def read_csv_mismatch_dtypes(prod_env_and_data):
#     env, data = prod_env_and_data
#     dtypes = {
#         env.alice: {'c1': str, 'c6': str},
#         env.bob: {'c1': str, 'c5': np.int64},
#     }
#     with pytest.raises(ValueError, 'Usecols do not match columns'):
#         read_csv(data, spu=env.spu, keys='c1', dtypes=dtypes, drop_keys='c1')

#     # reset spu to clear corrupted state
#     env.spu.reset()


def test_read_csv_duplicated_cols(prod_env_and_data):
    env, data = prod_env_and_data
    df1 = pd.DataFrame(
        {
            'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
            'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
            'c3': [5, 1, 2, 6, 4, 3],
        }
    )

    df2 = pd.DataFrame(
        {
            'c1': ['K3', 'K1', 'K9', 'K4'],
            'c2': ['B3', 'B1', 'B9', 'B4'],
            'c5': [3, 1, 9, 4],
        }
    )

    _, path1 = tempfile.mkstemp()
    _, path2 = tempfile.mkstemp()

    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    filepath = {env.alice: path1, env.bob: path2}
    with pytest.raises(AssertionError, match='duplicate in multiple devices'):
        read_csv(filepath, spu=env.spu, keys='c1', drop_keys='c1')

    for path in filepath.values():
        os.remove(path)


def test_read_csv_drop_keys_out_of_scope(prod_env_and_data):
    env, data = prod_env_and_data
    df1 = pd.DataFrame(
        {
            'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
            'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
            'c3': [5, 1, 2, 6, 4, 3],
        }
    )

    df2 = pd.DataFrame(
        {
            'c1': ['K3', 'K1', 'K9', 'K4'],
            'c2': ['B3', 'B1', 'B9', 'B4'],
            'c5': [3, 1, 9, 4],
        }
    )

    _, path1 = tempfile.mkstemp()
    _, path2 = tempfile.mkstemp()

    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    filepath = {env.alice: path1, env.bob: path2}
    with pytest.raises(
        AssertionError, match='can not find on device_psi_key_set of device'
    ):
        read_csv(
            filepath,
            spu=env.spu,
            keys=['c1', 'c2'],
            drop_keys={env.alice: ['c1', 'c3'], env.bob: ['c2']},
        )

    for path in filepath.values():
        os.remove(path)


def test_read_csv_without_psi(prod_env_and_data):
    env, data = prod_env_and_data
    df1 = pd.DataFrame({'c2': ['A5', 'A1', 'A2', 'A6'], 'c3': [5, 1, 2, 6]})

    df2 = pd.DataFrame({'c4': ['B3', 'B1', 'B9', 'B4'], 'c5': [3, 1, 9, 4]})

    _, path1 = tempfile.mkstemp()
    _, path2 = tempfile.mkstemp()

    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    filepath = {env.alice: path1, env.bob: path2}
    dtypes = {
        env.alice: {'c2': str, 'c3': np.int64},
        env.bob: {'c4': str, 'c5': np.int64},
    }
    df = read_csv(filepath, dtypes=dtypes)

    pd.testing.assert_frame_equal(
        reveal(df.partitions[env.alice].data).reset_index(drop=True), df1
    )
    pd.testing.assert_frame_equal(
        reveal(df.partitions[env.bob].data).reset_index(drop=True), df2
    )

    cleartmp([path1, path2])


def test_read_csv_without_psi_mismatch_length(prod_env_and_data):
    env, data = prod_env_and_data
    df1 = pd.DataFrame(
        {'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'], 'c3': [5, 1, 2, 6, 4, 3]}
    )

    df2 = pd.DataFrame({'c4': ['B3', 'B1', 'B9', 'B4'], 'c5': [3, 1, 9, 4]})

    _, path1 = tempfile.mkstemp()
    _, path2 = tempfile.mkstemp()

    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    filepath = {env.alice: path1, env.bob: path2}
    dtypes = {
        env.alice: {'c2': str, 'c3': np.int64},
        env.bob: {'c4': str, 'c5': np.int64},
    }
    with pytest.raises(AssertionError, match='number of samples must be equal'):
        read_csv(filepath, dtypes=dtypes)

    cleartmp([path1, path2])


def test_to_csv_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    _, path1 = tempfile.mkstemp()
    _, path2 = tempfile.mkstemp()
    file_uris = {env.alice: path1, env.bob: path2}
    df1 = pd.DataFrame({'c2': ['A5', 'A1', 'A2', 'A6'], 'c3': [5, 1, 2, 6]})

    df2 = pd.DataFrame({'c4': ['B3', 'B1', 'B9', 'B4'], 'c5': [3, 1, 9, 4]})

    df = VDataFrame(
        {
            env.alice: Partition(env.alice(lambda df: df)(df1)),
            env.bob: Partition(env.bob(lambda df: df)(df2)),
        }
    )

    # WHEN
    to_csv(df, file_uris, index=False)

    # THEN
    # Waiting a while for to_csv finish.
    import time

    time.sleep(5)
    actual_df = read_csv(file_uris)
    pd.testing.assert_frame_equal(reveal(actual_df.partitions[env.alice].data), df1)
    pd.testing.assert_frame_equal(reveal(actual_df.partitions[env.bob].data), df2)
    cleartmp([path1, path2])
