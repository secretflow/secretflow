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

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics

from secretflow import reveal
from secretflow.data import partition
from secretflow.data.ndarray import (
    histogram,
    load,
    mean_abs_err,
    mean_abs_percent_err,
    r2_score,
    rss,
    shuffle,
    subtract,
    tss,
)
from secretflow.data.split import train_test_split
from secretflow.data.vertical import VDataFrame
from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.simulation.data.ndarray import create_ndarray
from tests.sf_fixtures import mpc_fixture


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    # Ignore nan.
    return ((a == b) | ((a != a) & (b != b))).all()


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob

    def gen_arr():
        _, path = tempfile.mkstemp()
        arr = np.random.rand(20, 10)
        np.save(path, arr, allow_pickle=False)
        return path, arr

    alice_path, alice_arr = reveal(pyu_alice(gen_arr, num_returns=2)())
    bob_path, bob_arr = reveal(pyu_bob(gen_arr, num_returns=2)())
    path = {
        pyu_alice: f"{alice_path}.npy",
        pyu_bob: f"{bob_path}.npy",
    }

    y_true = reveal(pyu_alice(lambda: np.random.rand(100, 100))())
    y_pred = reveal(pyu_alice(lambda arr: arr + np.random.rand(100, 100) / 20)(y_true))
    y_true_fed_h = create_ndarray(
        y_true,
        {pyu_alice: 0.3, pyu_bob: 0.7},
        axis=0,
    )
    y_pred_fed_h = create_ndarray(
        y_pred,
        {pyu_alice: 0.3, pyu_bob: 0.7},
        axis=0,
    )
    y_true_fed_v = create_ndarray(y_true, {pyu_bob: 1.0}, axis=1)
    y_pred_fed_v = create_ndarray(y_pred, {pyu_bob: 1.0}, axis=1)

    yield sf_production_setup_devices, {
        "alice_path": alice_path,
        "alice_arr": alice_arr,
        "bob_path": bob_path,
        "bob_arr": bob_arr,
        "path": path,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_fed_h": y_true_fed_h,
        "y_pred_fed_h": y_pred_fed_h,
        "y_true_fed_v": y_true_fed_v,
        "y_pred_fed_v": y_pred_fed_v,
    }

    try:
        for filepath in path.values():
            os.remove(filepath)
    except OSError:
        pass


@pytest.mark.mpc
def test_load_file_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    fed_arr = load(data["path"], allow_pickle=True)
    # THEN
    assert array_equal(reveal(fed_arr.partitions[env.alice]), data["alice_arr"])
    assert array_equal(reveal(fed_arr.partitions[env.bob]), data["bob_arr"])


@pytest.mark.mpc
def test_load_pyu_object_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    alice_arr = env.alice(lambda: np.array([[1, 2, 3], [4, 5, 6]]))()
    bob_arr = env.bob(lambda: np.array([[11, 12, 13], [14, 15, 16]]))()

    # WHEN
    fed_arr = load({env.alice: alice_arr, env.bob: bob_arr})

    # THEN
    assert fed_arr.partitions[env.alice] == alice_arr
    assert fed_arr.partitions[env.bob] == bob_arr


@pytest.mark.mpc
def test_load_should_error_with_wrong_pyu_object(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    alice_arr = env.alice(lambda: np.array([[1, 2, 3], [4, 5, 6]]))()
    bob_arr = env.bob(lambda: np.array([[11, 12, 13], [14, 15, 16]]))()

    # WHEN & THEN
    with pytest.raises(
        InvalidArgumentError, match="Device of source differs with its key."
    ):
        load({env.alice: bob_arr, env.bob: alice_arr})


@pytest.mark.mpc
def test_train_test_split_on_hdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    fed_arr = load(data["path"], allow_pickle=True)

    # WHEN
    fed_arr0, fed_arr1 = train_test_split(fed_arr, train_size=0.6, shuffle=False)

    # THEN
    assert array_equal(
        np.concatenate(
            [
                reveal(fed_arr0.partitions[env.alice]),
                reveal(fed_arr1.partitions[env.alice]),
            ],
            axis=0,
        ),
        data["alice_arr"],
    )

    assert array_equal(
        np.concatenate(
            [
                reveal(fed_arr0.partitions[env.bob]),
                reveal(fed_arr1.partitions[env.bob]),
            ],
            axis=0,
        ),
        data["bob_arr"],
    )


@pytest.mark.mpc
def test_train_test_split_on_vdataframe_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    df_alice = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a1": ["K5", "K1", None, "K6"],
            "a2": ["A5", "A1", "A2", "A6"],
            "a3": [5, 1, 2, 6],
        }
    )

    df_bob = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "b4": [10.2, 20.5, None, -0.4],
            "b5": ["B3", None, "B9", "B4"],
            "b6": [3, 1, 9, 4],
        }
    )
    df = VDataFrame(
        {
            env.alice: partition(data=env.alice(lambda: df_alice)()),
            env.bob: partition(data=env.bob(lambda: df_bob)()),
        }
    )
    fed_arr = df.values

    # WHEN
    fed_arr0, fed_arr1 = train_test_split(fed_arr, train_size=0.6, shuffle=False)

    # THEN
    np.testing.assert_equal(
        reveal(fed_arr0.partitions[env.alice])[:, 0],
        reveal(fed_arr0.partitions[env.bob])[:, 0],
    )
    np.testing.assert_equal(
        reveal(fed_arr1.partitions[env.alice])[:, 0],
        reveal(fed_arr1.partitions[env.bob])[:, 0],
    )


@pytest.mark.mpc
def test_shuffle_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    df_alice = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a1": ["K5", "K1", None, "K6"],
            "a2": ["A5", "A1", "A2", "A6"],
            "a3": [5, 1, 2, 6],
        }
    )

    df_bob = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "b4": [10.2, 20.5, None, -0.4],
            "b5": ["B3", None, "B9", "B4"],
            "b6": [3, 1, 9, 4],
        }
    )
    df = VDataFrame(
        {
            env.alice: partition(data=env.alice(lambda: df_alice)()),
            env.bob: partition(data=env.bob(lambda: df_bob)()),
        }
    )
    fed_arr = df.values

    # WHEN
    shuffle(fed_arr)

    # THEN
    np.testing.assert_equal(
        reveal(fed_arr.partitions[env.alice])[:, 0],
        reveal(fed_arr.partitions[env.bob])[:, 0],
    )


@pytest.mark.mpc
def test_load_npz(prod_env_and_data):
    env, data = prod_env_and_data

    # GIVEN
    def gen_arr():
        _, path = tempfile.mkstemp()
        train = np.random.rand(20, 10)
        test = np.random.rand(10, 1)
        np.savez(path, train=train, test=test)
        return path, train, test

    alice_path, alice_train, alice_test = reveal(env.alice(gen_arr)())
    bob_path, bob_train, bob_test = reveal(env.alice(gen_arr)())
    # _, bob_path = tempfile.mkstemp()
    # alice_train = np.random.rand(20, 10)
    # alice_test = np.random.rand(20, 1)
    # bob_train = np.random.rand(10, 10)
    # bob_test = np.random.rand(10, 1)
    # np.savez(alice_path, train=alice_train, test=alice_test)
    # np.savez(bob_path, train=bob_train, test=bob_test)
    # WHEN
    data = load(
        {env.alice: f"{alice_path}.npz", env.bob: f"{bob_path}.npz"},
        allow_pickle=True,
    )

    # THEN
    for k, v in data.items():
        if k == "train":
            assert array_equal(reveal(v.partitions[env.alice]), alice_train)
            assert array_equal(reveal(v.partitions[env.bob]), bob_train)
        else:
            assert array_equal(reveal(v.partitions[env.alice]), alice_test)
            assert array_equal(reveal(v.partitions[env.bob]), bob_test)


@pytest.mark.mpc
def test_astype_should_ok(prod_env_and_data):
    env, data = prod_env_and_data
    # WHEN
    fed_arr = load(data["path"])
    fed_arr = fed_arr.astype(str)

    # THEN
    assert array_equal(
        reveal(fed_arr.partitions[env.alice]), data["alice_arr"].astype(str)
    )

    assert array_equal(reveal(fed_arr.partitions[env.bob]), data["bob_arr"].astype(str))


def operator_h_v_cases_test(prod_env_and_data, test_handle, true_val, binary=True):
    env, data = prod_env_and_data
    if not binary:
        a_h = reveal(test_handle(data["y_true_fed_h"], spu_device=env.spu))
        a_v = reveal(test_handle(data["y_true_fed_v"]))
    else:
        a_h = reveal(test_handle(data["y_true_fed_h"], data["y_pred_fed_h"], env.spu))
        a_v = reveal(test_handle(data["y_true_fed_v"], data["y_pred_fed_v"]))
        # Currently mixed case is not supported
    np.testing.assert_almost_equal(true_val, a_h, decimal=2)
    np.testing.assert_almost_equal(true_val, a_v, decimal=2)


@pytest.mark.mpc
def test_tss(prod_env_and_data):
    env, data = prod_env_and_data
    tss_val = np.sum(np.square(data["y_true"] - np.mean(data["y_true"])))
    operator_h_v_cases_test(prod_env_and_data, tss, tss_val, False)


@pytest.mark.mpc
def test_rss(prod_env_and_data):
    env, data = prod_env_and_data
    rss_val = np.sum(np.square(data["y_true"] - data["y_pred"]))
    operator_h_v_cases_test(prod_env_and_data, rss, rss_val)


@pytest.mark.mpc
def test_r2_score(prod_env_and_data):
    env, data = prod_env_and_data
    r2score_val = sklearn.metrics.r2_score(data["y_true"], data["y_pred"])
    operator_h_v_cases_test(prod_env_and_data, r2_score, r2score_val)


@pytest.mark.mpc
def test_mean_abs_err(prod_env_and_data):
    env, data = prod_env_and_data
    mae_val = sklearn.metrics.mean_absolute_error(data["y_true"], data["y_pred"])
    operator_h_v_cases_test(prod_env_and_data, mean_abs_err, mae_val)


@pytest.mark.mpc
def test_mean_abs_percent_err(prod_env_and_data):
    env, data = prod_env_and_data
    mape_val = sklearn.metrics.mean_absolute_percentage_error(
        data["y_true"], data["y_pred"]
    )
    operator_h_v_cases_test(prod_env_and_data, mean_abs_percent_err, mape_val)


@pytest.mark.mpc
def test_subtraction(prod_env_and_data):
    env, data = prod_env_and_data
    residual_val = data["y_true"] - data["y_pred"]
    operator_h_v_cases_test(prod_env_and_data, subtract, residual_val)


@pytest.mark.mpc
def test_histogram(prod_env_and_data):
    env, data = prod_env_and_data
    hist, edges = np.histogram(data["y_true"])
    h_v, e_v = reveal(histogram(data["y_true_fed_v"]))
    np.testing.assert_almost_equal(hist, reveal(h_v), decimal=2)
    np.testing.assert_almost_equal(edges, reveal(e_v), decimal=2)
    # TODO(zoupeicheng.zpc): pending on spu support for the following.
    # h_h, e_h = reveal(histogram(data['y_true_fed_h'], spu_device = self.spu))
    # np.testing.assert_almost_equal(hist, reveal(h_h), decimal=2)
    # np.testing.assert_almost_equal(edges, reveal(e_h), decimal=2)


# @unittest.skip('Not stable @jiuqi')
# def test_residual_histogram(prod_env_and_data):
#     env, data = prod_env_and_data
#     hist, edges = np.histogram(data['y_true'] - data['y_pred'])
#     h_v, e_v = reveal(residual_histogram(data['y_true_fed_v'], data['y_pred_fed_v']))
#     h_v_direct, e_v_direct = reveal(histogram(data['y_true_fed_v'] - data['y_pred_fed_v']))
#     np.testing.assert_almost_equal(hist, reveal(h_v), decimal=2)
#     np.testing.assert_almost_equal(edges, reveal(e_v), decimal=2)
#     np.testing.assert_almost_equal(hist, reveal(h_v_direct), decimal=2)
#     np.testing.assert_almost_equal(edges, reveal(e_v_direct), decimal=2)
