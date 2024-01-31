# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time

from sklearn.metrics import mean_squared_error, roc_auc_score

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.boost.sgb_v import Sgb
from secretflow.ml.boost.sgb_v.model import load_model
from secretflow.utils.simulation.datasets import load_dermatology, load_linear

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _run_sgb(
    env,
    test_name,
    v_data,
    label_data,
    y,
    logistic,
    subsample,
    colsample,
    audit_dict={},
    auc_bar=0.88,
    mse_hat=1.1,
    tree_grow_method='level',
    enable_goss=False,
    early_stop_criterion_g_abs_sum=10.0,
    num_boost_round=2,
    num_tree_cap=2,
):
    test_name = test_name + "_with_method_" + tree_grow_method
    sgb = Sgb(env.heu)
    start = time.perf_counter()
    params = {
        'tree_growing_method': tree_grow_method,
        # for first_tree_with_label_holder_feature is True
        'num_boost_round': num_boost_round + 1,
        'max_depth': 3,
        'max_leaf': 2**3,
        'sketch_eps': 0.25,
        'objective': 'logistic' if logistic else 'linear',
        'reg_lambda': 0.1,
        'gamma': 1,
        'rowsample_by_tree': subsample,
        'colsample_by_tree': colsample,
        'base_score': 0.5,
        'audit_paths': audit_dict,
        'seed': 42,
        'fixed_point_parameter': 20,
        # Turn these two options on for benchmarking or debugging.
        # Verbose mode will produce more logging information.
        'verbose': False,
        # Wait execution mode will syncronize operations and therefore reduce performance, but we now can measure component's time more accurately.
        # Wait execution mode execution time is expected to be slower than that when use in production.
        'wait_execution': False,
        'first_tree_with_label_holder_feature': True,
        'enable_goss': enable_goss,
        'enable_quantization': True,  # surprisingly, quantization may also improve auc on some datasets
        'early_stop_criterion_g_abs_sum': early_stop_criterion_g_abs_sum,
        'early_stop_criterion_g_abs_sum_change_ratio': 0.01,
        'enable_packbits': False,
    }
    model = sgb.train(params, v_data, label_data)
    reveal(model.trees[-1])
    logging.info(f"{test_name} train time: {time.perf_counter() - start}")
    start = time.perf_counter()
    yhat = model.predict(v_data)
    yhat = reveal(yhat)
    logging.info(f"{test_name} predict time: {time.perf_counter() - start}")
    if logistic:
        auc = roc_auc_score(y, yhat)
        logging.info(f"{test_name} auc: {auc}")
        assert auc > auc_bar
    else:
        mse = mean_squared_error(y, yhat)
        logging.info(f"{test_name} mse: {mse}")
        assert mse < mse_hat

    if num_tree_cap < num_boost_round:
        logging.info(
            f"current tree number is {len(model.trees)}, cap is {num_tree_cap}"
        )
        assert len(model.trees) <= num_tree_cap

    fed_yhat = model.predict(v_data, env.alice)
    assert len(fed_yhat.partitions) == 1 and env.alice in fed_yhat.partitions
    yhat = reveal(fed_yhat.partitions[env.alice])
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    if logistic:
        logging.info(f"{test_name} auc: {roc_auc_score(y, yhat)}")
    else:
        logging.info(f"{test_name} mse: {mean_squared_error(y, yhat)}")

    saving_path_dict = {
        device: "./" + test_name + "/" + device.party
        for device in v_data.partitions.keys()
    }
    label_holder_device = list(label_data.partitions.keys())[0]
    if label_holder_device not in saving_path_dict:
        saving_path_dict[label_holder_device] = (
            "./" + test_name + "/" + label_holder_device.party
        )
    model.save_model(saving_path_dict)
    model_loaded = load_model(saving_path_dict, env.alice)
    fed_yhat_loaded = model_loaded.predict(v_data, env.alice)
    yhat_loaded = reveal(fed_yhat_loaded.partitions[env.alice])

    assert (
        yhat == yhat_loaded
    ).all(), "loaded model predictions should match original, yhat {} vs yhat_loaded {}".format(
        yhat, yhat_loaded
    )


def _run_npc_linear(env, test_name, parts, label_device, auc=0.88):
    vdf = load_linear(parts=parts)

    label_data = vdf['y']
    y = reveal(label_data.partitions[label_device].data).values
    label_data = (label_data.values)[:500, :]
    y = y[:500, :]

    v_data = vdf.drop(columns="y").values
    v_data = v_data[:500, :]
    label_data = label_data[:500, :]

    logging.info("running XGB style test")
    _run_sgb(env, test_name, v_data, label_data, y, True, 0.9, 1, auc_bar=auc)
    logging.info("running lightGBM style test")
    # test with leaf wise growth and goss: lightGBM style
    _run_sgb(
        env,
        test_name,
        v_data,
        label_data,
        y,
        True,
        0.9,
        1,
        {},
        auc,
        2.3,
        'leaf',
        True,
    )


def test_2pc_linear(sf_production_setup_devices_aby3):
    parts = {
        sf_production_setup_devices_aby3.bob: (1, 11),
        sf_production_setup_devices_aby3.alice: (11, 22),
    }
    _run_npc_linear(
        sf_production_setup_devices_aby3,
        "2pc_linear",
        parts,
        sf_production_setup_devices_aby3.alice,
    )


def test_4pc_linear(sf_production_setup_devices_aby3):
    parts = {
        sf_production_setup_devices_aby3.davy: (1, 6),
        sf_production_setup_devices_aby3.bob: (6, 12),
        sf_production_setup_devices_aby3.carol: (12, 18),
        sf_production_setup_devices_aby3.alice: (18, 22),
    }
    _run_npc_linear(
        sf_production_setup_devices_aby3,
        "4pc_linear",
        parts,
        sf_production_setup_devices_aby3.alice,
    )


def test_2pc_linear_minimal(sf_production_setup_devices_aby3):
    parts = {
        sf_production_setup_devices_aby3.davy: (1, 2),
        sf_production_setup_devices_aby3.alice: (21, 22),
    }
    _run_npc_linear(
        sf_production_setup_devices_aby3,
        "2pc_linear_minimal",
        parts,
        sf_production_setup_devices_aby3.alice,
        auc=0.55,
    )


def test_breast_cancer(sf_production_setup_devices_aby3):
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']

    v_data = FedNdarray(
        {
            sf_production_setup_devices_aby3.alice: (
                sf_production_setup_devices_aby3.alice(lambda: x[:, :15])()
            ),
            sf_production_setup_devices_aby3.bob: (
                sf_production_setup_devices_aby3.bob(lambda: x[:, 15:])()
            ),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label_data = FedNdarray(
        {
            sf_production_setup_devices_aby3.alice: (
                sf_production_setup_devices_aby3.alice(lambda: y)()
            )
        },
        partition_way=PartitionWay.VERTICAL,
    )

    _run_sgb(
        sf_production_setup_devices_aby3,
        "breast_cancer",
        v_data,
        label_data,
        y,
        True,
        1,
        0.9,
        num_boost_round=0,
        auc_bar=0.5,
    )

    _run_sgb(
        sf_production_setup_devices_aby3,
        "breast_cancer",
        v_data,
        label_data,
        y,
        True,
        1,
        0.9,
    )

    # test with leaf wise growth
    _run_sgb(
        sf_production_setup_devices_aby3,
        "breast_cancer_early_stop",
        v_data,
        label_data,
        y,
        True,
        0.9,
        1,
        {},
        0.9,
        2.3,
        'leaf',
        early_stop_criterion_g_abs_sum=100,
        num_boost_round=10,
        num_tree_cap=3,
    )


def test_dermatology(sf_production_setup_devices_aby3):
    vdf = (
        load_dermatology(
            parts={
                sf_production_setup_devices_aby3.bob: (0, 17),
                sf_production_setup_devices_aby3.alice: (17, 35),
            },
            axis=1,
        )
        .fillna(0)
        .replace({None: 0})
    )
    label_data = vdf['class']
    y = reveal(
        label_data.partitions[sf_production_setup_devices_aby3.alice].data
    ).values
    v_data = vdf.drop(columns="class").values
    label_data = label_data.values

    audit_dict = {
        sf_production_setup_devices_aby3.alice.party: "./audit_alice",
        sf_production_setup_devices_aby3.bob.party: "./audit_bob",
    }
    _run_sgb(
        sf_production_setup_devices_aby3,
        "dermatology",
        v_data,
        label_data,
        y,
        False,
        0.9,
        0.9,
        audit_dict,
    )

    _run_sgb(
        sf_production_setup_devices_aby3,
        "dermatology",
        v_data,
        label_data,
        y,
        False,
        0.9,
        0.9,
        audit_dict,
        0.9,
        1,
        'leaf',
        True,
    )
