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

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, roc_auc_score

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.boost.sgb_v import Sgb
from secretflow.ml.boost.sgb_v.model import load_model
from secretflow.utils.simulation.datasets import load_dermatology, load_linear
from tests.sf_fixtures import SFProdParams

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
    auc_bar=0.87,
    mse_hat=1.1,
    tree_grow_method='level',
    enable_goss=False,
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
        'enable_packbits': False,
        'eval_metric': 'roc_auc' if logistic else 'mse',
        'enable_monitor': True,
        'enable_early_stop': True,
        'validation_fraction': 0.1,
        'stopping_rounds': 1,
        'stopping_tolerance': 0.01,
        'save_best_model': False,
    }
    sample_weight = np.ones(y.shape)
    sample_weight_v = FedNdarray(
        partitions={
            device: device(lambda x: x)(sample_weight)
            for device in label_data.partitions.keys()
        },
        partition_way=PartitionWay.VERTICAL,
    )
    model = sgb.train(
        params,
        v_data,
        label_data,
        sample_weight=sample_weight_v,
    )
    reveal(model.trees[-1])
    logging.info(f"{test_name} train time: {time.perf_counter() - start}")
    start = time.perf_counter()
    yhat = model.predict(v_data)
    yhat = reveal(yhat)
    logging.info(f"{test_name} predict time: {time.perf_counter() - start}")
    if logistic:
        auc = roc_auc_score(y, yhat)
        logging.info(f"{test_name} auc: {auc}")
        # the 0.01 error raises from fixed point encoding when caculating cleartext gh
        assert auc > auc_bar - 0.01
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
    # enabled early stop, our params is num_boost_round + 1
    # should not train e.g. 10 epochs
    assert len(model.get_trees()) <= num_boost_round + 1


def _run_npc_linear(env, test_name, parts, label_device, auc=0.87):
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


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_2pc_linear(sf_production_setup_devices):
    devices = sf_production_setup_devices

    parts = {
        devices.bob: (1, 11),
        devices.alice: (11, 22),
    }
    _run_npc_linear(devices, "2pc_linear", parts, devices.alice)


@pytest.mark.mpc(parties=4, params=SFProdParams.ABY3)
def test_4pc_linear(sf_production_setup_devices):
    devices = sf_production_setup_devices

    parts = {
        devices.davy: (1, 6),
        devices.bob: (6, 12),
        devices.carol: (12, 18),
        devices.alice: (18, 22),
    }
    _run_npc_linear(sf_production_setup_devices, "4pc_linear", parts, devices.alice)


@pytest.mark.mpc(parties=4, params=SFProdParams.ABY3)
def test_2pc_linear_minimal(sf_production_setup_devices):
    devices = sf_production_setup_devices

    parts = {devices.davy: (1, 2), devices.alice: (21, 22)}
    _run_npc_linear(
        devices,
        "2pc_linear_minimal",
        parts,
        devices.alice,
        auc=0.52,
    )


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_breast_cancer(sf_production_setup_devices):
    from sklearn.datasets import load_breast_cancer

    devices = sf_production_setup_devices

    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']

    v_data = FedNdarray(
        {
            devices.alice: (devices.alice(lambda: x[:, :15])()),
            devices.bob: (devices.bob(lambda: x[:, 15:])()),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label_data = FedNdarray(
        {devices.alice: (devices.alice(lambda: y)())},
        partition_way=PartitionWay.VERTICAL,
    )

    _run_sgb(
        sf_production_setup_devices,
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
        sf_production_setup_devices,
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
        devices,
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
        num_boost_round=10,
        num_tree_cap=3,
    )


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_dermatology(sf_production_setup_devices):
    devices = sf_production_setup_devices

    vdf = (
        load_dermatology(
            parts={devices.bob: (0, 17), devices.alice: (17, 35)},
            axis=1,
        )
        .fillna(0)
        .replace({None: 0})
    )
    label_data = vdf['class']
    y = reveal(label_data.partitions[devices.alice].data).values
    v_data = vdf.drop(columns="class").values
    label_data = label_data.values

    audit_dict = {
        devices.alice.party: "./audit_alice",
        devices.bob.party: "./audit_bob",
    }
    _run_sgb(
        devices,
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
        devices,
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
