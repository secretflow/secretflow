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
import json
import logging
import os
import time

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, roc_auc_score

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.boost.core.data_preprocess import prepare_dataset
from secretflow.ml.boost.sgb_v import Sgb
from secretflow.ml.boost.sgb_v.complete_model import from_dict, from_sgb_model
from secretflow.ml.boost.sgb_v.complete_tree import from_distributed_tree
from secretflow.ml.boost.sgb_v.core.distributed_tree.distributed_tree import (
    DistributedTree,
)
from secretflow.ml.boost.sgb_v.model import load_model
from tests.sf_fixtures import SFProdParams

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _gen_sgb_model_complete_model_and_x(devices):
    from sklearn.datasets import load_breast_cancer

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

    model = _run_sgb(
        devices,
        "breast_cancer",
        v_data,
        label_data,
        y,
        True,
        1,
        0.9,
    )
    complete_model_object = from_sgb_model(model, devices.alice)
    return model, complete_model_object, x


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
    auc_bar=0.9,
    mse_hat=1,
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
    model.save_model(saving_path_dict)
    model_loaded = load_model(saving_path_dict, env.alice)
    fed_yhat_loaded = model_loaded.predict(v_data, env.alice)
    yhat_loaded = reveal(fed_yhat_loaded.partitions[env.alice])

    assert (
        yhat == yhat_loaded
    ).all(), "loaded model predictions should match original, yhat {} vs yhat_loaded {}".format(
        yhat, yhat_loaded
    )
    return model


@pytest.mark.parametrize("tree_idx", [0, 1, 2])
@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_split_tree(sf_production_setup_devices, tree_idx):
    logging.info("start testing split tree")
    devices = sf_production_setup_devices
    model, _, x = _gen_sgb_model_complete_model_and_x(devices)
    v_data = FedNdarray(
        {
            devices.alice: (devices.alice(lambda: x[:, :15])()),
            devices.bob: (devices.bob(lambda: x[:, 15:])()),
        },
        partition_way=PartitionWay.VERTICAL,
    )

    tree: DistributedTree = model.trees[tree_idx]
    x_fed, _ = prepare_dataset(v_data)
    x_fed = v_data.partitions

    complete_tree_object = from_distributed_tree(devices.bob, tree)

    complete_tree_predict = reveal(
        devices.bob(lambda tree, x: tree.predict(x))(complete_tree_object, x)
    )
    true_predict = reveal(tree.predict(x_fed))
    np.testing.assert_array_almost_equal(true_predict, complete_tree_predict, decimal=3)


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_complete_model(sf_production_setup_devices):
    logging.info("start testing complete model")
    devices = sf_production_setup_devices
    model, complete_model_object, x = _gen_sgb_model_complete_model_and_x(devices)
    v_data = FedNdarray(
        {
            devices.alice: (devices.alice(lambda: x[:, :15])()),
            devices.bob: (devices.bob(lambda: x[:, 15:])()),
        },
        partition_way=PartitionWay.VERTICAL,
    )

    true_predict = reveal(model.predict(v_data))

    assert model.base == reveal(
        devices.alice(lambda model: model.base)(complete_model_object)
    )

    complete_model_predict = reveal(
        devices.alice(lambda model, x: model.predict(x))(complete_model_object, x)
    )
    np.testing.assert_array_almost_equal(
        true_predict, complete_model_predict, decimal=3
    )


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_complete_model_serde(sf_production_setup_devices):
    logging.info("start testing complete model serde test")
    devices = sf_production_setup_devices
    _, complete_model_object, x = _gen_sgb_model_complete_model_and_x(devices)

    complete_model_predict = reveal(
        devices.alice(lambda model, x: model.predict(x))(complete_model_object, x)
    )

    loaded_obj = devices.alice(
        lambda x: from_dict(json.loads(json.dumps(x.to_dict())))
    )(complete_model_object)

    loaded_complete_model_predict = reveal(
        devices.alice(lambda model, x: model.predict(x))(loaded_obj, x)
    )
    np.testing.assert_array_almost_equal(
        loaded_complete_model_predict, complete_model_predict, decimal=8
    )
