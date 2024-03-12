# Copyright 2024 Ant Group Co., Ltd.
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
import xgboost as xgb

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.boost.sgb_v import Sgb
from secretflow.ml.boost.sgb_v.core.params import apply_new_params, xgb_params_converter
from secretflow.ml.boost.sgb_v.model import load_model
from secretflow.utils.simulation.datasets import load_dermatology, load_linear
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

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
):
    test_name = test_name + "_with_method_" + 'level'
    sgb = Sgb(env.heu)
    start = time.perf_counter()

    xgb_params = {
        "tree_method": "hist",
        "n_estimators": 10,
        "max_depth": 5,
        "learning_rate": 0.3,
        "max_bin": 5,
        "base_score": 0.5,
        "eval_metric": "auc",
        "reg_lambda": 0.1,
        "min_child_weight": 0,
        "objective": "binary:logistic" if logistic else "reg:squarederror",
        'reg_lambda': 0.1,
        'gamma': 1,
        'subsample': subsample,
        'colsample_bytree': colsample,
        'base_score': 0.5,
        "seed": 94,
    }

    additional_params = {
        'tree_growing_method': 'level',
        'audit_paths': audit_dict,
        'fixed_point_parameter': 20,
        # Turn these two options on for benchmarking or debugging.
        # Verbose mode will produce more logging information.
        'verbose': False,
        # Wait execution mode will syncronize operations and therefore reduce performance, but we now can measure component's time more accurately.
        # Wait execution mode execution time is expected to be slower than that when use in production.
        'eval_metric': 'roc_auc' if logistic else 'mse',
        'enable_monitor': True,
    }
    params = apply_new_params(xgb_params_converter(xgb_params), additional_params)
    model = sgb.train(params, v_data, label_data)
    reveal(model.trees[-1])
    logging.info(f"{test_name} train time: {time.perf_counter() - start}")
    start = time.perf_counter()
    yhat = model.predict(v_data)
    yhat = reveal(yhat)
    logging.info(f"{test_name} predict time: {time.perf_counter() - start}")

    clf = xgb.XGBClassifier(**xgb_params)
    X = np.concatenate(
        [reveal(partition_data) for partition_data in v_data.partitions.values()],
        axis=1,
    )
    clf.fit(X, y)

    if logistic:
        auc = roc_auc_score(y, yhat)
        logging.info(f"{test_name} sgb auc: {auc}")

        auc_xgb = roc_auc_score(y, clf.predict_proba(X)[:, 1])
        logging.info(f"{test_name} xgb auc: {auc_xgb}")
        assert abs(auc - auc_xgb) <= 0.005
    else:
        mse = mean_squared_error(y, yhat)
        logging.info(f"{test_name} mse: {mse}")

        mse_xgb = mean_squared_error(y, clf.predict(X))
        logging.info(f"{test_name} mse: {mse_xgb}")
        assert abs(mse - mse_xgb) <= 0.3


def _run_npc_linear(env, test_name, parts, label_device):
    vdf = load_linear(parts=parts)

    label_data = vdf['y']
    y = reveal(label_data.partitions[label_device].data).values
    label_data = (label_data.values)[:500, :]
    y = y[:500, :]

    v_data = vdf.drop(columns="y").values
    v_data = v_data[:500, :]
    label_data = label_data[:500, :]

    logging.info("running XGB style test")
    _run_sgb(env, test_name, v_data, label_data, y, True, 0.9, 1)


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
