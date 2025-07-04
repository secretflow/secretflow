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
import pytest
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.boost.sgb_v import Sgb
from secretflow.ml.boost.sgb_v.core.params import (
    RegType,
    apply_new_params,
    xgb_params_converter,
)
from secretflow.ml.linear.ss_glm.core import get_dist
from secretflow.ml.linear.ss_glm.metrics import deviance

from sklearn.datasets import fetch_openml
from secretflow.utils.simulation.datasets import load_linear
from tests.sf_fixtures import SFProdParams

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def balanced_sample_weight(y: np.ndarray):
    """
    Generate sample weights based on class imbalance.

    Parameters:
    - y: An array of ground truth labels (0 1 ... k).

    Returns:
    - weights: An array of weights for each sample in y.
    """
    # Counts of each class (0 and 1)
    n_samples = len(y)
    n_classes = np.unique(y)
    counts = np.array([len(y[y == i]) for i in n_classes])

    # Calculate the weight for each class
    # The weight for each class is inversely proportional to its frequency
    weights_for_classes = n_samples / (len(n_classes) * counts)

    # Map the class weights to each sample
    weights = np.array([weights_for_classes[i] for i in y])

    return weights


def _run_sgb_tweedie(env, test_name, v_data, label_data, y, subsample, colsample):
    test_name = test_name + "_with_method_" + 'level' + '_tweedie'
    sgb = Sgb(env.heu)
    start = time.perf_counter()
    tweedie_variance_power = 1.5
    xgb_params = {
        "tree_method": "hist",
        "n_estimators": 15,
        "max_depth": 5,
        "learning_rate": 0.3,
        "max_bin": 5,
        "base_score": 0.5,
        "eval_metric": f"tweedie-nloglik@{tweedie_variance_power}",
        "reg_lambda": 0.1,
        "min_child_weight": 0,
        "objective": "reg:tweedie",
        'reg_lambda': 0.1,
        'gamma': 1,
        'subsample': subsample,
        'colsample_bytree': colsample,
        'base_score': 0.5,
        "seed": 94,
        "tweedie_variance_power": tweedie_variance_power,
    }

    additional_params = {
        'tree_growing_method': 'level',
        'fixed_point_parameter': 20,
        # Turn these two options on for benchmarking or debugging.
        # Verbose mode will produce more logging information.
        'verbose': False,
        # Wait execution mode will syncronize operations and therefore reduce performance, but we now can measure component's time more accurately.
        # Wait execution mode execution time is expected to be slower than that when use in production.
        'eval_metric': 'tweedie_nll',
        'enable_monitor': True,
    }
    params = apply_new_params(xgb_params_converter(xgb_params), additional_params)
    # we use the balanced approach to set the sample weight
    assert params['objective'] == 'tweedie'

    model = sgb.train(params, v_data, label_data)
    reveal(model.trees[-1])
    logging.info(f"{test_name} train time: {time.perf_counter() - start}")
    start = time.perf_counter()
    yhat = model.predict(v_data)
    assert model.get_objective() == RegType.Tweedie
    yhat = reveal(yhat)
    logging.info(f"{test_name} predict time: {time.perf_counter() - start}")

    clf = xgb.XGBRegressor(
        **xgb_params,
    )
    X = np.concatenate(
        [reveal(partition_data) for partition_data in v_data.partitions.values()],
        axis=1,
    )
    clf.fit(X, y)
    yhat_xgb = clf.predict(X)
    dist = 'Tweedie'

    sgb_deviance = eval(
        yhat.reshape(
            -1,
        ),
        y.reshape(
            -1,
        ),
        dist,
        tweedie_variance_power,
    )
    xgboost_deviance = eval(
        yhat_xgb.reshape(
            -1,
        ),
        y.reshape(
            -1,
        ),
        dist,
        tweedie_variance_power,
    )
    assert (
        abs(sgb_deviance - xgboost_deviance) <= 0.1
    ), f"{sgb_deviance}, {xgboost_deviance}"


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
    # we use the balanced approach to set the sample weight
    sample_weight = balanced_sample_weight(y)
    sample_weight_v = FedNdarray(
        partitions={
            device: device(lambda x: x)(sample_weight)
            for device in label_data.partitions.keys()
        },
        partition_way=PartitionWay.VERTICAL,
    )
    model = sgb.train(params, v_data, label_data, sample_weight=sample_weight_v)

    reveal(model.trees[-1])
    logging.info(f"{test_name} train time: {time.perf_counter() - start}")
    start = time.perf_counter()
    yhat = model.predict(v_data)
    yhat = reveal(yhat)
    logging.info(f"{test_name} predict time: {time.perf_counter() - start}")

    clf = xgb.XGBClassifier(
        **xgb_params, sample_weight=sample_weight, importance_type="weight"
    )
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

    feature_importance = model.feature_importance_flatten(v_data, "weight")
    feature_importance_gain = model.feature_importance_flatten(v_data, "gain")
    xgb_feature_importance = clf.feature_importances_

    assert (
        feature_importance.shape == xgb_feature_importance.shape
    ), f"feature importance shape mismatch, {feature_importance.shape} vs {xgb_feature_importance.shape}"

    # XGB and SGB use seemingly different way to calculate gains
    # cannot compare feature importance
    # yet SGB feature importance
    # should show relative importance among features as well

    logging.info(f"feature importance: {feature_importance}")
    logging.info(f"feature importance gain: {feature_importance_gain}")
    logging.info(f"xgb feature importance: {xgb_feature_importance}")


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

    _run_sgb_tweedie(env, test_name, v_data, label_data, y, 0.9, 1)


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_2pc_linear(sf_production_setup_devices):
    devices = sf_production_setup_devices
    parts = {devices.bob: (1, 11), devices.alice: (11, 22)}
    _run_npc_linear(devices, "2pc_linear", parts, devices.alice)


def load_mtpl2(n_samples=50000):
    """Fetch the French Motor Third-Party Liability Claims dataset.

    Parameters
    ----------
    n_samples: int, default=None
      number of samples to select (for faster run time). Full dataset has
      678013 samples.
    """
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = fetch_openml(data_id=41215, as_frame=True).data

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"].fillna(0, inplace=True)

    # unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")
    return df.iloc[:n_samples]


def prepare_data():
    # change this number to test on larger dataset
    df = load_mtpl2(2500)

    # Note: filter out claims with zero amount, as the severity model
    # requires strictly positive target values.
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    # Correct for unreasonable observations (that might be data error)
    # and a few exceptionally large claim amounts
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)

    log_scale_transformer = make_pipeline(
        FunctionTransformer(func=np.log), StandardScaler()
    )

    column_trans = ColumnTransformer(
        [
            (
                "binned_numeric",
                KBinsDiscretizer(n_bins=10, subsample=int(2e5), random_state=0),
                ["VehAge", "DrivAge"],
            ),
            (
                "onehot_categorical",
                OneHotEncoder(),
                ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
            ),
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        ],
        remainder="drop",
    )
    X = column_trans.fit_transform(df)

    # Insurances companies are interested in modeling the Pure Premium, that is
    # the expected total claim amount per unit of exposure for each policyholder
    # in their portfolio:
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]

    # This can be indirectly approximated by a 2-step modeling: the product of the
    # Frequency times the average claim amount per claim:
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)
    return X, df


def dataset_to_federated(x, y, w, env):
    def x_to_vdata(x):
        if not isinstance(x, np.ndarray):
            x = x.todense()
        v_data = FedNdarray(
            partitions={
                env.alice: env.alice(lambda: x[:, :15])(),
                env.bob: env.bob(lambda: x[:, 15:])(),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        return v_data

    v_data = x_to_vdata(x)

    label_data = FedNdarray(
        partitions={env.alice: env.alice(lambda: y.values)()},
        partition_way=PartitionWay.VERTICAL,
    )

    weight = FedNdarray(
        partitions={env.alice: env.alice(lambda: w.values)()},
        partition_way=PartitionWay.VERTICAL,
    )
    return v_data, label_data, weight


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_tweedie(sf_production_setup_devices):
    devices = sf_production_setup_devices
    X, df = prepare_data()
    v_data, label_data, w = dataset_to_federated(
        X, df["PurePremium"], df["Exposure"], devices
    )
    y = df["PurePremium"].values
    _run_sgb_tweedie(devices, "mtpl2", v_data, label_data, y, 0.9, 1)


def eval(yhat, y, dist, power, w=None):
    deviance_ = deviance(y, yhat, w, get_dist(dist, 1, power))
    assert not np.isnan(deviance_), f"{yhat}, {y}, {w}"
    y_mean = np.mean(y) + np.zeros_like(y)
    null_deviance = get_dist(dist, 1, power).deviance(y_mean, y.reshape(-1), w)
    d2 = 1 - deviance_ / null_deviance
    return d2
