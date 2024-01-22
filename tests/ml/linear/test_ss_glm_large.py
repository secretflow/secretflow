import logging
import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import SPU, reveal, wait
from secretflow.ml.linear.ss_glm import SSGLM
from secretflow.ml.linear.ss_glm.core import get_dist


def load_mtpl2(n_samples=None):
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
    df = load_mtpl2(20000)

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


def run_sklearn(test_name, X_train, df_train, power, alpha, dist):
    glm_pure_premium = TweedieRegressor(
        power=power, alpha=alpha, solver='newton-cholesky'
    )
    y = df_train["PurePremium"].values
    test_name = test_name + "_sklearn"
    start = time.time()
    glm_pure_premium.fit(
        X_train, df_train["PurePremium"], sample_weight=df_train["Exposure"]
    )
    logging.info(f"{test_name} train time: {time.time() - start}")
    start = time.time()
    yhat = glm_pure_premium.predict(X_train)
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    logging.info(f"{test_name} predict time: {time.time() - start}")
    d2 = eval(yhat, y, dist, power, df_train["Exposure"].values)
    logging.info(f"{test_name} deviance: {d2}")
    return d2


def run_irls(
    devices,
    test_name,
    X,
    df,
    link,
    dist,
    l2_lambda=None,
    ss_glm_power=1.9,
    n_iter=4,
):
    # overwrite SPU because we must have FM128 and fxp 40
    cluster_def = devices.spu.cluster_def
    cluster_def["runtime_config"].update({"field": "FM128", "fxp_fraction_bits": 40})
    spu = SPU(cluster_def, link_desc=devices.spu.link_desc)
    devices.spu = spu
    model = SSGLM(spu)
    v_data, label_data, w = dataset_to_federated(
        X, df["PurePremium"], df["Exposure"], devices
    )
    y = df["PurePremium"].values
    test_name = test_name + "_irls"

    start = time.time()
    model.fit_irls(
        v_data,
        label_data,
        None,
        w,
        n_iter,
        link,
        dist,
        ss_glm_power,
        1,
        l2_lambda=l2_lambda,
    )
    logging.info(f"{test_name} train time: {time.time() - start}")
    start = time.time()
    spu_yhat = model.predict(v_data)
    yhat = reveal(spu_yhat).reshape((-1,))
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    logging.info(f"{test_name} predict time: {time.time() - start}")
    d2 = eval(yhat, y, dist, ss_glm_power, df["Exposure"].values)
    logging.info(f"{test_name} deviance: {d2}")
    return d2


def eval(yhat, y, dist, power, w):
    deviance = get_dist(dist, 1, power).deviance(yhat, y.reshape(-1), w)
    assert not np.isnan(deviance), f"{yhat}, {y}, {w}"
    y_mean = np.mean(y) + np.zeros_like(y)
    null_deviance = get_dist(dist, 1, power).deviance(y_mean, y.reshape(-1), w)
    d2 = 1 - deviance / null_deviance

    return d2


def _run_test(devices, test_name, X, df, link, dist, l2_lambda=None, power=1.9):
    sklearn_deviance = run_sklearn(test_name, X, df, power, l2_lambda, dist)
    irls_deviance = run_irls(
        devices=devices,
        test_name=test_name,
        X=X,
        df=df,
        link=link,
        dist=dist,
        l2_lambda=l2_lambda,
        ss_glm_power=power,
        n_iter=4,
    )
    wait(irls_deviance)
    logging.info(irls_deviance)
    assert (
        abs(irls_deviance - sklearn_deviance) <= 0.05
    ), f"{irls_deviance}, {sklearn_deviance}"


def test_mtpl2(sf_production_setup_devices_aby3):
    X, df = prepare_data()
    _run_test(
        devices=sf_production_setup_devices_aby3,
        test_name="mtpl2",
        X=X,
        df=df,
        link='Log',
        dist='Tweedie',
        l2_lambda=0.1,
        power=1.9,
    )
