# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stat
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from tweedie import tweedie

from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.ml.linear import RegType, SSRegression
from secretflow.ml.linear.ss_glm.core.distribution import DistributionType
from secretflow.ml.linear.ss_glm.core.link import LinkType
from secretflow.stats import SSPValue
from secretflow.utils.sigmoid import real_sig
from secretflow.utils.simulation.datasets import dataset
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyus = [
        sf_production_setup_devices.alice,
        sf_production_setup_devices.bob,
        sf_production_setup_devices.carol,
        sf_production_setup_devices.davy,
    ]
    pyus = [p for p in pyus if p]

    return sf_production_setup_devices, pyus


def _build_splited_ds(pyus, x, cols, parties):
    assert x.shape[1] >= parties
    assert len(pyus) >= parties, f"size mismatch,{pyus}, {parties}"
    step = math.ceil(x.shape[1] / parties)
    fed_x = VDataFrame({})
    for r in range(parties):
        start = r * step
        end = start + step if r != parties - 1 else x.shape[1]
        pyu_x = pyus[r](lambda x: pd.DataFrame(x))(x[:, start:end])
        fed_x.partitions[pyus[r]] = partition(data=pyu_x)
    if isinstance(cols, list):
        fed_cs = []
        for c in cols:
            pyu_c = pyus[parties - 1](lambda c: pd.DataFrame(c))(c)
            fed_c = VDataFrame({pyus[parties - 1]: partition(data=pyu_c)})
            fed_cs.append(fed_c)
        return fed_x, fed_cs
    else:
        pyu_c = pyus[parties - 1](lambda cols: pd.DataFrame(cols))(cols)
        fed_c = VDataFrame({pyus[parties - 1]: partition(data=pyu_c)})
        return fed_x, fed_c


def _run_ss(
    env,
    pyus,
    x,
    y,
    yhat,
    p,
    w,
    parties,
    model_type: Any,
    tweedie_p,
    y_scale,
    sample_weights,
):
    # weights to spu
    pyu_w = env.alice(lambda w: np.array(w))(w)
    spu_w = pyu_w.to(env.spu)
    pyu_yhat = env.alice(lambda yhat: np.array(yhat))(yhat)
    spu_yhat = pyu_yhat.to(env.spu)

    # x,y to pyu
    pyu_x, pyu_y = _build_splited_ds(pyus, x, y, parties)

    sspv = SSPValue(env.spu)
    if model_type == RegType.Linear:
        pvalues = sspv.t_statistic_p_value(pyu_x, pyu_y, spu_yhat, spu_w)
    else:
        if model_type == RegType.Logistic:
            link = LinkType.Logit
            model_type = DistributionType.Bernoulli
        elif model_type == DistributionType.Tweedie:
            link = LinkType.Log
        elif model_type == DistributionType.Gamma:
            link = LinkType.Reciprocal
        elif model_type == DistributionType.Poisson:
            link = LinkType.Log
        elif model_type == DistributionType.Bernoulli:
            link = LinkType.Logit
        if sample_weights is not None:
            sample_weights = env.alice(lambda w: np.array(w))(sample_weights)
            sample_weights_df = env.alice(
                lambda w: pd.DataFrame(w.reshape(-1), columns=['w'])
            )(sample_weights)
            sample_weights = VDataFrame({env.alice: partition(data=sample_weights_df)})

        pvalues = sspv.z_statistic_p_value(
            pyu_x,
            pyu_y,
            spu_yhat,
            spu_w,
            link,
            model_type,
            tweedie_power=tweedie_p,
            y_scale=y_scale,
            sample_weights=sample_weights,
        )

    p = np.array(p)
    abs_err = np.absolute(pvalues - p)
    radio_err = abs_err / np.maximum(pvalues, p)

    logging.warn(f"radio_err .... \n{radio_err}")

    # for pvalue < 0.2, check abs err < 0.01
    abs_assert = np.select([p < 0.2], [abs_err], 0)
    assert (
        np.amax(abs_assert) < 0.01
    ), f"\n{abs_assert}, \n our p value: {p}, sm pvalue: {pvalues}"
    # else check radio error < 20%
    radio_assert = np.select([p >= 0.2], [radio_err], 0)
    assert np.amax(radio_assert) < 0.2, f"\n{radio_err}"


def _run_test(
    env, pyus, x, orig_y, model_type: Any, tweedie_p=1.5, sample_weights=None
):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    ones_x = sm.add_constant(x)
    y_scale = orig_y.max() / 2
    if y_scale <= 1:
        y = orig_y
        y_scale = 1
    else:
        y = orig_y / y_scale

    if model_type == RegType.Logistic:
        # breast_cancer & linear dataset not converged using sm.Logit
        # not sure WHY, use sklearn instead.
        # sample weight function is not implemented for this model
        sample_weights = None
        sk_model = linear_model.LogisticRegression()
        sk_model.fit(x, y)
        weights = [sk_model.intercept_[0]]
        weights.extend(list(sk_model.coef_[0]))
        yhat = real_sig(np.matmul(ones_x, np.array(weights)))
        denom = 2.0 * (1.0 + np.cosh(sk_model.decision_function(x)))
        denom = np.tile(denom, (ones_x.shape[1], 1)).T
        F_ij = np.dot((ones_x / denom).T, ones_x)
        Cramer_Rao = np.linalg.pinv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = weights / sigma_estimates
        pvalues = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
    else:
        if model_type == RegType.Linear:
            model = sm.OLS(y, ones_x, freq_weights=sample_weights).fit()
        elif model_type == DistributionType.Tweedie:
            model = sm.GLM(
                y,
                ones_x,
                family=sm.families.Tweedie(var_power=tweedie_p),
                freq_weights=sample_weights,
            ).fit()
        elif model_type == DistributionType.Gamma:
            model = sm.GLM(
                y,
                ones_x,
                family=sm.families.Gamma(),
                freq_weights=sample_weights,
            ).fit()
        elif model_type == DistributionType.Poisson:
            model = sm.GLM(
                y,
                ones_x,
                family=sm.families.Poisson(),
                freq_weights=sample_weights,
            ).fit()
        elif model_type == DistributionType.Bernoulli:
            model = sm.GLM(
                y,
                ones_x,
                family=sm.families.Binomial(),
                freq_weights=sample_weights,
            ).fit()
        else:
            raise AttributeError(f"model_type {model_type} unknown")
        yhat = model.predict(ones_x)
        weights = list(model.params)
        pvalues = list(model.pvalues)

    bias = weights.pop(0)
    weights.append(bias)
    bias = pvalues.pop(0)
    pvalues.append(bias)
    yhat = yhat * y_scale
    _run_ss(
        env,
        pyus,
        x,
        orig_y,
        yhat,
        pvalues,
        weights,
        2,
        model_type,
        tweedie_p,
        y_scale,
        sample_weights=sample_weights,
    )

    if model_type == RegType.Linear:
        _run_ss(
            env,
            pyus,
            x,
            orig_y,
            yhat,
            pvalues,
            weights,
            3,
            model_type,
            tweedie_p,
            y_scale,
            sample_weights=sample_weights,
        )


@pytest.mark.mpc
def random_test_dist(prod_env_and_data, x, sample_weights, dist_type: DistributionType):
    env, data = prod_env_and_data
    n, p = x.shape

    x_one = np.concatenate([x, np.ones((n, 1))], axis=1)
    model = np.concatenate([np.random.randint(-100, 100, p), [300]]) / 100

    if dist_type == DistributionType.Gamma:
        y = np.random.gamma(shape=2, scale=1, size=n)
    elif dist_type == DistributionType.Tweedie:
        mu = np.exp(np.dot(x_one, model))
        y = tweedie(mu=mu, p=1.5, phi=20).rvs(x.shape[0])
    elif dist_type == DistributionType.Poisson:
        y = np.random.poisson(lam=2, size=n)
    elif dist_type == DistributionType.Bernoulli:
        y = np.random.binomial(1, p=0.5, size=n)
    _run_test(env, data, x, y, dist_type, sample_weights=sample_weights)


@pytest.mark.parametrize(
    "dist",
    [
        DistributionType.Tweedie,
        DistributionType.Poisson,
        DistributionType.Bernoulli,
        DistributionType.Gamma,
    ],
)
@pytest.mark.parametrize("use_sample_weights", [True, False])
@pytest.mark.mpc
def test_random_tests(prod_env_and_data, dist, use_sample_weights):
    p = 10
    n = 30000
    np.random.seed(42)
    x = np.random.rand(n, p)
    sample_weights = np.random.rand(n) if use_sample_weights else None
    random_test_dist(prod_env_and_data, x, sample_weights, dist)


@pytest.mark.mpc(parties=3)
def test_linear_ds(prod_env_and_data):
    env, data = prod_env_and_data
    ds = pd.read_csv(dataset('linear'))
    y = ds['y'].values
    x = ds.drop(['y', 'id'], axis=1).values

    _run_test(env, data, x, y, RegType.Logistic)
    _run_test(env, data, x, y, RegType.Linear)


@pytest.mark.mpc(parties=3)
def test_breast_cancer_ds(prod_env_and_data):
    env, data = prod_env_and_data
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']

    _run_test(env, data, x, y, RegType.Logistic)
    _run_test(env, data, x, y, RegType.Linear)


@pytest.mark.mpc
def test_ss_lr_logistic(prod_env_and_data):
    env, data = prod_env_and_data
    ds = pd.read_csv(dataset('linear'))
    y = ds['y'].values
    x = ds.drop(['y', 'id'], axis=1).values
    x, y = _build_splited_ds(data, x, y, 2)

    sslr = SSRegression(env.spu)
    sslr.fit(x, y, 3, 0.3, 128, 't1', 'logistic', 'l2', 0.5)
    yhat = sslr.predict(x)
    sspv = SSPValue(env.spu)
    pvalues = sspv.z_statistic_p_value(
        x,
        y,
        yhat,
        sslr.spu_w,
        LinkType.Logit,
        DistributionType.Bernoulli,
        infeed_elements_limit=1000,
    )
    print(f" test_ss_lr_logistic {pvalues}\n")


@pytest.mark.mpc
def test_ss_lr_linear(prod_env_and_data):
    env, data = prod_env_and_data
    ds = pd.read_csv(dataset('linear'))
    y = ds['y'].values
    x = ds.drop(['y', 'id'], axis=1).values
    x, y = _build_splited_ds(data, x, y, 2)

    sslr = SSRegression(env.spu)
    sslr.fit(x, y, 3, 0.3, 128, 't1', 'linear', 'l2', 0.5)
    yhat = sslr.predict(x)
    sspv = SSPValue(env.spu)
    pvalues = sspv.t_statistic_p_value(
        x, y, yhat, sslr.spu_w, infeed_elements_limit=1000
    )
    print(f" test_ss_lr_linear {pvalues}\n")
