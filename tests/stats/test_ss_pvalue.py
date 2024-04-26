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


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
    pyus_dict = {
        sf_production_setup_devices.alice: 1,
        sf_production_setup_devices.bob: 1,
        sf_production_setup_devices.carol: 1,
        sf_production_setup_devices.davy: 1,
    }
    pyus = list(pyus_dict.keys())

    yield sf_production_setup_devices, pyus


def _build_splited_ds(pyus, x, cols, parties):
    assert x.shape[1] >= parties
    assert len(pyus) >= parties
    step = math.ceil(x.shape[1] / parties)
    fed_x = VDataFrame({})
    for r in range(parties):
        start = r * step
        end = start + step if r != parties - 1 else x.shape[1]
        split_x = x[:, start:end]
        pyu_x = pyus[r](lambda: pd.DataFrame(split_x))()
        fed_x.partitions[pyus[r]] = partition(data=pyu_x)
    if isinstance(cols, list):
        fed_cs = []
        for c in cols:
            pyu_c = pyus[parties - 1](lambda: pd.DataFrame(c))()
            fed_c = VDataFrame({pyus[parties - 1]: partition(data=pyu_c)})
            fed_cs.append(fed_c)
        return fed_x, fed_cs
    else:
        pyu_c = pyus[parties - 1](lambda: pd.DataFrame(cols))()
        fed_c = VDataFrame({pyus[parties - 1]: partition(data=pyu_c)})
        return fed_x, fed_c


def _run_ss(env, pyus, x, y, yhat, p, w, parties, model_type: Any, tweedie_p):
    # weights to spu
    pyu_w = env.alice(lambda: np.array(w))()
    spu_w = pyu_w.to(env.spu)
    pyu_yhat = env.alice(lambda: np.array(yhat))()
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

        pvalues = sspv.z_statistic_p_value(
            pyu_x,
            pyu_y,
            spu_yhat,
            spu_w,
            link,
            model_type,
            tweedie_power=tweedie_p,
        )

    p = np.array(p)
    abs_err = np.absolute(pvalues - p)
    radio_err = abs_err / np.maximum(pvalues, p)

    logging.warn(f"radio_err .... \n{radio_err}")

    # for pvalue < 0.2, check abs err < 0.01
    abs_assert = np.select([p < 0.2], [abs_err], 0)
    assert np.amax(abs_assert) < 0.01, f"\n{abs_assert}"
    # else check radio error < 20%
    radio_assert = np.select([p >= 0.2], [radio_err], 0)
    assert np.amax(radio_assert) < 0.2, f"\n{radio_err}"


def _run_test(env, pyus, x, y, model_type: Any, tweedie_p=1.5):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    ones_x = sm.add_constant(x)

    if model_type == RegType.Logistic:
        # breast_cancer & linear dataset not converged using sm.Logit
        # not sure WHY, use sklearn instead.
        sk_model = linear_model.LogisticRegression()
        sk_model.fit(x, y)
        weights = [sk_model.intercept_[0]]
        weights.extend(list(sk_model.coef_[0]))
        yhat = real_sig(np.matmul(ones_x, np.array(weights)))
        denom = 2.0 * (1.0 + np.cosh(sk_model.decision_function(x)))
        denom = np.tile(denom, (ones_x.shape[1], 1)).T
        F_ij = np.dot((ones_x / denom).T, ones_x)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = weights / sigma_estimates
        pvalues = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
    else:
        if model_type == RegType.Linear:
            model = sm.OLS(y, ones_x).fit()
        elif model_type == DistributionType.Tweedie:
            model = sm.GLM(
                y, ones_x, family=sm.families.Tweedie(var_power=tweedie_p)
            ).fit()
        elif model_type == DistributionType.Gamma:
            model = sm.GLM(y, ones_x, family=sm.families.Gamma()).fit()
        elif model_type == DistributionType.Poisson:
            model = sm.GLM(y, ones_x, family=sm.families.Poisson()).fit()
        elif model_type == DistributionType.Bernoulli:
            model = sm.GLM(y, ones_x, family=sm.families.Binomial()).fit()
        else:
            raise AttributeError(f"model_type {model_type} unknown")
        yhat = model.predict(ones_x)
        weights = list(model.params)
        pvalues = list(model.pvalues)

    bias = weights.pop(0)
    weights.append(bias)
    bias = pvalues.pop(0)
    pvalues.append(bias)
    _run_ss(env, pyus, x, y, yhat, pvalues, weights, 2, model_type, tweedie_p)

    if model_type == RegType.Linear:
        _run_ss(env, pyus, x, y, yhat, pvalues, weights, 3, model_type, tweedie_p)


def test_poisson_ds(prod_env_and_data):
    env, data = prod_env_and_data

    p = 10
    n = 300
    np.random.seed(42)
    x = np.random.rand(n, p)
    y = np.random.poisson(lam=2, size=n)

    _run_test(env, data, x, y, DistributionType.Poisson)


def test_binomial_ds(prod_env_and_data):
    env, data = prod_env_and_data

    p = 10
    n = 300
    np.random.seed(42)
    x = np.random.rand(n, p)
    y = np.random.binomial(1, p=0.5, size=n)

    _run_test(env, data, x, y, DistributionType.Bernoulli)


def test_gamma_ds(prod_env_and_data):
    env, data = prod_env_and_data

    p = 10
    n = 300
    np.random.seed(42)
    x = np.random.rand(n, p)
    y = np.random.gamma(shape=2, scale=1, size=n)

    _run_test(env, data, x, y, DistributionType.Gamma)


def test_tweedie_ds(prod_env_and_data):
    env, data = prod_env_and_data

    p = 10
    n = 300
    np.random.seed(42)
    x = np.random.rand(n, p)
    x_one = np.concatenate([x, np.ones((n, 1))], axis=1)
    model = np.concatenate([np.random.randint(-100, 100, p), [300]]) / 100
    mu = np.exp(np.dot(x_one, model))
    y = tweedie(mu=mu, p=1.5, phi=20).rvs(n)

    _run_test(env, data, x, y, DistributionType.Tweedie)


def test_linear_ds(prod_env_and_data):
    env, data = prod_env_and_data
    ds = pd.read_csv(dataset('linear'))
    y = ds['y'].values
    x = ds.drop(['y', 'id'], axis=1).values

    _run_test(env, data, x, y, RegType.Logistic)
    _run_test(env, data, x, y, RegType.Linear)


def test_breast_cancer_ds(prod_env_and_data):
    env, data = prod_env_and_data
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']

    _run_test(env, data, x, y, RegType.Logistic)
    _run_test(env, data, x, y, RegType.Linear)


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
