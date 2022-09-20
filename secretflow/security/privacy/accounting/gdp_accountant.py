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

import numpy as np
from scipy import optimize
from scipy.stats import norm

"""Implements privacy accounting for Gaussian Differential Privacy.
"""


def cal_mu_uniform(step: int, noise_multiplier, num_samples, batch_size):
    """Calculate mu from uniform subsampling.

    Args:
        step: The current step of model training or prediction.
        epnoise_multipliers: Epsilon for pure DP.
        batch_size: Batch size.
        num_samples: Number of all samples.
    """
    c = batch_size * np.sqrt(step) / num_samples
    return (
        np.sqrt(2)
        * c
        * np.sqrt(
            np.exp(noise_multiplier ** (-2)) * norm.cdf(1.5 / noise_multiplier)
            + 3 * norm.cdf(-0.5 / noise_multiplier)
            - 2
        )
    )


def cal_mu_poisson(step: int, noise_multiplier, num_samples, batch_size):
    """Calculate mu from Poisson subsampling.

    Args:
        step: The current step of model training or prediction.
        epnoise_multipliers: Epsilon for pure DP.
        batch_size: Batch size.
        num_samples: Number of all samples.
    """
    return (
        np.sqrt(np.exp(noise_multiplier ** (-2)) - 1)
        * np.sqrt(step)
        * batch_size
        / num_samples
    )


def delta_gaussian(eps, mu):
    """Calculate dual between mu-GDP and (epsilon, delta)-DP.

    Args:
        mu: The parameters of the GDP.
        eps: The parameters of the (epsilon, delta)-DP.

    """
    return norm.cdf(mu / 2 - eps / mu) - np.exp(eps) * norm.cdf(-mu / 2 - eps / mu)


def get_eps_from_mu(mu, delta):
    """Get epsilon from mu given delta via inverse dual.

    Args:
        mu: The parameters of the GDP.
        delta: The parameters of the (epsilon, delta)-DP.
    """

    if delta >= delta_gaussian(0, mu):
        return 0

    def f(x):
        return delta_gaussian(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0.0, 500.0], method='brentq').root
