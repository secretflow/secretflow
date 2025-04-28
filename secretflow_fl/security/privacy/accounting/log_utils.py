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

import math

import numpy as np
import six
from scipy import special

"""
LOG-SPACE ARITHMETIC
"""


def add_log(logx, logy):
    """Add two numbers in the log space."""
    x, y = min(logx, logy), max(logx, logy)
    if x == -np.inf:
        return y
    return math.log1p(math.exp(x - y)) + y


def sub_log(logx, logy):
    """Subtract two numbers in the log space. The return value must be non-negative."""
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:
        return logx
    if logx == logy:
        return -np.inf

    try:
        return math.log(math.expm1(logx - logy)) + logy
    except OverflowError:
        return logx


def erfc_log(x):
    """Calculate log(erfc(x)) with high accuracy for large x."""
    try:
        return math.log(2) + special.log_ndtr(-x * 2**0.5)
    except NameError:
        r = special.erfc(x)
        if r == 0.0:
            return (
                -math.log(math.pi) / 2
                - math.log(x)
                - x**2
                - 0.5 * x**-2
                + 0.625 * x**-4
                - 37.0 / 24.0 * x**-6
                + 353.0 / 64.0 * x**-8
            )
        else:
            return math.log(r)


def comb_log(n, k):
    return special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)


def log_alpha_int(q, sigma, alpha):
    """Calculate log(A_alpha) for integer alpha. 0 < q < 1."""
    assert isinstance(alpha, six.integer_types)

    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
            comb_log(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q)
        )

        s = log_coef_i + (i * i - i) / (2 * (sigma**2))
        log_a = add_log(log_a, s)

    return float(log_a)


def log_alpha_frac(q, sigma, alpha):
    """Calculate log(A_alpha) for fractional alpha. 0 < q < 1."""
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma**2 * math.log(1 / q - 1) + 0.5

    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + erfc_log((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + erfc_log((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

        if coef > 0:
            log_a0 = add_log(log_a0, log_s0)
            log_a1 = add_log(log_a1, log_s1)
        else:
            log_a0 = sub_log(log_a0, log_s0)
            log_a1 = sub_log(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return add_log(log_a0, log_a1)


def log_alpha(q, sigma, alpha):
    """Calculate log(A_alpha) for any positive finite alpha."""
    if float(alpha).is_integer():
        return log_alpha_int(q, sigma, int(alpha))
    else:
        return log_alpha_frac(q, sigma, alpha)
