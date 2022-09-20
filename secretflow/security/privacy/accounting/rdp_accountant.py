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
from .log_utils import log_alpha

"""Implements privacy accounting for Rényi Differential Privacy.
"""


def rdp_core(q: float, noise_multiplier: float, alpha: float):
    """Comnpute RDP of the Sampled Gaussian mechanism at order alpha.

    Args:
        q: The sampling rate.
        noise_multiplier: The noise_multiplier used for calculating the std of the additive Gaussian noise.
        alpha: The order at which RDP is cald

    Returns:
        RDP at alpha, can be np.inf.
    """
    if q == 0:
        return 0

    if q == 1:
        return alpha / (2 * noise_multiplier**2)

    if np.isinf(alpha):
        return np.inf

    return log_alpha(q, noise_multiplier, alpha) / (alpha - 1)


def get_rdp(q: float, noise_multiplier: float, steps: int, orders):
    """Calculate RDP of the Sampled Gaussian Mechanism.

    Args:
        q: The sampling rate.
        noise_multiplier: The ratio of the standard deviation of the Gaussian noise
         to the l2-sensitivity of the function to which it is added
        steps: The number of steps.
        orders: An array (or a scalar) of RDP orders.

    Returns:
        The RDPs at all orders. Can be `np.inf`.
    """

    if np.isscalar(orders):
        rdp = rdp_core(q, noise_multiplier, orders)
    else:
        rdp = np.array([rdp_core(q, noise_multiplier, order) for order in orders])
    return rdp * steps


def cal_delta(orders, rdp, eps: float):
    """Calculate delta given a list of RDP values and target epsilon.
    Args:
      orders: An array (or a scalar) of orders.
      rdp: A list (or a scalar) of RDP guarantees.
      eps: The target epsilon.
    Returns:
      Pair of (delta, optimal_order).
    Raises:
      ValueError: If input is malformed.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if eps < 0:
        raise ValueError("Value of privacy loss bound epsilon must be >=0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")
    logdeltas = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")
        logdelta = 0.5 * math.log1p(-math.exp(-r))
        if a > 1.01:
            rdp_bound = (a - 1) * (r - eps + math.log1p(-1 / a)) - math.log(a)
            logdelta = min(logdelta, rdp_bound)

        logdeltas.append(logdelta)

    idx_opt = np.argmin(logdeltas)
    return min(math.exp(logdeltas[idx_opt]), 1.0), orders_vec[idx_opt]


def cal_eps(orders, rdp, delta: float):
    """Calculate epsilon given a list of RDP values and target delta.
    Args:
      orders: An array (or a scalar) of orders.
      rdp: A list (or a scalar) of RDP guarantees.
      delta: The target delta.
    Returns:
      Pair of (eps, optimal_order).
    Raises:
      ValueError: If input is malformed.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")

        if delta**2 + math.expm1(-r) >= 0:
            eps = 0
        elif a > 1.01:
            eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
        else:
            eps = np.inf
        eps_vec.append(eps)

    idx_opt = np.argmin(eps_vec)
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def get_privacy_spent_rdp(
    orders, rdp, target_eps: float = None, target_delta: float = None
):
    """Calculates delta (or eps) for given eps (or delta) from RDP values.

    Args:
        orders: An array (or a scalar) of RDP orders.
        rdp: An array of RDP values. Must be of the same length as the orders
            list.
        target_eps: If not `None`, the epsilon for which we cal the
            corresponding delta.
        target_delta: If not `None`, the delta for which we cal the
            corresponding epsilon. Exactly one of `target_eps` and
            `target_delta` must be `None`.

    Returns:
      A tuple of epsilon, delta, and the optimal order.

    Raises:
      ValueError: If target_eps and target_delta are messed up.
    """
    if target_eps is None and target_delta is None:
        raise ValueError("Exactly one out of eps and delta must be None. (Both are).")

    if target_eps is not None and target_delta is not None:
        raise ValueError("Exactly one out of eps and delta must be None. (None is).")

    if target_eps is not None:
        delta, opt_order = cal_delta(orders, rdp, target_eps)
        return target_eps, delta, opt_order
    else:
        eps, opt_order = cal_eps(orders, rdp, target_delta)
        return eps, target_delta, opt_order
