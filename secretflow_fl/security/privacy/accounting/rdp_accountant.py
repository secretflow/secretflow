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

from typing import List, Optional, Union

import dp_accounting.rdp.rdp_privacy_accountant as rdp_accountant
import numpy as np

"""
Implements privacy accounting for Rényi Differential Privacy.
Wrapper of the dp-accounting library.
"""

DEFAULT_RDP_ORDERS = rdp_accountant.DEFAULT_RDP_ORDERS


def rdp_core(q: float, noise_multiplier: float, alpha: float):
    """Compute RDP of the Sampled Gaussian mechanism at order alpha.

    Args:
        q: The sampling rate.
        noise_multiplier: A non-negative float representing the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of the
      function to which it is added.
        alpha: The order at which RDP is called

    Returns:
        RDP at alpha, can be np.inf.
    """
    # dp_accounting has the wrong type hint for the return value, so we suppress the type check here.
    return rdp_accountant._compute_rdp_poisson_subsampled_gaussian(  # type: ignore
        q, noise_multiplier, [alpha]
    )[0]


def get_rdp(
    q: float,
    noise_multiplier: float,
    steps: int,
    orders: Union[float, List[float]],
):
    """Calculate RDP of the Sampled Gaussian Mechanism.

    Args:
        q: The sampling rate.
        noise_multiplier: A non-negative float representing the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of the
      function to which it is added.
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


def cal_delta(
    orders: Union[float, List[float]], rdp: Union[float, List[float]], eps: float
):
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

    # we force all r in rdp_vec to be >= 0, although dp_accounting does not require this
    if np.any(rdp_vec < 0):
        raise ValueError("Renyi divergence must be >=0.")

    return rdp_accountant.compute_delta(orders_vec, rdp_vec, eps)


def cal_eps(
    orders: Union[float, List[float]], rdp: Union[float, List[float]], delta: float
):
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

    # we force delta to be > 0, although dp_accounting does admit delta = 0
    if delta <= 0:
        raise ValueError("Privacy failure probability bound delta must be >0.")
    # we force all r in rdp_vec to be >= 0, although dp_accounting does not require this
    if np.any(rdp_vec < 0):
        raise ValueError("Renyi divergence must be >=0.")

    return rdp_accountant.compute_epsilon(orders_vec, rdp_vec, delta)


def get_privacy_spent_rdp(
    orders: Union[float, List[float]],
    rdp: Union[float, List[float]],
    target_eps: Optional[float] = None,
    target_delta: Optional[float] = None,
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
