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
import jax.numpy as jnp


def newton_matrix_inverse(x: np.ndarray, iter_round: int = 20):
    """
    computing the inverse of a matrix by newton iteration.
    https://aalexan3.math.ncsu.edu/articles/mat-inv-rep.pdf
    """
    assert x.shape[0] == x.shape[1], "x need be a (n x n) matrix"
    E = jnp.identity(x.shape[0])
    a = (1 / jnp.trace(x)) * E
    for _ in range(iter_round):
        a = jnp.matmul(a, (2 * E - jnp.matmul(x, a)))
    return a


def equal_obs(x, n_bin):
    """
    Equal Frequency Split Point Search in x with bin size = n_bins
    In each bin, there is equal number of points in them

    Args:
        x: array
        n_bin: int

    Returns:
        jnp.array with size n_bin+1
    """
    n_len = len(x)
    return jnp.interp(
        x=jnp.linspace(0, n_len, n_bin + 1), xp=jnp.arange(n_len), fp=jnp.sort(x)
    )


def equal_range(x, n_bin):
    """
    Equal Range Search Split Points in x with bin size = n_bins
    Returns:
        jnp.array with size n_bin+1
    """
    min_val = jnp.min(x)
    max_val = jnp.max(x)
    segment = (max_val - min_val) / n_bin
    return jnp.array([min_val + i * segment for i in range(n_bin + 1)])
