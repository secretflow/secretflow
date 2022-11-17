# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

# This is a single party based population stability index calculation

from typing import Union

import jax.numpy as jnp
import pandas as pd


def psi_index(a, b):
    """(a - b) * ln(a/b).

    Args:
        a: array or float

        b: array or float
            a, b must be of same type.
            They can be float or jnp.array or np.array.
    Returns:
        result: array or float
            same type as a or b.
    """
    return (a - b) * jnp.log(a / b)


def psi_score(A: jnp.array, B: jnp.array):
    """Computes the psi score.

    Args:
        A: jnp.array
            Distribution of sample A
        B: jnp.array
            Distribution of sample B
    Returns:
        result: float
    """
    index_arr = psi_index(A, B)
    return jnp.sum(index_arr)


def distribution_generation(X: jnp.array, split_points: jnp.array):
    """Generate a distribution of X according to split points.

    Args:
        X: jnp.array
            a collection of samples
        split_points: jnp.array
            an ordered sequence of split points
    Returns:
        dist_X: jnp.array
            distribution in forms of percentage of counts in each bin.
            bin[0] is [split_points[0], split_points[1])
    """
    assert split_points.size > 1, "there must be at least one bin"
    assert X.size > 1, "there must be at least one sample"
    result, _ = jnp.histogram(X, bins=split_points, density=False)
    return result / X.size


def psi(
    X: Union[pd.DataFrame, jnp.array],
    Y: Union[pd.DataFrame, jnp.array],
    split_points: jnp.array,
):
    """Calculate population stability index.

    Args:
        X: Union[pd.DataFrame, jnp.array]
            a collection of samples
        Y: Union[pd.DataFrame, jnp.array]
            a collection of samples
        split_points: jnp.array
            an ordered sequence of split points
    Returns:
        result: float
            population stability index
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    dist_x = distribution_generation(X, split_points)
    dist_y = distribution_generation(Y, split_points)
    return psi_score(dist_x, dist_y)
