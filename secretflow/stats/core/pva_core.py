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

# This is a single party based prediction vs actual calculation

from typing import Union

import jax.numpy as jnp
import pandas as pd


def pva(
    actual: Union[pd.DataFrame, jnp.array],
    prediction: Union[pd.DataFrame, jnp.array],
    target,
):
    """Compute Prediction Vs Actual score.

    Args:
        actual: Union[pd.DataFrame, jnp.array]

        prediction: Union[pd.DataFrame, jnp.array]

        target: numeric
            the target label in actual entries to consider.

    Returns:
        result: float
            abs(mean(prediction) - sum(actual == target)/count(actual))
    """
    if isinstance(actual, pd.DataFrame):
        actual = actual.to_numpy()
    if isinstance(prediction, pd.DataFrame):
        prediction = prediction.to_numpy()
    assert actual.size > 1, "there must be at least one actual"
    assert (
        prediction.size == actual.size
    ), "there must be at equal number of actuals and predictions"
    score_p = jnp.mean(prediction)
    score_a = jnp.sum(actual == target) / actual.size
    return jnp.abs(score_p - score_a)
