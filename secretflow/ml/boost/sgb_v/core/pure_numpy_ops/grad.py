# Copyright 2023 Ant Group Co., Ltd.
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
from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np

from .pred import sigmoid


def compute_sum_abs(y: np.ndarray) -> float:
    return np.sum(abs(y))


def compute_relative_scaling_factor(y_sum_abs: float, scaling_factor: float) -> float:
    # no need to shrink
    return min(1.0, scaling_factor / y_sum_abs)


def scale(y: Union[np.ndarray, jnp.ndarray, List, Tuple], scaling_factor: float):
    if isinstance(y, np.ndarray) or isinstance(y, jnp.ndarray):
        return y * scaling_factor
    elif isinstance(y, tuple):
        return tuple(element * scaling_factor for element in y)
    elif isinstance(y, list):
        return [element * scaling_factor for element in y]
    else:
        assert False, f"y is Unsupported type, y is type {type(y)}"


def compute_gh_linear(y: np.ndarray, pred: np.ndarray):
    return pred - y, np.ones(pred.shape)


def compute_gh_logistic(y: np.ndarray, pred: np.ndarray):
    yhat = sigmoid(pred)
    return yhat - y, yhat * (1 - yhat)


def split_GH(x) -> Tuple[np.ndarray, np.ndarray]:
    return x[:, 0].reshape(1, -1), x[:, 1].reshape(1, -1)
