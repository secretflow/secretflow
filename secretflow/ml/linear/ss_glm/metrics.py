# Copyright 2024 Ant Group Co., Ltd.
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


# metrics unique to glm
import jax.numpy as jnp

from secretflow.stats.core.metrics import (
    mean_squared_error,
    roc_auc_score,
    root_mean_squared_error,
)

from .core import Distribution


def deviance(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, w: jnp.ndarray, dist: Distribution
) -> jnp.ndarray:
    return dist.deviance(y_pred.reshape(-1), y_true.reshape(-1), w)


SUPPORTED_METRICS = {
    "deviance": deviance,
    "MSE": mean_squared_error,
    "RMSE": root_mean_squared_error,
    "AUC": roc_auc_score,
}


# a metric is increasing if the larger metric the better
def increasing_better_metric(a, b):
    return a >= b


# a metric is decreasing if the smaller metric the better
def decreasing_better_metric(a, b):
    return a <= b


BETTER_DEF = {
    "deviance": decreasing_better_metric,
    "MSE": decreasing_better_metric,
    "RMSE": decreasing_better_metric,
    "AUC": increasing_better_metric,
}


# a metric is positve if range is in [0, inf)
# a metric is negative if range is in (-inf, 0]


# example: AUC
def improve_ratio_for_positive_increasing_metric(old_metric, new_metric):
    return (new_metric - old_metric) / old_metric


# examples: RMSE, MSE, deviance
def improve_ratio_for_positive_decreasing_metric(old_metric, new_metric):
    return -(new_metric - old_metric) / old_metric


IMPROVE_DEF = {
    "deviance": improve_ratio_for_positive_decreasing_metric,
    "MSE": improve_ratio_for_positive_decreasing_metric,
    "RMSE": improve_ratio_for_positive_decreasing_metric,
    "AUC": improve_ratio_for_positive_increasing_metric,
}
