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

# metrics that can be calculated using jax

import jax.numpy as jnp

from secretflow.data.ndarray import mean_of_difference_squares
from secretflow.stats.core.biclassification_eval_core import (
    binary_roc_auc,
    create_sorted_label_score_pair,
)


def roc_auc_score(y_true: jnp.array, y_pred: jnp.array) -> jnp.float32:
    sorted_label_score_pair_arr = create_sorted_label_score_pair(y_true, y_pred)
    auc = binary_roc_auc(sorted_label_score_pair_arr)
    return auc


def mean_squared_error(y_true: jnp.array, y_pred: jnp.array) -> jnp.float32:
    return mean_of_difference_squares(y_true, y_pred)


def root_mean_squared_error(y_true: jnp.array, y_pred: jnp.array) -> jnp.float32:
    return jnp.sqrt(mean_of_difference_squares(y_true, y_pred))
