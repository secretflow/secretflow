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


from typing import Callable, Tuple

from secretflow.device import PYUObject, reveal
from secretflow.ml.boost.core.callback import VData
from secretflow.stats.core.metrics import (
    mean_squared_error,
    roc_auc_score,
    root_mean_squared_error,
)
from secretflow.utils.consistent_ops import cast_float

Metric = Callable[[PYUObject, VData], Tuple[str, float]]


# all metrics are revealed to allow ML engineer to the the information
def metric_wrapper(metric: Callable, metric_name: str):
    def wrapped_metric(y_true: VData, y_pred: PYUObject) -> Tuple[str, float]:
        assert len(y_true.partitions) == 1
        y_true_object = list(y_true.partitions.values())[0]
        return metric_name, cast_float(
            reveal(y_pred.device(metric)(y_true_object, y_pred))
        )

    return wrapped_metric


METRICS = {
    'roc_auc': metric_wrapper(roc_auc_score, 'roc_auc'),
    'rmse': metric_wrapper(root_mean_squared_error, 'rmse'),
    'mse': metric_wrapper(mean_squared_error, 'mse'),
}
