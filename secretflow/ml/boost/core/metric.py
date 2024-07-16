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

from typing import Callable, Tuple, Union

import numpy as np

from secretflow.device import PYUObject, reveal
from secretflow.ml.boost.core.callback import VData
from secretflow.ml.linear.ss_glm.core.distribution import DistributionTweedie
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


def tweedie_deviance_producer(tweedie_variance_power: float):
    # note that the unlike GLM, the scale does not influence the training process in any way,
    # since none of g, h and prediction depends on it
    # so we can fix the scale to be 1
    dist = DistributionTweedie(1, tweedie_variance_power)
    return lambda y_true, y_pred: dist.deviance(y_true, y_pred)


def tweedie_negative_log_likelihood(tweedie_variance_power: float):
    rho = tweedie_variance_power

    def f(y, p):
        y = y.flatten()
        p = p.flatten()
        log_p = np.log(p)

        one_minus_rho = 1 - rho
        two_minus_rho = 2 - rho
        a = y * np.exp(one_minus_rho * log_p) / one_minus_rho
        b = np.exp(two_minus_rho * log_p) / two_minus_rho
        return np.sum(-a + b)

    return f


_METRICS = {
    'roc_auc': metric_wrapper(roc_auc_score, 'roc_auc'),
    'rmse': metric_wrapper(root_mean_squared_error, 'rmse'),
    'mse': metric_wrapper(mean_squared_error, 'mse'),
}


def MetricProducer(metric_name: str, **kwargs) -> Tuple[Union[Callable, None], str]:
    if metric_name == 'tweedie_deviance':
        assert (
            'tweedie_variance_power' in kwargs
        ), f'tweedie_deviance requires tweedie_variance_power'
        tweedie_variance_power = kwargs['tweedie_variance_power']
        return (
            metric_wrapper(
                tweedie_deviance_producer(tweedie_variance_power),
                f'tweedie_deviance_{tweedie_variance_power}',
            ),
            f'tweedie_deviance_{tweedie_variance_power}',
        )
    elif metric_name == 'tweedie_nll':
        assert (
            'tweedie_variance_power' in kwargs
        ), f'tweedie_nll requires tweedie_variance_power'
        tweedie_variance_power = kwargs['tweedie_variance_power']
        return (
            metric_wrapper(
                tweedie_negative_log_likelihood(tweedie_variance_power),
                f'tweedie_nll_{tweedie_variance_power}',
            ),
            f'tweedie_nll_{tweedie_variance_power}',
        )
    else:
        return _METRICS.get(metric_name, None), metric_name


ALL_METRICS_NAMES = list(_METRICS.keys()) + ["tweedie_deviance", "tweedie_nll"]
